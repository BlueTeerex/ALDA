import streamlit as st
import numpy as np
import ephem
import pandas as pd
from astropy.time import Time
import matplotlib.pyplot as plt
from math import cos, sin, pi, acos, degrees, radians

# --- 1. 論文原始計算邏輯 (核心重構) ---
class AsteroidEngine:
    def __init__(self, oe_line):
        self.oe_line = oe_line

    def get_metrics(self, jd):
        # 轉換時間
        date = Time(jd, format='jd').datetime
        
        # 初始化天體
        ast = ephem.readdb(self.oe_line)
        ast.compute(date)
        sun = ephem.Sun(date)
        
        # 獲取黃道座標 (Heliocentric Ecliptic)
        # s_dist: r (太陽到小行星), e_dist: Δ (地球到小行星)
        r = ast.sun_distance
        delta = ast.earth_distance
        R = sun.earth_distance # 太陽到地球距離
        
        # --- 根據論文 3.4 & 3.5 節的向量邏輯計算 ---
        # 使用餘弦定理計算相位角 Alpha (太陽-小行星-地球 夾角)
        # 公式: cos(alpha) = (r^2 + delta^2 - R^2) / (2 * r * delta)
        cos_alpha = (r**2 + delta**2 - R**2) / (2 * r * delta)
        cos_alpha = max(-1, min(1, cos_alpha)) # 防止數值溢出
        phase_angle = degrees(acos(cos_alpha))
        
        # 使用餘弦定理計算距角 Elongation (太陽-地球-小行星 夾角)
        # 公式: cos(elong) = (R^2 + delta^2 - r^2) / (2 * R * delta)
        cos_elong = (R**2 + delta**2 - r**2) / (2 * R * delta)
        cos_elong = max(-1, min(1, cos_elong))
        elongation = degrees(acos(cos_elong))
        
        return {
            'Date': date,
            'Phase': phase_angle,
            'Elongation': elongation,
            'r': r,
            'delta': delta
        }

# --- 2. 數據與設定 ---
PAPER_DATA = {
    "162173 Ryugu": "162173 Ryugu,e,5.86663,251.29446,211.61035,1.1910091,0,0.19111632,327.3279370,5/31.0/2020,2000,H19.55,0.15",
    "101955 Bennu": "101955 Bennu,e,6.03494,2.06087,66.22307,1.1259673,0,0.20374511,101.7039655,5/31.0/2020,2000,H20.45,0.15"
}

# --- 3. 網頁介面 ---
st.set_page_config(page_title="ALDA - 觀測預測系統", layout="wide")
st.title("☄️ ALDA 小行星觀測視窗預測 (修復版)")

with st.sidebar:
    st.header("⚙️ 設置")
    target = st.selectbox("選擇目標", list(PAPER_DATA.keys()))
    start_year = st.number_input("預測起始年份", value=2025)
    years = st.slider("預測年數", 1, 25, 20)
    
    st.write("---")
    st.subheader("📊 論文過濾條件")
    m_phase = st.slider("最大相位角 (Phase < α)", 0, 180, 30)
    m_elong = st.slider("最小距角 (Elongation > θ)", 0, 180, 90)

if st.button("🔍 開始搜尋最佳視窗"):
    engine = AsteroidEngine(PAPER_DATA[target])
    
    # 建立時間序列
    start_jd = Time(f"{start_year}-01-01").jd
    end_jd = start_jd + (years * 365)
    jd_steps = np.arange(start_jd, end_jd, 2) # 每兩天算一次
    
    data_list = []
    with st.spinner("正在進行天文幾何計算..."):
        for jd in jd_steps:
            res = engine.get_metrics(jd)
            # 判斷是否符合條件
            res['is_valid'] = (res['Phase'] < m_phase) and (res['Elongation'] > m_elong)
            data_list.append(res)
    
    full_df = pd.DataFrame(data_list)
    valid_df = full_df[full_df['is_valid']]

    if not valid_df.empty:
        # --- 視窗合併邏輯 ---
        st.header(f"📅 {target} 的最佳觀測視窗")
        
        valid_df = valid_df.copy()
        valid_df['group'] = (valid_df['Date'].diff().dt.days > 5).cumsum()
        
        windows = []
        for _, group in valid_df.groupby('group'):
            start_d = group['Date'].iloc[0]
            end_d = group['Date'].iloc[-1]
            duration = (end_d - start_d).days
            if duration > 3: # 忽略太短的視窗
                st.success(f"✅ **{start_d.strftime('%Y-%m-%d')} 至 {end_d.strftime('%Y-%m-%d')}** (持續 {duration} 天)")
        
        # --- 圖表展示 ---
        st.subheader("觀測參數變化趨勢")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(full_df['Date'], full_df['Phase'], label='Phase Angle (α)', color='orange', alpha=0.6)
        ax.plot(full_df['Date'], full_df['Elongation'], label='Elongation (θ)', color='blue', alpha=0.6)
        
        # 標示出符合條件的區域
        ax.fill_between(full_df['Date'], 0, 180, where=full_df['is_valid'], color='green', alpha=0.2, label='Best Window')
        
        ax.axhline(y=m_phase, color='red', linestyle='--', label='Max Phase')
        ax.axhline(y=m_elong, color='navy', linestyle='--', label='Min Elong')
        
        ax.set_ylim(0, 180)
        ax.set_ylabel("Degrees")
        ax.legend(loc='upper right')
        st.pyplot(fig)
        
    else:
        st.error("❌ 找不到符合條件的觀測視窗。")
        st.info("💡 提示：Ryugu 的軌道很特殊，請嘗試將相位角放寬到 40，距角放低到 70 看看是否有數據產出。")
