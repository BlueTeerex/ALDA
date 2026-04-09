import streamlit as st
import numpy as np
import ephem
import pandas as pd
from astropy.time import Time
import matplotlib.pyplot as plt
from math import acos, degrees

# --- 1. 高精度科研引擎 (同步盧教授 20260303.py 向量邏輯) ---
class ALDAEngine:
    def __init__(self, oe_line):
        self.oe_line = oe_line

    def get_metrics(self, jd):
        date = Time(jd, format='jd').datetime
        ast = ephem.readdb(self.oe_line)
        ast.compute(date)
        sun = ephem.Sun(date)
        
        # r: 太陽到小行星距離, delta: 地球到小行星距離, R: 太陽到地球距離
        r = ast.sun_distance
        delta = ast.earth_distance
        R = sun.earth_distance 
        
        # --- 論文 3.4 節：相位角 (α) 計算 ---
        cos_alpha = (r**2 + delta**2 - R**2) / (2 * r * delta)
        phase_angle = degrees(acos(max(-1, min(1, cos_alpha))))
        
        # --- 論文 3.4 節：距角 (θ) 計算 ---
        # 根據向量 ab (小行星到地球) 和 cb (太陽到地球) 的夾角關係
        cos_theta = (R**2 + delta**2 - r**2) / (2 * R * delta)
        elongation = degrees(acos(max(-1, min(1, cos_theta))))
        
        return {
            'Date': date, 
            'Phase': phase_angle, 
            'Elongation': elongation, 
            'JD': jd
        }

# 論文精確軌道根數
PAPER_ASTEROIDS = {
    "162173 Ryugu": "162173 Ryugu,e,5.86663,251.29446,211.61035,1.1910091,0,0.19111632,327.3279370,5/31.0/2020,2000,H19.55,0.15",
    "101955 Bennu": "101955 Bennu,e,6.03494,2.06087,66.22307,1.1259673,0,0.20374511,101.7039655,5/31.0/2020,2000,H20.45,0.15"
}

# --- 2. 完整論文描述與多語言字典 (內容絕不刪減) ---
LANG = {
    "zh": {
        "title": "☄️ ALDA: 小行星光變數據擴增系統",
        "lang_btn": "English Interface",
        "about_tab": "關於 ALDA",
        "calc_tab": "觀測預測",
        "val_tab": "模型驗證",
        "why_title": "💡 為什麼會有這個網站？",
        "why_text": "小行星的形狀重構和物理性質研究高度依賴「光變曲線」。然而，許多小行星在特定的 PABS 緯度區間缺乏數據，導致模型不夠準確。ALDA 旨在幫助全球天文愛好者精確找到最佳觀測時段，填補這些數據缺口。",
        "func_title": "🛠️ 網站功能",
        "func_text": "* **自動計算幾何參數**：基於論文公式計算相位角、距角等關鍵天文數據。\n* **精確視窗預測**：過濾出符合科學觀測條件（Phase < 30°, Elongation > 90°）的時段。\n* **國際協作**：讓全球業餘觀測者能共同參與科學任務，為專業研究提供支持。",
        "how_title": "📖 如何使用",
        "how_text": "1. 在左側選單選擇目標小行星（如 Ryugu 或 Bennu）。\n2. 設定你想要預測的起始年份與持續年數。\n3. 點擊「開始分析」，系統將根據盧教授高精度算法列出建議時段。",
        "val_title": "🔍 模型準確性驗證 (Validation)",
        "val_desc": "根據本論文第五章，我們將本預測模型與 ALCDEF (Asteroid Lightcurve Photometry Database) 資料庫進行了比對驗證。",
        "val_table_p": "參數",
        "val_table_e": "驗證平均誤差",
        "val_table_s": "數據來源",
        "settings": "⚙️ 參數設定",
        "target": "選擇目標小行星",
        "start_year": "預測起始年份",
        "years": "預測年數",
        "run_btn": "🚀 開始執行高精度分析",
        "result_title": "📅 建議觀測時段 (符合科學條件)",
        "no_result": "❌ 找不到符合條件的觀測視窗。",
        "chart_title": "觀測參數趨勢圖 (α & θ)",
        "school": "製作單位：澳門濠江中學附屬英才學校 學生團隊",
        "copy": "Copyright © 2026 ALDA Project. All Rights Reserved."
    },
    "en": {
        "title": "☄️ ALDA: Asteroid Lightcurve Data Augmentor",
        "lang_btn": "中文介面",
        "about_tab": "About ALDA",
        "calc_tab": "Prediction",
        "val_tab": "Validation",
        "why_title": "💡 Why ALDA?",
        "why_text": "Asteroid physical modeling relies heavily on lightcurve data. Many asteroids lack data at specific PABS latitudes, leading to inaccurate models. ALDA helps amateur astronomers find optimal observation windows to fill these scientific gaps.",
        "func_title": "🛠️ Functions",
        "func_text": "* **Geometric Calculation**: Auto-calculate Phase Angle, Elongation based on academic models.\n* **Window Prediction**: Filter time slots meeting scientific criteria (Phase < 30°, Elongation > 90°).\n* **Global Collaboration**: Enabling amateurs worldwide to contribute to professional research.",
        "how_title": "📖 How to Use",
        "how_text": "1. Select a target asteroid (e.g., Ryugu or Bennu) from the sidebar.\n2. Set the prediction start year and duration.\n3. Click 'Run Analysis' to generate recommended observation dates.",
        "val_title": "🔍 Model Validation",
        "val_desc": "As per Chapter 5 of our research paper, the prediction model has been validated against the ALCDEF database.",
        "val_table_p": "Parameter",
        "val_table_e": "Mean Error",
        "val_table_s": "Source",
        "settings": "⚙️ Settings",
        "target": "Select Target",
        "start_year": "Start Year",
        "years": "Duration (Years)",
        "run_btn": "🚀 Run High-Precision Analysis",
        "result_title": "📅 Recommended Windows",
        "no_result": "❌ No windows found matching the criteria.",
        "chart_title": "Observation Trends (α & θ)",
        "school": "Produced by students of Premier School Affiliated to Hou Kong Middle School (Macau)",
        "copy": "Copyright © 2026 ALDA Project. All Rights Reserved."
    }
}

# --- 3. UI 邏輯控制 ---
if 'lang' not in st.session_state: st.session_state.lang = 'zh'
def toggle_lang(): st.session_state.lang = 'en' if st.session_state.lang == 'zh' else 'zh'
l = LANG[st.session_state.lang]

st.set_page_config(page_title="ALDA - Asteroid High Precision", layout="wide")

# 頁首與切換按鈕
h_col1, h_col2 = st.columns([8, 2])
h_col1.title(l["title"])
h_col2.button(l["lang_btn"], on_click=toggle_lang)

tab_about, tab_calc, tab_val = st.tabs([l["about_tab"], l["calc_tab"], l["val_tab"]])

with tab_about:
    st.header(l["why_title"])
    st.write(l["why_text"])
    col_f, col_h = st.columns(2)
    with col_f:
        st.header(l["func_title"])
        st.write(l["func_text"])
    with col_h:
        st.header(l["how_title"])
        st.write(l["how_text"])
    st.info(f"🏫 {l['school']}")

with tab_val:
    st.header(l["val_title"])
    st.write(l["val_desc"])
    st.table(pd.DataFrame({
        l["val_table_p"]: ["Phase Angle (α)", "Observation Window", "Database"],
        l["val_table_e"]: ["± 0.42°", "± 2.5 Days", "Validated"],
        l["val_table_s"]: ["ALCDEF / JPL", "ALCDEF", "Chapter 5 Paper"]
    }))

with tab_calc:
    with st.sidebar:
        st.header(l["settings"])
        target_name = st.selectbox(l["target"], list(PAPER_ASTEROIDS.keys()))
        s_year = st.number_input(l["start_year"], value=2025)
        duration_val = st.slider(l["years"], 1, 25, 20)
        st.write("---")
        st.caption("Algorithm: LUXP High-Precision Model")

    if st.button(l["run_btn"]):
        engine = ALDAEngine(PAPER_ASTEROIDS[target_name])
        start_jd = Time(f"{s_year}-01-01").jd
        jd_steps = np.arange(start_jd, start_jd + (duration_val * 365), 2)
        
        # 遍歷計算
        results = []
        for jd in jd_steps:
            results.append(engine.get_metrics(jd))
        
        df = pd.DataFrame(results)
        # 嚴格執行論文約束條件：α < 30°, θ > 90°
        valid_df = df[(df['Phase'] < 30) & (df['Elongation'] > 90)].copy()

        if not valid_df.empty:
            st.header(l["result_title"])
            valid_df['group'] = (valid_df['Date'].diff().dt.days > 10).cumsum()
            for _, gp in valid_df.groupby('group'):
                s_d, e_d = gp['Date'].iloc[0], gp['Date'].iloc[-1]
                st.success(f"✅ **{s_d.strftime('%Y-%m-%d')} ~ {e_d.strftime('%Y-%m-%d')}**")
            
            # 趨勢圖 (全英文標籤避開亂碼)
            st.subheader(l["chart_title"])
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df['Date'], df['Phase'], label="α (Phase)", color='orange', alpha=0.8)
            ax.plot(df['Date'], df['Elongation'], label="θ (Elongation)", color='blue', alpha=0.8)
            ax.fill_between(df['Date'], 0, 180, where=(df['Phase']<30)&(df['Elongation']>90), 
                            color='green', alpha=0.2, label="Best Window")
            ax.set_ylim(0, 180)
            ax.set_ylabel("Degrees")
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.warning(l["no_result"])

# --- 4. 底部版權宣告 ---
st.markdown("---")
f_c1, f_c2 = st.columns(2)
f_c1.caption(f"🏫 {l['school']}")
f_c2.markdown(f"<div style='text-align: right; color: gray; font-size: 0.8em;'>{l['copy']}</div>", unsafe_allow_html=True)
