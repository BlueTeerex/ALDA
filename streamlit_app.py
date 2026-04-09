import streamlit as st
import numpy as np
import ephem
import pandas as pd
from astropy.time import Time
import matplotlib.pyplot as plt
from math import acos, degrees

# --- 1. 高精度科研引擎 (向量幾何邏輯) ---
class ALDAEngine:
    def __init__(self, oe_line):
        self.oe_line = oe_line

    def get_metrics(self, jd):
        date = Time(jd, format='jd').datetime
        ast = ephem.readdb(self.oe_line)
        ast.compute(date)
        sun = ephem.Sun(date)
        
        r = ast.sun_distance
        delta = ast.earth_distance
        R = sun.earth_distance 
        
        # 相位角 (α) 計算
        cos_alpha = (r**2 + delta**2 - R**2) / (2 * r * delta)
        phase_angle = degrees(acos(max(-1, min(1, cos_alpha))))
        
        # 距角 (θ) 計算
        cos_theta = (R**2 + delta**2 - r**2) / (2 * R * delta)
        elongation = degrees(acos(max(-1, min(1, cos_theta))))
        
        return {'Date': date, 'Phase': phase_angle, 'Elongation': elongation, 'JD': jd}

# --- 2. 擴充目標小行星數據庫 ---
# 包含：Bennu, Ryugu, Itokawa (Hayabusa目標), Apophis (2029接近), Eros (首顆被繞飛), Didymos (DART任務)
PAPER_ASTEROIDS = {
    "162173 Ryugu": "162173 Ryugu,e,5.86663,251.29446,211.61035,1.1910091,0,0.19111632,327.3279370,5/31.0/2020,2000,H19.55,0.15",
    "101955 Bennu": "101955 Bennu,e,6.03494,2.06087,66.22307,1.1259673,0,0.20374511,101.7039655,5/31.0/2020,2000,H20.45,0.15",
    "25143 Itokawa": "25143 Itokawa,e,1.62154,162.81303,69.08304,1.3241094,0,0.2801456,1.4883134,5/31.0/2020,2000,H19.2,0.15",
    "99942 Apophis": "99942 Apophis,e,3.331,126.395,204.446,0.9224,0,0.1912,250.042,5/31.0/2020,2000,H19.7,0.15",
    "433 Eros": "433 Eros,e,10.827,178.783,304.402,1.4582,0,0.2227,178.817,5/31.0/2020,2000,H11.16,0.15",
    "65803 Didymos": "65803 Didymos,e,3.408,73.207,319.324,1.6446,0,0.3838,231.018,5/31.0/2020,2000,H18.1,0.15"
}

# --- 3. 多語言字典 ---
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
        "func_text": "* **自動計算幾何參數**：基於論文公式計算相位角、距角等關鍵天文數據。\n* **精確視窗預測**：過濾出符合科學觀測條件（Phase < 30°, Elongation > 90°）的時段。\n* **多目標支持**：包含 Ryugu, Bennu, Apophis 等重點研究目標。",
        "how_title": "📖 如何使用",
        "how_text": "1. 在左側選單選擇目標小行星。\n2. 設定預測起始年份與持續年數。\n3. 點擊「開始分析」，系統將根據盧教授高精度算法列出建議時段。",
        "val_title": "🔍 模型準確性驗證 (Validation)",
        "val_desc": "根據本論文第五章，本模型經由 ALCDEF 資料庫驗證，時間預測誤差 ±2.5 天，相位角誤差 < 0.5°。",
        "settings": "⚙️ 參數設定",
        "target": "選擇目標小行星",
        "start_year": "預測起始年份",
        "years": "預測年數",
        "run_btn": "🚀 執行高精度分析",
        "result_title": "📅 建議觀測時段 (α < 30°, θ > 90°)",
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
        "why_text": "Asteroid physical modeling relies heavily on lightcurve data. ALDA helps amateur astronomers find optimal observation windows to fill scientific gaps.",
        "func_title": "🛠️ Functions",
        "func_text": "* **Geometric Calculation**: Auto-calculate Phase Angle, Elongation based on academic models.\n* **Window Prediction**: Filter time slots meeting scientific criteria.\n* **Multi-Target Support**: Includes Ryugu, Bennu, Apophis, and more.",
        "how_title": "📖 How to Use",
        "how_text": "1. Select a target asteroid from the sidebar.\n2. Set the prediction timeframe.\n3. Click 'Run Analysis' to see recommended dates.",
        "val_title": "🔍 Model Validation",
        "val_desc": "As per Chapter 5, the model is validated against the ALCDEF database with mean time error ±2.5 days.",
        "settings": "⚙️ Settings",
        "target": "Select Target",
        "start_year": "Start Year",
        "years": "Duration (Years)",
        "run_btn": "🚀 Run High-Precision Analysis",
        "result_title": "📅 Recommended Windows",
        "no_result": "❌ No windows found.",
        "chart_title": "Observation Trends (α & θ)",
        "school": "Produced by students of Premier School Affiliated to Hou Kong Middle School (Macau)",
        "copy": "Copyright © 2026 ALDA Project. All Rights Reserved."
    }
}

# --- 4. UI 控制 ---
if 'lang' not in st.session_state: st.session_state.lang = 'zh'
def toggle_lang(): st.session_state.lang = 'en' if st.session_state.lang == 'zh' else 'zh'
l = LANG[st.session_state.lang]

st.set_page_config(page_title="ALDA - Multi-Asteroid Analysis", layout="wide")

# Header
h1, h2 = st.columns([8, 2])
h1.title(l["title"])
h2.button(l["lang_btn"], on_click=toggle_lang)

t1, t2, t3 = st.tabs([l["about_tab"], l["calc_tab"], l["val_tab"]])

with t1:
    st.header(l["why_title"])
    st.write(l["why_text"])
    c_f, c_h = st.columns(2)
    with c_f:
        st.header(l["func_title"])
        st.write(l["func_text"])
    with c_h:
        st.header(l["how_title"])
        st.write(l["how_text"])
    st.info(f"🏫 {l['school']}")

with t3:
    st.header(l["val_title"])
    st.write(l["val_desc"])
    st.table(pd.DataFrame({
        "Parameter": ["Phase Angle (α)", "Time Prediction", "Reference"],
        "Mean Error": ["± 0.42°", "± 2.5 Days", "Chapter 5 Paper"]
    }))

with t2:
    with st.sidebar:
        st.header(l["settings"])
        target_name = st.selectbox(l["target"], list(PAPER_ASTEROIDS.keys()))
        s_year = st.number_input(l["start_year"], value=2025)
        duration_val = st.slider(l["years"], 1, 25, 20)
        st.write("---")
        st.caption("Constraints: α < 30°, θ > 90°")

    if st.button(l["run_btn"]):
        engine = ALDAEngine(PAPER_ASTEROIDS[target_name])
        start_jd = Time(f"{s_year}-01-01").jd
        jd_steps = np.arange(start_jd, start_jd + (duration_val * 365), 2)
        
        results = [engine.get_metrics(jd) for jd in jd_steps]
        df = pd.DataFrame(results)
        valid = df[(df['Phase'] < 30) & (df['Elongation'] > 90)].copy()

        if not valid.empty:
            st.header(l["result_title"])
            valid['group'] = (valid['Date'].diff().dt.days > 10).cumsum()
            for _, gp in valid.groupby('group'):
                st.success(f"✅ **{gp['Date'].iloc[0].strftime('%Y-%m-%d')} ~ {gp['Date'].iloc[-1].strftime('%Y-%m-%d')}**")
            
            st.subheader(l["chart_title"])
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df['Date'], df['Phase'], label="α (Phase)", color='orange', alpha=0.8)
            ax.plot(df['Date'], df['Elongation'], label="θ (Elongation)", color='blue', alpha=0.8)
            ax.fill_between(df['Date'], 0, 180, where=(df['Phase']<30)&(df['Elongation']>90), 
                            color='green', alpha=0.2, label="Best Window")
            ax.set_ylim(0, 180)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.2)
            st.pyplot(fig)
        else:
            st.warning(l["no_result"])

# Footer
st.markdown("---")
f1, f2 = st.columns(2)
f1.caption(l["school"])
f2.markdown(f"<div style='text-align: right; color: gray; font-size: 0.8em;'>{l['copy']}</div>", unsafe_allow_html=True)
