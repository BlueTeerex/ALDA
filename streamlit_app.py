import streamlit as st
import numpy as np
import ephem
import pandas as pd
from astropy.time import Time
import matplotlib.pyplot as plt
from math import acos, degrees

# --- 1. 高精度科研計算引擎 ---
class ALDAEngine:
    def __init__(self, oe_line):
        self.oe_line = oe_line

    def get_metrics(self, jd):
        date = Time(jd, format='jd').datetime
        ast = ephem.readdb(self.oe_line)
        ast.compute(date)
        sun = ephem.Sun(date)
        
        # 提取日心距離與地心距離
        r = ast.sun_distance
        delta = ast.earth_distance
        R = sun.earth_distance 
        
        # 相位角 (α) 計算: 太陽-小行星-地球夾角
        cos_alpha = (r**2 + delta**2 - R**2) / (2 * r * delta)
        phase_angle = degrees(acos(max(-1, min(1, cos_alpha))))
        
        # 距角 (θ) 計算: 太陽-地球-小行星夾角
        cos_theta = (R**2 + delta**2 - r**2) / (2 * R * delta)
        elongation = degrees(acos(max(-1, min(1, cos_theta))))
        
        return {'Date': date, 'Phase': phase_angle, 'Elongation': elongation, 'JD': jd}

# --- 2. 目標小行星資料庫 (包含新增目標) ---
PAPER_ASTEROIDS = {
    "162173 Ryugu": "162173 Ryugu,e,5.86663,251.29446,211.61035,1.1910091,0,0.19111632,327.3279370,5/31.0/2020,2000,H19.55,0.15",
    "101955 Bennu": "101955 Bennu,e,6.03494,2.06087,66.22307,1.1259673,0,0.20374511,101.7039655,5/31.0/2020,2000,H20.45,0.15",
    "25143 Itokawa": "25143 Itokawa,e,1.62154,162.81303,69.08304,1.3241094,0,0.2801456,1.4883134,5/31.0/2020,2000,H19.2,0.15",
    "99942 Apophis": "99942 Apophis,e,3.331,126.395,204.446,0.9224,0,0.1912,250.042,5/31.0/2020,2000,H19.7,0.15",
    "433 Eros": "433 Eros,e,10.827,178.783,304.402,1.4582,0,0.2227,178.817,5/31.0/2020,2000,H11.16,0.15"
}

# --- 3. 三語專業字典 ---
LANG = {
    "zh_TW": {
        "title": "ALDA: 小行星光變數據擴增系統",
        "lang_label": "語言選擇 (Language)",
        "about_tab": "關於本站",
        "calc_tab": "觀測預測",
        "val_tab": "模型驗證",
        "why_title": "開發背景",
        "why_text": "小行星形狀重構與物理性質研究高度依賴光變曲線。然而多數小行星在特定PABS緯度區間缺乏數據。ALDA旨在協助觀測者尋找最佳觀測時段，填補科學數據缺口。",
        "func_title": "系統功能",
        "func_text": "1. 自動計算相位角、距角等幾何參數。\n2. 根據科學約束（α < 30°, θ > 90°）預測最佳視窗。\n3. 支持多目標小行星追蹤與國際協作。",
        "how_title": "使用指南",
        "how_text": "請於側邊欄選擇目標天體及預測年份，點擊「執行分析」即可獲取高精度觀測時段與趨勢圖表。",
        "val_title": "準確性驗證",
        "val_desc": "根據本研究第五章，預測模型經ALCDEF資料庫驗證，預測誤差穩定於 ±2.5 天內。",
        "settings": "參數設定",
        "target": "目標天體",
        "start_year": "起始年份",
        "years": "預測跨度 (年)",
        "run_btn": "執行分析",
        "result_title": "建議觀測視窗",
        "no_result": "在設定範圍內未發現符合條件之視窗。",
        "chart_title": "觀測參數變化趨勢 (α & θ)",
        "school": "製作單位：澳門濠江中學附屬英才學校 學生團隊",
        "copy": "Copyright © 2026 ALDA Project. All Rights Reserved."
    },
    "zh_CN": {
        "title": "ALDA: 小行星光变数据扩增系统",
        "lang_label": "语言选择 (Language)",
        "about_tab": "关于本站",
        "calc_tab": "观测预测",
        "val_tab": "模型验证",
        "why_title": "开发背景",
        "why_text": "小行星形状重构与物理性质研究高度依赖光变曲线。然而多数小行星在特定PABS纬度区间缺乏数据。ALDA旨在协助观测者寻找最佳观测时段，填补科学数据缺口。",
        "func_title": "系统功能",
        "func_text": "1. 自动计算相位角、距角等几何参数。\n2. 根据科学约束（α < 30°, θ > 90°）预测最佳视窗。\n3. 支持多目标小行星追踪与国际协作。",
        "how_title": "使用指南",
        "how_text": "请在侧边栏选择目标天体及预测年份，点击“执行分析”即可获取高精度观测时段与趋势图表。",
        "val_title": "准确性验证",
        "val_desc": "根据本研究第五章，预测模型经ALCDEF数据库验证，预测误差稳定在 ±2.5 天内。",
        "settings": "参数设定",
        "target": "目标天体",
        "start_year": "起始年份",
        "years": "预测跨度 (年)",
        "run_btn": "执行分析",
        "result_title": "建议观测视窗",
        "no_result": "在设定范围内未发现符合条件之视窗。",
        "chart_title": "观测参数变化趋势 (α & θ)",
        "school": "制作单位：澳门濠江中学附属英才学校 学生团队",
        "copy": "Copyright © 2026 ALDA Project. All Rights Reserved."
    },
    "en": {
        "title": "ALDA: Asteroid Lightcurve Data Augmentor",
        "lang_label": "Language",
        "about_tab": "About",
        "calc_tab": "Prediction",
        "val_tab": "Validation",
        "why_title": "Background",
        "why_text": "Asteroid physical modeling relies heavily on lightcurve data. ALDA helps astronomers find optimal observation windows to fill scientific gaps at specific PABS latitudes.",
        "func_title": "Functions",
        "func_text": "1. Auto-calculate geometric parameters (Phase, Elongation).\n2. Predict windows meeting scientific criteria (α < 30°, θ > 90°).\n3. Multi-target support for global collaboration.",
        "how_title": "Guide",
        "how_text": "Select a target and timeframe from the sidebar, then click 'Run Analysis' to generate data.",
        "val_title": "Model Validation",
        "val_desc": "Validated against the ALCDEF database (Chapter 5), showing mean error within ±2.5 days.",
        "settings": "Settings",
        "target": "Target Asteroid",
        "start_year": "Start Year",
        "years": "Duration (Years)",
        "run_btn": "Run Analysis",
        "result_title": "Recommended Windows",
        "no_result": "No windows found matching the criteria.",
        "chart_title": "Observation Parameter Trends (α & θ)",
        "school": "Produced by students of Premier School Affiliated to Hou Kong Middle School (Macau)",
        "copy": "Copyright © 2026 ALDA Project. All Rights Reserved."
    }
}

# --- 4. 網頁渲染架構 ---
if 'lang_key' not in st.session_state:
    st.session_state.lang_key = 'zh_TW'

st.set_page_config(page_title="ALDA Scientific", layout="wide")

# 頂部語言切換
st.session_state.lang_key = st.selectbox("Language / 語言", options=['zh_TW', 'zh_CN', 'en'], index=['zh_TW', 'zh_CN', 'en'].index(st.session_state.lang_key))
l = LANG[st.session_state.lang_key]

st.title(l["title"])

tab_about, tab_calc, tab_val = st.tabs([l["about_tab"], l["calc_tab"], l["val_tab"]])

with tab_about:
    st.header(l["why_title"])
    st.write(l["why_text"])
    col1, col2 = st.columns(2)
    with col1:
        st.header(l["func_title"])
        st.write(l["func_text"])
    with col2:
        st.header(l["how_title"])
        st.write(l["how_text"])
    st.info(f"Institution: {l['school']}")

with tab_val:
    st.header(l["val_title"])
    st.write(l["val_desc"])
    st.table(pd.DataFrame({
        "Parameter": ["Phase Angle (α)", "Window Accuracy", "Dataset"],
        "Value": ["± 0.42°", "± 2.5 Days", "ALCDEF"]
    }))

with tab_calc:
    with st.sidebar:
        st.header(l["settings"])
        target_id = st.selectbox(l["target"], list(PAPER_ASTEROIDS.keys()))
        s_year = st.number_input(l["start_year"], value=2025)
        span = st.slider(l["years"], 1, 25, 15)
    
    if st.button(l["run_btn"]):
        engine = ALDAEngine(PAPER_ASTEROIDS[target_id])
        jd_start = Time(f"{s_year}-01-01").jd
        jd_array = np.arange(jd_start, jd_start + (span * 365), 2)
        
        results = [engine.get_metrics(jd) for jd in jd_array]
        df = pd.DataFrame(results)
        valid = df[(df['Phase'] < 30) & (df['Elongation'] > 90)].copy()

        if not valid.empty:
            st.header(l["result_title"])
            valid['group'] = (valid['Date'].diff().dt.days > 10).cumsum()
            for _, gp in valid.groupby('group'):
                st.success(f"{gp['Date'].iloc[0].strftime('%Y-%m-%d')} —— {gp['Date'].iloc[-1].strftime('%Y-%m-%d')}")
            
            # 專業圖表渲染
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df['Date'], df['Phase'], label="Phase (α)", color='#ff7f0e', linewidth=1)
            ax.plot(df['Date'], df['Elongation'], label="Elongation (θ)", color='#1f77b4', linewidth=1)
            ax.fill_between(df['Date'], 0, 180, where=(df['Phase']<30)&(df['Elongation']>90), color='green', alpha=0.15, label="Window")
            ax.set_ylim(0, 180)
            ax.set_ylabel("Degrees")
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True, linestyle=':', alpha=0.6)
            st.pyplot(fig)
        else:
            st.warning(l["no_result"])

# 頁腳
st.markdown("---")
footer_l, footer_r = st.columns(2)
footer_l.caption(l["school"])
footer_r.markdown(f"<div style='text-align: right; color: gray; font-size: 0.8em;'>{l['copy']}</div>", unsafe_allow_html=True)
