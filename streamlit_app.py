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
        r, delta, R = ast.sun_distance, ast.earth_distance, sun.earth_distance 
        cos_alpha = (r**2 + delta**2 - R**2) / (2 * r * delta)
        phase_angle = degrees(acos(max(-1, min(1, cos_alpha))))
        cos_theta = (R**2 + delta**2 - r**2) / (2 * R * delta)
        elongation = degrees(acos(max(-1, min(1, cos_theta))))
        return {'Date': date, 'Phase': phase_angle, 'Elongation': elongation}

# --- 2. 擴充後的軌道數據庫 ---
# 包含近地小行星與主帶小行星，適合各種科研模擬
PAPER_ASTEROIDS = {
    "162173 Ryugu": "162173 Ryugu,e,5.86663,251.29446,211.61035,1.1910091,0,0.19111632,327.3279370,5/31.0/2020,2000,H19.55,0.15",
    "101955 Bennu": "101955 Bennu,e,6.03494,2.06087,66.22307,1.1259673,0,0.20374511,101.7039655,5/31.0/2020,2000,H20.45,0.15",
    "25143 Itokawa": "25143 Itokawa,e,1.62154,162.81303,69.08304,1.3241094,0,0.2801456,1.4883134,5/31.0/2020,2000,H19.2,0.15",
    "99942 Apophis": "99942 Apophis,e,3.331,126.395,204.446,0.9224,0,0.1912,250.042,5/31.0/2020,2000,H19.7,0.15",
    "433 Eros": "433 Eros,e,10.827,178.783,304.402,1.4582,0,0.2227,178.817,5/31.0/2020,2000,H11.16,0.15",
    "1 Ceres": "1 Ceres,f,10.59,80.31,72.52,2.767,0.076,0.214,102.83,5/31/2020,2000,H3.3,0.15",
    "2 Pallas": "2 Pallas,f,34.84,173.07,310.20,2.772,0.231,0.213,24.12,5/31/2020,2000,H4.1,0.15",
    "4 Vesta": "4 Vesta,f,7.14,103.85,149.75,2.361,0.089,0.272,254.12,5/31/2020,2000,H3.2,0.15",
    "16 Psyche": "16 Psyche,f,3.09,150.31,121.33,2.924,0.134,0.191,228.46,5/31/2020,2000,H5.9,0.15",
    "65803 Didymos": "65803 Didymos,e,3.408,164.63,319.32,1.644,0,0.384,204.44,5/31/2020,2000,H18.1,0.15"
}

# --- 3. 三語專業字典 (確保翻譯統一) ---
LANG_MAP = {"繁體中文": "zh_TW", "简体中文": "zh_CN", "English": "en"}
LANG_DICT = {
    "zh_TW": {
        "full_name": "小行星光變數據擴增系統", "nav_label": "導航選單", "nav_predict": "觀測視窗預測",
        "nav_background": "開發背景", "nav_val": "準確性驗證", "settings": "觀測參數設定",
        "target": "選取目標小行星", "start_year": "預測起始年份", "years": "跨度 (年)",
        "run_btn": "開始分析", "result_title": "建議觀測時間表 (YYYY-MM-DD)", "chart_title": "幾何演化趨勢 (α & θ)",
        "school": "澳門濠江中學附設英才學校 學生團隊", "date_to": "至", "m_target": "目標",
        "m_win": "視窗數", "m_span": "週期", "legend_p": "相位角 (α)", "legend_e": "距角 (θ)", 
        "legend_o": "最佳視窗", "y_axis": "角度 (度)", "copy": "© 2026 ALDA 項目組"
    },
    "zh_CN": {
        "full_name": "小行星光变数据扩增系统", "nav_label": "导航菜单", "nav_predict": "观测视窗预测",
        "nav_background": "开发背景", "nav_val": "准确性验证", "settings": "观测参数设定",
        "target": "选取目标小行星", "start_year": "预测起始年份", "years": "跨度 (年)",
        "run_btn": "开始分析", "result_title": "建议观测时间表 (YYYY-MM-DD)", "chart_title": "几何演化趋势 (α & θ)",
        "school": "澳门濠江中学附属英才学校 学生团队", "date_to": "至", "m_target": "目标",
        "m_win": "视窗数", "m_span": "周期", "legend_p": "相位角 (α)", "legend_e": "距角 (θ)", 
        "legend_o": "最佳视窗", "y_axis": "角度 (度)", "copy": "© 2026 ALDA 项目组"
    },
    "en": {
        "full_name": "Asteroid Lightcurve Data Augmentor", "nav_label": "Navigation", "nav_predict": "Prediction",
        "nav_background": "Background", "nav_val": "Validation", "settings": "Configuration",
        "target": "Select Target", "start_year": "Start Year", "years": "Time Span",
        "run_btn": "Analyze", "result_title": "Observation Schedule (YYYY-MM-DD)", "chart_title": "Evolution Trends (α & θ)",
        "school": "Team Hou Kong Premier School (Macau)", "date_to": "to", "m_target": "Target",
        "m_win": "Windows", "m_span": "Span", "legend_p": "Phase (α)", "legend_e": "Elongation (θ)", 
        "legend_o": "Optimal", "y_axis": "Degrees", "copy": "© 2026 ALDA Research Team"
    }
}

# --- 4. UI 設定 ---
st.set_page_config(page_title="ALDA Scientific", layout="wide")
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; background-color: #0071e3; color: white; font-weight: bold; height: 3em; }
    .stMetric { background-color: #fcfcfc; padding: 15px; border-radius: 10px; border: 1px solid #eee; }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>☄️ ALDA</h1>", unsafe_allow_html=True)
    sel_lang = st.selectbox("Language", list(LANG_MAP.keys()), label_visibility="collapsed")
    l = LANG_DICT[LANG_MAP[sel_lang]]
    st.divider()
    page = st.radio(l["nav_label"], [l["nav_predict"], l["nav_background"], l["nav_val"]])
    st.divider()
    st.write(f"**{l['school']}**")

# --- 5. 分頁邏輯 ---
st.title("ALDA")
st.markdown(f"<p style='color: grey; font-size: 1.2rem;'>{l['full_name']}</p>", unsafe_allow_html=True)
st.divider()

if page == l["nav_predict"]:
    with st.container():
        st.subheader(l["settings"])
        c_t, c_y, c_s = st.columns([2, 1, 1])
        target_id = c_t.selectbox(l["target"], list(PAPER_ASTEROIDS.keys()))
        s_year = c_y.number_input(l["start_year"], value=2025)
        span = c_s.slider(l["years"], 1, 25, 15)
        btn_run = st.button(l["run_btn"])

    if btn_run:
        engine = ALDAEngine(PAPER_ASTEROIDS[target_id])
        jd_start = Time(f"{s_year}-01-01").jd
        jd_array = np.arange(jd_start, jd_start + (span * 365), 2)
        results = [engine.get_metrics(jd) for jd in jd_array]
        df = pd.DataFrame(results)
        valid = df[(df['Phase'] < 30) & (df['Elongation'] > 90)].copy()

        # 頂部數據卡片
        m1, m2, m3 = st.columns(3)
        m1.metric(l["m_target"], target_id)
        m2.metric(l["m_win"], len(valid['Date'].diff().dt.days > 10) if not valid.empty else 0)
        m3.metric(l["m_span"], f"{span}Y")

        if not valid.empty:
            st.markdown(f"#### {l['result_title']}")
            valid['group'] = (valid['Date'].diff().dt.days > 10).cumsum()
            res_cols = st.columns(3)
            for idx, (_, gp) in enumerate(valid.groupby('group')):
                with res_cols[idx % 3]:
                    # 嚴格輸出 YYYY-MM-DD
                    start = gp['Date'].iloc[0].strftime('%Y-%m-%d')
                    end = gp['Date'].iloc[-1].strftime('%Y-%m-%d')
                    st.success(f"{start} {l['date_to']} {end}")
            
            st.divider()
            st.subheader(l["chart_title"])
            fig, ax = plt.subplots(figsize=(12, 4.5))
            ax.plot(df['Date'], df['Phase'], label=l["legend_p"], color='#E67E22', linewidth=2)
            ax.plot(df['Date'], df['Elongation'], label=l["legend_e"], color='#2E86C1', linewidth=2)
            ax.fill_between(df['Date'], 0, 180, where=(df['Phase']<30)&(df['Elongation']>90), 
                            color='#2ECC71', alpha=0.2, label=l["legend_o"])
            ax.set_ylabel(l["y_axis"])
            ax.legend(loc='upper right')
            st.pyplot(fig)
        else:
            st.warning("No data found.")

# 全域頁腳
st.divider()
st.markdown(f"<p style='text-align: center; color: #999;'>{l['copy']}</p>", unsafe_allow_html=True)
