import streamlit as st
import numpy as np
import ephem
import pandas as pd
from astropy.time import Time
import matplotlib.pyplot as plt
from math import acos, degrees

# --- 1. 計算引擎 ---
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

# --- 2. 小行星數據 ---
PAPER_ASTEROIDS = {
    "162173 Ryugu": "162173 Ryugu,e,5.86663,251.29446,211.61035,1.1910091,0,0.19111632,327.3279370,5/31.0/2020,2000,H19.55,0.15",
    "101955 Bennu": "101955 Bennu,e,6.03494,2.06087,66.22307,1.1259673,0,0.20374511,101.7039655,5/31.0/2020,2000,H20.45,0.15",
    "25143 Itokawa": "25143 Itokawa,e,1.62154,162.81303,69.08304,1.3241094,0,0.2801456,1.4883134,5/31.0/2020,2000,H19.2,0.15",
    "99942 Apophis": "99942 Apophis,e,3.331,126.395,204.446,0.9224,0,0.1912,250.042,5/31.0/2020,2000,H19.7,0.15",
    "433 Eros": "433 Eros,e,10.827,178.783,304.402,1.4582,0,0.2227,178.817,5/31.0/2020,2000,H11.16,0.15"
}

# --- 3. 語言字典 ---
LANG_MAP = {"繁體中文": "zh_TW", "简体中文": "zh_CN", "English": "en"}
LANG_DICT = {
    "zh_TW": {
        "full_name": "小行星光變數據擴增系統", "nav_predict": "觀測視窗預測", "nav_background": "開發背景", "nav_val": "準確性驗證",
        "settings": "參數配置", "target": "選定天體", "start_year": "起始年份", "years": "時間跨度", "run_btn": "開始分析",
        "result_title": "建議觀測時間表", "chart_title": "幾何演化趨勢 (α & θ)", "school": "澳門濠江中學附設英才學校 團隊",
        "date_to": "至", "m_target": "目標", "m_win": "視窗數", "m_span": "週期", "legend_p": "相位角 (α)", 
        "legend_e": "距角 (θ)", "legend_o": "最佳觀測期", "y_axis": "角度 (度)", "copy": "© 2026 ALDA 科研組"
    },
    "zh_CN": {
        "full_name": "小行星光变数据扩增系统", "nav_predict": "观测视窗预测", "nav_background": "开发背景", "nav_val": "准确性验证",
        "settings": "参数配置", "target": "选定天体", "start_year": "起始年份", "years": "时间跨度", "run_btn": "开始分析",
        "result_title": "建议观测时间表", "chart_title": "几何演化趋势 (α & θ)", "school": "澳门濠江中学附属英才学校 团队",
        "date_to": "至", "m_target": "目标", "m_win": "视窗数", "m_span": "周期", "legend_p": "相位角 (α)", 
        "legend_e": "距角 (θ)", "legend_o": "最佳观测期", "y_axis": "角度 (度)", "copy": "© 2026 ALDA 科研组"
    },
    "en": {
        "full_name": "Asteroid Lightcurve Data Augmentor", "nav_predict": "Prediction", "nav_background": "Background", "nav_val": "Validation",
        "settings": "Parameters", "target": "Target Object", "start_year": "Start Year", "years": "Time Span", "run_btn": "Analyze Now",
        "result_title": "Recommended Schedule", "chart_title": "Geometric Trends (α & θ)", "school": "Team Hou Kong Premier School (Macau)",
        "date_to": "to", "m_target": "Target", "m_win": "Windows", "m_span": "Span", "legend_p": "Phase (α)", 
        "legend_e": "Elongation (θ)", "legend_o": "Optimal", "y_axis": "Degrees", "copy": "© 2026 ALDA Research Team"
    }
}

# --- 4. UI 深度美化 ---
st.set_page_config(page_title="ALDA | Scientific", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #ffffff; color: #1d1d1f; }
    .stApp { background: radial-gradient(circle at top right, #f5f5f7, #ffffff); }
    .stButton>button { border-radius: 12px; background-color: #0071e3; color: white; border: none; font-weight: 600; }
    .stButton>button:hover { background-color: #0077ed; transform: scale(1.02); }
    </style>
    """, unsafe_allow_html=True)

# --- 5. 側邊欄佈局 ---
with st.sidebar:
    st.markdown("<div style='padding: 10px 0;'><h2 style='text-align: center;'>ALDA</h2></div>", unsafe_allow_html=True)
    sel_lang = st.selectbox("Language", list(LANG_MAP.keys()), label_visibility="collapsed")
    l = LANG_DICT[LANG_MAP[sel_lang]]
    
    st.divider()
    page = st.radio("Navigation", [l["nav_predict"], l["nav_background"], l["nav_val"]], label_visibility="collapsed")
    
    # 使用 HTML 增加間距避免報錯
    st.markdown("<br>" * 10, unsafe_allow_html=True)
    st.markdown(f"<div style='color: #86868b; font-size: 0.8rem; text-align: center;'>{l['school']}</div>", unsafe_allow_html=True)

# --- 6. 主頁面內容 ---
st.markdown(f"<h1 style='font-size: 3.5rem; font-weight: 700; margin-bottom: 0;'>ALDA</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='font-size: 1.2rem; color: #86868b;'>{l['full_name']}</p>", unsafe_allow_html=True)
st.divider()

if page == l["nav_predict"]:
    with st.container():
        c_t, c_y, c_s = st.columns([2, 1, 1])
        target_id = c_t.selectbox(l["target"], list(PAPER_ASTEROIDS.keys()))
        s_year = c_y.number_input(l["start_year"], value=2025)
        span = c_s.slider(l["years"], 1, 25, 15)
        btn_run = st.button(l["run_btn"], type="primary", use_container_width=True)

    if btn_run:
        engine = ALDAEngine(PAPER_ASTEROIDS[target_id])
        jd_start = Time(f"{s_year}-01-01").jd
        jd_array = np.arange(jd_start, jd_start + (span * 365), 2)
        results = [engine.get_metrics(jd) for jd in jd_array]
        df = pd.DataFrame(results)
        valid = df[(df['Phase'] < 30) & (df['Elongation'] > 90)].copy()

        if not valid.empty:
            st.markdown(f"### {l['result_title']}")
            valid['group'] = (valid['Date'].diff().dt.days > 10).cumsum()
            res_cols = st.columns(3)
            for idx, (_, gp) in enumerate(valid.groupby('group')):
                with res_cols[idx % 3]:
                    s_str = gp['Date'].iloc[0].strftime('%Y-%m-%d')
                    e_str = gp['Date'].iloc[-1].strftime('%Y-%m-%d')
                    st.success(f"{s_str} {l['date_to']} {e_str}")
            
            # 圖表
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(df['Date'], df['Phase'], label=l["legend_p"], color='#ff7f0e', linewidth=2)
            ax.plot(df['Date'], df['Elongation'], label=l["legend_e"], color='#0071e3', linewidth=2)
            ax.fill_between(df['Date'], 0, 180, where=(df['Phase']<30)&(df['Elongation']>90), color='#34c759', alpha=0.15)
            ax.legend(loc='upper right')
            st.pyplot(fig)
        else:
            st.warning("No data found.")

elif page == l["nav_background"]:
    st.info(l["why_text"])

elif page == l["nav_val"]:
    st.write(l["nav_val"])

st.markdown(f"<div style='text-align: center; margin-top: 50px; color: #bfbfbf;'>{l['copy']}</div>", unsafe_allow_html=True)
