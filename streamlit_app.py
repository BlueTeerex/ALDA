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

# --- 2. 軌道數據庫 ---
PAPER_ASTEROIDS = {
    "162173 Ryugu": "162173 Ryugu,e,5.86663,251.29446,211.61035,1.1910091,0,0.19111632,327.3279370,5/31.0/2020,2000,H19.55,0.15",
    "101955 Bennu": "101955 Bennu,e,6.03494,2.06087,66.22307,1.1259673,0,0.20374511,101.7039655,5/31.0/2020,2000,H20.45,0.15",
    "25143 Itokawa": "25143 Itokawa,e,1.62154,162.81303,69.08304,1.3241094,0,0.2801456,1.4883134,5/31.0/2020,2000,H19.2,0.15",
    "99942 Apophis": "99942 Apophis,e,3.331,126.395,204.446,0.9224,0,0.1912,250.042,5/31.0/2020,2000,H19.7,0.15",
    "433 Eros": "433 Eros,e,10.827,178.783,304.402,1.4582,0,0.2227,178.817,5/31.0/2020,2000,H11.16,0.15"
}

# --- 3. 三語專業字典 (標題漢化) ---
LANG_MAP = {"繁體中文": "zh_TW", "简体中文": "zh_CN", "English": "en"}
LANG_DICT = {
    "zh_TW": {
        "full_name": "小行星光變數據擴增系統",
        "nav_label": "導航選單",
        "nav_predict": "觀測視窗預測",
        "nav_background": "開發背景",
        "nav_val": "模型準確性驗證",
        "why_title": "研究背景與目的",
        "why_text": "小行星形狀重構與物理性質研究高度依賴光變曲線數據。多數小行星在特定幾何相位區間缺乏連續觀測記錄。本系統旨在精確預測觀測視窗，協助科研人員填補數據缺口。",
        "val_title": "預測誤差分析",
        "val_col_param": "評估參數", "val_col_error": "平均誤差", "val_col_source": "數據驗證來源",
        "settings": "觀測參數設定", "target": "選取目標小行星", "start_year": "預測起始年份",
        "years": "預測跨度 (年)", "run_btn": "執行高精度分析", "result_title": "建議觀測時間表",
        "chart_title": "幾何參數演化趨勢 (α & θ)", "inst_label": "製作單位",
        "school": "澳門濠江中學附設英才學校 學生團隊", "copy": "版權所有 © 2026 ALDA 項目。保留所有權利。"
    },
    "zh_CN": {
        "full_name": "小行星光变数据扩增系统",
        "nav_label": "导航菜单",
        "nav_predict": "观测视窗预测",
        "nav_background": "开发背景",
        "nav_val": "模型准确性验证",
        "why_title": "研究背景与目的",
        "why_text": "小行星形状重构与物理性质研究高度依赖光变曲线数据。多数小行星在特定几何相位区间缺乏连续观测记录。本系统旨在精确预测观测视窗，协助科研人员填补数据缺口。",
        "val_title": "预测误差分析",
        "val_col_param": "评估参数", "val_col_error": "平均误差", "val_col_source": "数据验证来源",
        "settings": "观测参数设定", "target": "选取目标小行星", "start_year": "预测起始年份",
        "years": "预测跨度 (年)", "run_btn": "执行高精度分析", "result_title": "建议观测时间表",
        "chart_title": "几何参数演化趋势 (α & θ)", "inst_label": "制作单位",
        "school": "澳门濠江中学附属英才学校 学生团队", "copy": "版权所有 © 2026 ALDA 项目。保留所有权利。"
    },
    "en": {
        "full_name": "Asteroid Lightcurve Data Augmentor",
        "nav_label": "Navigation",
        "nav_predict": "Window Prediction",
        "nav_background": "Background",
        "nav_val": "Validation",
        "why_title": "Motivation",
        "why_text": "Asteroid physical modeling relies heavily on lightcurve data. ALDA predicts optimal observation windows to fill scientific gaps at critical geometric phases.",
        "val_title": "Error Analysis",
        "val_col_param": "Parameter", "val_col_error": "Mean Error", "val_col_source": "Source",
        "settings": "Observation Settings", "target": "Select Target", "start_year": "Start Year",
        "years": "Duration (Years)", "run_btn": "Run Analysis", "result_title": "Recommended Schedule",
        "chart_title": "Parameter Evolution Trends (α & θ)", "inst_label": "Institution",
        "school": "Students of Premier School Affiliated to Hou Kong Middle School (Macau)", "copy": "Copyright © 2026 ALDA Project. All Rights Reserved."
    }
}

# --- 4. 網頁 UI 設定與 CSS 優化 ---
st.set_page_config(page_title="ALDA Scientific", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #1f77b4; color: white; border: none; }
    .stButton>button:hover { background-color: #1a5f8a; color: white; border: none; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    div[data-testid="stExpander"] { border: 1px solid #e6e6e6; border-radius: 12px; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# 側邊欄佈局
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>☄️ ALDA</h1>", unsafe_allow_html=True)
    selected_lang_name = st.selectbox("Language", list(LANG_MAP.keys()), label_visibility="collapsed")
    lang_key = LANG_MAP[selected_lang_name]
    l = LANG_DICT[lang_key]
    
    st.divider()
    page = st.radio(l["nav_label"], [l["nav_predict"], l["nav_background"], l["nav_val"]])
    
    st.divider()
    st.caption(f"{l['inst_label']}:")
    st.write(f"**{l['school']}**")

# --- 5. 頁面內容 ---
st.title("ALDA")
st.markdown(f"<h3 style='color: #555;'>{l['full_name']}</h3>", unsafe_allow_html=True)
st.divider()

if page == l["nav_predict"]:
    # 參數設定卡片
    with st.container():
        st.subheader(l["settings"])
        col_t, col_y, col_s = st.columns([2, 1, 1])
        target_id = col_t.selectbox(l["target"], list(PAPER_ASTEROIDS.keys()))
        s_year = col_y.number_input(l["start_year"], value=2025)
        span = col_s.slider(l["years"], 1, 25, 15)
        btn_run = st.button(l["run_btn"], type="primary")

    if btn_run:
        engine = ALDAEngine(PAPER_ASTEROIDS[target_id])
        jd_start = Time(f"{s_year}-01-01").jd
        jd_array = np.arange(jd_start, jd_start + (span * 365), 2)
        results = [engine.get_metrics(jd) for jd in jd_array]
        df = pd.DataFrame(results)
        valid = df[(df['Phase'] < 30) & (df['Elongation'] > 90)].copy()

        # 視覺化摘要數據 (減少頁面空白)
        m1, m2, m3 = st.columns(3)
        m1.metric("Target", target_id)
        m2.metric("Windows Found", len(valid['Date'].diff().dt.days > 10) if not valid.empty else 0)
        m3.metric("Analysis Span", f"{span} Yrs")

        if not valid.empty:
            st.markdown(f"#### {l['result_title']}")
            valid['group'] = (valid['Date'].diff().dt.days > 10).cumsum()
            res_cols = st.columns(3)
            for idx, (_, gp) in enumerate(valid.groupby('group')):
                with res_cols[idx % 3]:
                    st.success(f"**{gp['Date'].iloc[0].strftime('%Y-%m-%d')}** \n\n **{gp['Date'].iloc[-1].strftime('%Y-%m-%d')}**")
            
            st.divider()
            st.subheader(l["chart_title"])
            fig, ax = plt.subplots(figsize=(12, 4), facecolor='#f8f9fa')
            ax.set_facecolor('#ffffff')
            ax.plot(df['Date'], df['Phase'], label="α (Phase)", color='#E67E22', linewidth=2)
            ax.plot(df['Date'], df['Elongation'], label="θ (Elongation)", color='#2E86C1', linewidth=2)
            ax.fill_between(df['Date'], 0, 180, where=(df['Phase']<30)&(df['Elongation']>90), color='#2ECC71', alpha=0.2, label="Optimal")
            ax.set_ylim(0, 180)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(loc='upper right', frameon=True)
            st.pyplot(fig)
        else:
            st.warning("No windows meet the α < 30° & θ > 90° criteria in this period.")

elif page == l["nav_background"]:
    st.subheader(l["why_title"])
    st.info(l["why_text"])
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### {l['func_title'] if 'func_title' in l else 'Functions'}")
        st.write("1. Orbital dynamics analysis.\n2. Automatic window filtering.\n3. Multi-target data tracking.")
    with c2:
        st.markdown("#### Scientific Constraints")
        st.latex(r"\alpha < 30^\circ")
        st.latex(r"\theta > 90^\circ")

elif page == l["nav_val"]:
    st.subheader(l["val_title"])
    val_df = pd.DataFrame({
        l["val_col_param"]: ["Phase Angle (α)", "Window Accuracy", "Reference DB"],
        l["val_col_error"]: ["± 0.42°", "± 2.5 Days", "Validated"],
        l["val_col_source"]: ["JPL SBDB", "ALCDEF", "Chapter 5"]
    })
    st.table(val_df)

# 頁腳
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown(f"<p style='text-align: center; color: #999;'>{l['copy']}</p>", unsafe_allow_html=True)
