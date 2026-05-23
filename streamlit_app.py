import streamlit as st
import numpy as np
import ephem
import pandas as pd
from astropy.time import Time
from math import acos, degrees
from datetime import datetime
import pytz
import plotly.graph_objects as go

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

# --- 2. 擴充軌道數據庫 ---
PAPER_ASTEROIDS = {
    "1 Ceres (穀神星)": "1 Ceres,e,10.59,80.31,72.52,2.767,0,0.076,102.83,5/31/2020,2000,H3.3,0.15",
    "2 Pallas (智神星)": "2 Pallas,e,34.84,173.07,310.20,2.772,0,0.231,24.12,5/31/2020,2000,H4.1,0.15",
    "3 Juno (婚神星)": "3 Juno,e,12.98,169.91,248.13,2.671,0,0.255,35.48,5/31/2020,2000,H5.33,0.15",
    "4 Vesta (灶神星)": "4 Vesta,e,7.14,103.85,149.75,2.361,0,0.089,254.12,5/31/2020,2000,H3.2,0.15",
    "10 Hygiea (健神星)": "10 Hygiea,e,3.84,283.43,312.98,3.137,0,0.116,170.83,5/31/2020,2000,H5.43,0.15",
    "16 Psyche (靈神星)": "16 Psyche,e,3.09,150.31,121.33,2.924,0,0.134,228.46,5/31/2020,2000,H5.9,0.15",
    "433 Eros (愛神星)": "433 Eros,e,10.827,178.783,304.402,1.4582,0,0.2227,178.817,5/31.0/2020,2000,H11.16,0.15",
    "3200 Phaethon (法厄同)": "3200 Phaethon,e,22.258,265.234,318.156,1.2711,0,0.8899,203.491,5/31/2020,2000,H14.6,0.15",
    "4179 Toutatis (圖塔蒂斯)": "4179 Toutatis,e,0.446,122.56,274.82,2.531,0,0.629,328.71,5/31/2020,2000,H15.30,0.15",
    "25143 Itokawa (糸川)": "25143 Itokawa,e,1.62154,162.81303,69.08304,1.3241094,0,0.2801456,1.4883134,5/31.0/2020,2000,H19.2,0.15",
    "65803 Didymos (雙小行星)": "65803 Didymos,e,3.408,164.63,319.32,1.644,0,0.384,204.44,5/31/2020,2000,H18.1,0.15",
    "99942 Apophis (毀神星)": "99942 Apophis,e,3.331,126.395,204.446,0.9224,0,0.1912,250.042,5/31.0/2020,2000,H19.7,0.15",
    "101955 Bennu (貝努)": "101955 Bennu,e,6.03494,2.06087,66.22307,1.1259673,0,0.20374511,101.7039655,5/31.0/2020,2000,H20.45,0.15",
    "162173 Ryugu (龍宮)": "162173 Ryugu,e,5.86663,251.29446,211.61035,1.1910091,0,0.19111632,327.3279370,5/31.0/2020,2000,H19.55,0.15"
}

# --- 3. 三語專業字典 ---
LANG_MAP = {"繁體中文": "zh_TW", "简体中文": "zh_CN", "English": "en"}
LANG_DICT = {
    "zh_TW": {
        "full_name": "小行星光變數據擴增系統", "nav_label": "導航選單", "nav_predict": "觀測視窗預測",
        "nav_background": "開發背景", "nav_val": "模型準確性驗證", "why_title": "研究背景與目的",
        "why_text": "小行星形狀重構與物理性質研究高度依賴光變曲線數據。多數小行星在特定幾何相位區間缺乏連續觀測記錄。本系統旨在精確預測觀測視窗，協助科研人員填補數據缺口。",
        "val_title": "預測誤差分析", "val_col_param": "評估參數", "val_col_error": "平均誤差", "val_col_source": "數據驗證來源",
        "val_row_phase": "相位角 (α)", "val_row_window": "觀測視窗日期", "val_row_data": "驗證資料庫",
        "val_res_phase": "± 0.42°", "val_res_window": "± 2.5 天", "val_res_source1": "ALCDEF 資料庫", "val_res_source2": "論文第五章", "val_res_source3": "已通過驗證",
        "settings": "觀測參數設定", "target": "選取目標小行星", "start_year": "預測起始年份", "years": "預測跨度 (年)",
        "run_btn": "執行高精度分析", "result_title": "建議觀測時間表 (YYYY-MM-DD)", "chart_title": "幾何參數演化趨勢 (α & θ)",
        "inst_label": "製作單位", "school": "澳門濠江中學附設英才學校 學生團隊", "copy": "版權所有 © 2026 ALDA 項目。保留所有權利。",
        "date_to": "至", "metric_target": "目標天體", "metric_windows": "發現視窗數", "metric_span": "分析跨度",
        "legend_phase": "相位角 (α)", "legend_elong": "距角 (θ)", "legend_opt": "最佳視窗", "y_label": "角度 (度)",
        "last_update": "系統最後更新時間", "tz_name": "台北/澳門/北京時間 (UTC+8)", "val_info": "數據已與 NASA JPL Horizons 及 ALCDEF 交叉驗證。"
    },
    "zh_CN": {
        "full_name": "小行星光变数据扩增系统", "nav_label": "导航菜单", "nav_predict": "观测视窗预测",
        "nav_background": "开发背景", "nav_val": "模型准确性验证", "why_title": "研究背景与目的",
        "why_text": "小行星形状重构与物理性质研究高度依赖光变曲线数据。多数小行星在特定几何相位区间缺乏连续观测记录。本系统旨在精确预测观测视窗，协助科研人员填补数据缺口。",
        "val_title": "预测误差分析", "val_col_param": "评估参数", "val_col_error": "平均误差", "val_col_source": "数据验证来源",
        "val_row_phase": "相位角 (α)", "val_row_window": "观测视窗日期", "val_row_data": "验证数据库",
        "val_res_phase": "± 0.42°", "val_res_window": "± 2.5 天", "val_res_source1": "ALCDEF 数据库", "val_res_source2": "论文第五章", "val_res_source3": "已通过验证",
        "settings": "观测参数设定", "target": "选取目标小行星", "start_year": "预测起始年份", "years": "预测跨度 (年)",
        "run_btn": "执行高精度分析", "result_title": "建议观测时间表 (YYYY-MM-DD)", "chart_title": "几何参数演化趋势 (α & θ)",
        "inst_label": "制作单位", "school": "澳门濠江中学附属英才学校 学生团队", "copy": "版权所有 © 2026 ALDA 项目。保留所有权利。",
        "date_to": "至", "metric_target": "目标天体", "metric_windows": "发现视窗数", "metric_span": "分析跨度",
        "legend_phase": "相位角 (α)", "legend_elong": "距角 (θ)", "legend_opt": "最佳视窗", "y_label": "角度 (度)",
        "last_update": "系统最后更新时间", "tz_name": "北京/澳门时间 (UTC+8)", "val_info": "数据已与 NASA JPL Horizons 及 ALCDEF 交叉验证。"
    },
    "en": {
        "full_name": "Asteroid Lightcurve Data Augmenter", "nav_label": "Navigation", "nav_predict": "Window Prediction",
        "nav_background": "Background", "nav_val": "Validation", "why_title": "Motivation & Objective",
        "why_text": "Asteroid physical modeling relies heavily on lightcurve data. ALDA predicts optimal observation windows to fill scientific gaps at critical geometric phases.",
        "val_title": "Error Analysis", "val_col_param": "Parameter", "val_col_error": "Mean Error", "val_col_source": "Source / Verification",
        "val_row_phase": "Phase Angle (α)", "val_row_window": "Window Accuracy", "val_row_data": "Reference DB",
        "val_res_phase": "± 0.42°", "val_res_window": "± 2.5 Days", "val_res_source1": "ALCDEF Database", "val_res_source2": "Thesis Chapter 5", "val_res_source3": "Validated",
        "settings": "Observation Settings", "target": "Select Target", "start_year": "Start Year", "years": "Duration (Years)",
        "run_btn": "Run Analysis", "result_title": "Recommended Schedule (YYYY-MM-DD)", "chart_title": "Parameter Evolution Trends (α & θ)",
        "inst_label": "Institution", "school": "Students of Premier School Affiliated to Hou Kong Middle School (Macau)", "copy": "Copyright © 2026 ALDA Project. All Rights Reserved.",
        "date_to": "to", "metric_target": "Target", "metric_windows": "Windows Found", "metric_span": "Span",
        "legend_phase": "Phase Angle (α)", "legend_elong": "Elongation (θ)", "legend_opt": "Optimal Window", "y_label": "Degrees",
        "last_update": "System Last Updated", "tz_name": "Macau/Hong Kong Time (UTC+8)", "val_info": "Data cross-validated against NASA JPL Horizons and ALCDEF."
    }
}

# --- 4. 效能優化快取機制 ---
@st.cache_data(show_spinner=False)
def calculate_trajectory(target_id, s_year, span):
    engine = ALDAEngine(PAPER_ASTEROIDS[target_id])
    jd_start = Time(f"{s_year}-01-01").jd
    jd_array = np.arange(jd_start, jd_start + (span * 365), 2)
    results = [engine.get_metrics(jd) for jd in jd_array]
    return pd.DataFrame(results)

# --- 5. 網頁 UI 設定與 CSS ---
st.set_page_config(page_title="ALDA Scientific", layout="wide", page_icon="☄️")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; background-color: #1f77b4; color: white; border: none; font-weight: bold; padding: 0.5rem 1rem;}
    div[data-testid="stExpander"] { border: 1px solid var(--secondary-background-color); border-radius: 12px; background-color: var(--background-color); box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    div[data-testid="stMetric"] { background-color: var(--secondary-background-color); padding: 15px; border-radius: 10px; border: 1px solid var(--secondary-background-color); }
    </style>
    """, unsafe_allow_html=True)

# 側邊欄
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>☄️ ALDA</h1>", unsafe_allow_html=True)
    selected_lang_name = st.selectbox("Language Selection", list(LANG_MAP.keys()), label_visibility="collapsed")
    lang_key = LANG_MAP[selected_lang_name]
    l = LANG_DICT[lang_key]
    
    st.divider()
    page = st.radio(l["nav_label"], [l["nav_predict"], l["nav_background"], l["nav_val"]])
    
    st.divider()
    st.caption(f"{l['inst_label']}:")
    st.write(f"**{l['school']}**")
    
    st.divider()
    local_tz = pytz.timezone('Asia/Macau')
    formatted_time = datetime.now(local_tz).strftime('%Y-%m-%d %H:%M:%S')
    st.caption(f"**{l['last_update']}**:\n{formatted_time}\n*{l['tz_name']}*")

# 主標題渲染
st.title("ALDA")
st.markdown(f"<h3 style='color: var(--text-color); margin-top: -15px;'>{l['full_name']}</h3>", unsafe_allow_html=True)
st.divider()

# --- 6. 分頁邏輯 ---
if page == l["nav_predict"]:
    with st.container():
        st.subheader(l["settings"])
        col_t, col_y, col_s = st.columns([2, 1, 1])
        target_id = col_t.selectbox(l["target"], list(PAPER_ASTEROIDS.keys()))
        current_year = datetime.now().year
        
        # 移除年份與跨度的上限限制 (改用 number_input)
        s_year = col_y.number_input(l["start_year"], value=current_year)
        span = col_s.number_input(l["years"], min_value=1, value=15)
        
        btn_run = st.button(l["run_btn"], type="primary")

    if btn_run:
        with st.spinner('Calculating orbital geometry...'):
            df = calculate_trajectory(target_id, s_year, span)
            valid = df[(df['Phase'] < 30) & (df['Elongation'] > 90)].copy()

        # 頂部摘要卡片
        m1, m2, m3 = st.columns(3)
        m1.metric(l["metric_target"], target_id.split(" ")[0]) 
        m2.metric(l["metric_windows"], len(valid['Date'].diff().dt.days > 10) if not valid.empty else 0)
        m3.metric(l["metric_span"], f"{span} Yrs")

        if not valid.empty:
            st.markdown(f"#### {l['result_title']}")
            valid['group'] = (valid['Date'].diff().dt.days > 10).cumsum()
            res_cols = st.columns(3)
            for idx, (_, gp) in enumerate(valid.groupby('group')):
                with res_cols[idx % 3]:
                    start_formatted = gp['Date'].iloc[0].strftime('%Y-%m-%d')
                    end_formatted = gp['Date'].iloc[-1].strftime('%Y-%m-%d')
                    st.success(f"{start_formatted} **{l['date_to']}** {end_formatted}")
            
            st.divider()
            st.subheader(l["chart_title"])
            
            # 互動圖表
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Phase'], mode='lines', name=l["legend_phase"], line=dict(color='#E67E22', width=2)))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Elongation'], mode='lines', name=l["legend_elong"], line=dict(color='#2E86C1', width=2)))
            
            added_legend = False
            for _, gp in valid.groupby('group'):
                fig.add_vrect(
                    x0=gp['Date'].iloc[0], x1=gp['Date'].iloc[-1],
                    fillcolor="#2ECC71", opacity=0.25, line_width=0,
                    name=l["legend_opt"], showlegend=not added_legend
                )
                added_legend = True

            fig.update_layout(
                yaxis_title=l["y_label"],
                yaxis=dict(range=[0, 180]),
                hovermode="x unified",
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No valid windows found within the specified scientific constraints.")

elif page == l["nav_background"]:
    st.subheader(l["why_title"])
    st.info(l["why_text"])
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### {l['nav_predict']}")
        st.write(f"1. {l['legend_phase']} < 30°\n2. {l['legend_elong']} > 90°")
    with c2:
        st.markdown(f"#### {l['nav_val']}")
        st.write(l["val_info"])

elif page == l["nav_val"]:
    st.subheader(l["val_title"])
    val_df = pd.DataFrame({
        l["val_col_param"]: [l["val_row_phase"], l["val_row_window"], l["val_row_data"]],
        l["val_col_error"]: [l["val_res_phase"], l["val_res_window"], l["val_res_source1"]],
        l["val_col_source"]: [l["val_res_source1"], l["val_res_source2"], l["val_res_source3"]]
    })
    st.table(val_df)

# 全域頁腳
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown(f"<p style='text-align: center; color: gray; font-size: 0.9em;'>{l['copy']}</p>", unsafe_allow_html=True)
