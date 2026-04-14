# --- 修正後的側邊欄佔位邏輯 ---
with st.sidebar:
    st.markdown("<div style='padding: 20px 0;'><h2 style='text-align: center;'>ALDA</h2></div>", unsafe_allow_html=True)
    sel_lang = st.selectbox("Language", list(LANG_MAP.keys()), label_visibility="collapsed")
    l = LANG_DICT[LANG_MAP[sel_lang]]
    
    st.divider()
    page = st.radio("Navigation", [l["nav_predict"], l["nav_background"], l["nav_val"]], label_visibility="collapsed")
    
    # 這裡修正了錯誤：使用空白容器或 HTML 來增加間距
    st.markdown("<br>" * 10, unsafe_allow_html=True) 
    
    st.markdown(f"<div style='color: #86868b; font-size: 0.8rem; text-align: center;'>{l['school']}</div>", unsafe_allow_html=True)
