import streamlit as st
from src import pages

HOME_PAGE = "home"
IMAGE_PAGE = "image"
VIDEO_PAGE = "video"
RESULT_PAGE = "result"
  
def router():

    page = st.session_state.page
    
    if page == HOME_PAGE:
        pages.page_home()
    
    elif page == IMAGE_PAGE:
        pages.page_image()
            
    elif page == VIDEO_PAGE:
        pages.page_video()
    
    elif page == RESULT_PAGE:
        pages.page_result()

def init_session_state():

    defaults = {
        "page": "home",
        "uploaded_file": None,
        "video_path": None,
        "analysis_type": None,
        "conf_threshold": 0.3
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value

def render_footer():

    st.markdown("---")

    st.caption(
        "YOLO 기반 딸기 병해충 진단 시스템"
    )
