import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import subprocess
import time
from src import pages

# -----------------------------------
# session_state 초기화
# -----------------------------------

if "page" not in st.session_state:
    st.session_state.page = "home"

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# -----------------------------------
# 업로드 페이지
# -----------------------------------

if st.session_state.page == "home":
    pages.page_home()


elif st.session_state.page == "image":
    pages.page_image()
        
# -----------------------------------
# 동영상 분석 페이지
# -----------------------------------

elif st.session_state.page == "video":
    pages.page_video()


# -----------------------------------
# 결과 페이지
# -----------------------------------

elif st.session_state.page == "result":
    pages.page_result()

st.markdown("---")

st.caption("YOLO 기반 딸기 병해충 진단 시스템")

