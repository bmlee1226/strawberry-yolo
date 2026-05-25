import streamlit as st
from ultralytics import YOLO
from PIL import Image

model = YOLO("best.pt")

st.title("딸기 병해충 탐지")

uploaded_file = st.file_uploader("이미지 업로드")
camera_image = st.camera_input("사진 촬영")

disease_info = {
    0: "잿빛곰팡이병 (gray_mold) 입니다.",
    1: "흰가루병 (powdery_mildew) 입니다.",
}

if uploaded_file:

    image = Image.open(uploaded_file)

elif camera_image:

    image = Image.open(camera_image)

if (uploaded_file) | (camera_image):

    results = model(image)

    plotted = results[0].plot()

    st.image(plotted)

    if len(results[0].boxes) > 0:

        best_idx = results[0].boxes.conf.argmax()
    
        class_id = int(results[0].boxes.cls[best_idx])
    
        conf = float(results[0].boxes.conf[best_idx])
    
        st.write(f"클래스: {disease_info[class_id]}")
        st.write(f"신뢰도: {conf:.2f}")
    
    else:
        st.warning("탐지된 병해충이 없습니다.")


