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

    results = model(image)

    plotted = results[0].plot()

    st.image(plotted)

    best_idx = results[0].boxes.conf.argmax()

    class_name = int(results[0].boxes.cls[best_idx])

    st.write(disease_info[class_name])

if camera_image:

    image = Image.open(camera_image)

    results = model(image)

    plotted = results[0].plot()

    st.image(plotted)

    best_idx = results[0].boxes.conf.argmax()

    class_name = int(results[0].boxes.cls[best_idx])

    st.write(disease_info[class_name])
