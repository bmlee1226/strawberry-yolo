import streamlit as st
from ultralytics import YOLO
from PIL import Image

model = YOLO("best.pt")

st.title("딸기 병해충 탐지")

st.markdown("""
딸기 이미지를 업로드하면  
AI가 병해충을 탐지하고 원인 및 해결 방법을 안내합니다.
""")

st.info("현재 지원: 흰가루병, 잿빛곰팡이병")

st.warning("""
본 결과는 AI 예측이며
정확한 진단은 전문가 확인이 필요합니다.
""")

colum1, colum2 = st.columns(2)

with colum1:

    uploaded_file = st.file_uploader("이미지 업로드")

with colum2:
    
    camera_image = st.camera_input("사진 촬영")

disease_info = {
    0: "잿빛곰팡이병 (gray_mold) 입니다.",
    1: "흰가루병 (powdery_mildew) 입니다.",
}

disease_info = {
    0: {
        "name": "잿빛곰팡이병 (gray_mold) 입니다.",
        "cause": "🦠 원인: 습도가 높고 통풍이 부족할 때 발생",
        "solution": "💊 해결책: 습도 조절 및 병든 부위 제거",
        "image": "gray_mold.png"
    },
    
    1: {
        "name": "흰가루병 (powdery_mildew) 입니다.",
        "cause": "🦠 원인: 고온다습한 환경에서 발생",
        "solution": "💊 해결책: 환기 개선 및 감염 잎 제거",
        "image": "powdery_mildew.jpg"
    },
    
    "healthy": {
        "name": "정상입니다.",
        "cause": "🦠 원인:건강한 상태",
        "solution": "💊 해결책:현재 상태 유지",
        "image": "healthy.png"
    }
}



if uploaded_file:

    image = Image.open(uploaded_file)

elif camera_image:

    image = Image.open(camera_image)

if uploaded_file or camera_image:

    with st.spinner("AI가 병해충을 분석중입니다..."):
        results = model(image)

    plotted = results[0].plot()

    col1, col2 = st.columns(2)

    with col1:
        st.image(plotted)

    if len(results[0].boxes) > 0:

        best_idx = results[0].boxes.conf.argmax()
    
        class_id = int(results[0].boxes.cls[best_idx])
    
        conf = float(results[0].boxes.conf[best_idx])

        info = disease_info[class_id]
        
        with col2:
            st.subheader(info["name"])

            confidence = float(conf)
            st.progress(confidence)
            st.write(f"신뢰도: {conf:.2f}")

        with st.expander("병해 상세 정보"):
            st.write(info["cause"])

            st.write(info["solution"])

            st.write("<예시 이미지>")
    
            st.image(info["image"])
    
    else:
        with col2:
            st.subheader("탐지된 병해충이 없습니다.")

st.markdown("---")

st.caption("YOLO 기반 딸기 병해충 진단 시스템")

