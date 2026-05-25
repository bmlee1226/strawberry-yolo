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

st.subheader("예시 이미지")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.image("gray_mold.jpg", use_container_width=True)

with col_b:
    st.image("powdery_mildew.jpg", use_container_width=True)

with col_c:
    st.image("healthy.png", use_container_width=True)

with st.sidebar:

    st.header("⚙️ 설정")

    st.divider()

    conf_threshold = st.slider(
        "신뢰도 임계값 (Confidence Threshold)",
        0.1,
        1.0,
        0.5
    )

    st.caption("""
        값이 낮을수록 더 많은 객체를 탐지합니다.
        """)

colum1, colum2 = st.columns(2)

st.divider()

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
        "name": "잿빛곰팡이병 (gray_mold)",
        "explain": "잿빛곰팡이병 (gray_mold) 입니다.",
        "cause": "🦠 원인: 습도가 높고 통풍이 부족할 때 발생",
        "solution": "💊 해결책: 습도 조절 및 병든 부위 제거",
        "image": "gray_mold.png"
    },
    
    1: {
        "name": "흰가루병 (powdery_mildew)" ,
        "explain": "흰가루병 (powdery_mildew) 입니다.",
        "cause": "🦠 원인: 고온다습한 환경에서 발생",
        "solution": "💊 해결책: 환기 개선 및 감염 잎 제거",
        "image": "powdery_mildew.jpg"
    },
    
    "healthy": {
        "name": "정상",
        "explain": "정상입니다.",
        "cause": "🦠 원인:건강한 상태",
        "solution": "💊 해결책:현재 상태 유지",
        "image": "healthy.png"
    }
}

st.divider()

if uploaded_file:

    image = Image.open(uploaded_file)

elif camera_image:

    image = Image.open(camera_image)

if uploaded_file or camera_image:

    with st.spinner("AI가 병해충을 분석중입니다..."):
        results = model(image, conf=conf_threshold)

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
            st.subheader(info["explain"])

            confidence = float(conf)
            st.progress(confidence)
            st.write(f"신뢰도: {conf:.2f}")

        with st.container(border=True):
        # with st.expander("병해 상세 정보"):
            st.write(info["cause"])

            st.write(info["solution"])

            with st.container(border=True):
                st.write("예시 이미지")
                st.image(info["image"])
                st.caption("흰가루병")
    
    else:
        with col2:
            st.subheader("탐지된 병해충이 없습니다.")
            st.success("건강한 딸기로 보입니다 🍓")
            st.balloons()

st.markdown("---")

st.caption("YOLO 기반 딸기 병해충 진단 시스템")

