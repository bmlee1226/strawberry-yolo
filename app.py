import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2

st.title("🍓 딸기 병해충 진단 AI")

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
    st.image("gray_mold.png", use_container_width=True)

with col_b:
    st.image("powdery_mildew.jpg", use_container_width=True)

with col_c:
    st.image("healthy.png", use_container_width=True)

with st.sidebar:

    st.header("⚙️ 설정")

    st.divider()

    conf_threshold = st.slider(
        "신뢰도 임계값 (Confidence Threshold)",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.05
    )

    st.caption("""
        값이 낮을수록 더 많은 병해를 탐지하지만
오탐 가능성이 증가할 수 있습니다.
        """)

st.divider()

colum1, colum2 = st.columns(2)

with colum1:

    uploaded_file = st.file_uploader("이미지 업로드")

with colum2:
    
    camera_image = st.camera_input("사진 촬영")

# 동영상 업로드
uploaded_video_file = st.file_uploader(
    "동영상을 업로드하세요",
    type=["mp4", "avi", "mov"]
)

disease_info = {
    0: "잿빛곰팡이병 (gray_mold) 입니다.",
    1: "흰가루병 (powdery_mildew) 입니다.",
}

disease_info = {
    0: {
        "name": "잿빛곰팡이병 (gray_mold)",
        "explain": "잿빛곰팡이병 (gray_mold) 입니다.",
        "symptom": """🍃 증상: 딸기의 잎, 꽃, 과실 등 지상부 전반에 발생할 수 있지만
특히 과실 피해가 가장 심한 병해입니다.

초기에는 상처 부위나 노화된 조직에서 물러지는 증상이 나타나며,
병이 진행되면 과실 표면에 회색 또는 쥐털 모양의 곰팡이가
무수히 발생합니다.

잎은 가장자리부터 마르거나 썩기 시작하며,
과실은 착색이 시작되는 시기에 피해가 급격히 증가합니다.

특히 개화기에 꽃잎과 꽃받침에 감염된 후
어린 과실로 전염되며,
과실이 커질수록 병반도 함께 확대됩니다.""",
        "cause": """🦠 원인: 시설 내부의 높은 습도와 부족한 환기 환경에서 주로 발생합니다.

병원균은 병든 식물 잔재물, 시설 자재 등에서
균핵 또는 포자 형태로 월동한 후,
봄철 습한 환경에서 다시 활성화됩니다.

발생한 포자는 바람이나 물방울을 통해 빠르게 확산되며,
밀식 재배나 과습 환경에서는 피해가 크게 증가할 수 있습니다.""",
        "solution": """💊 해결책: 
- 병든 과실과 잎은 즉시 제거하여 전염원을 줄입니다.
- 시설 내부 환기를 강화하여 습도를 낮춥니다.
- 과습 상태가 지속되지 않도록 관수량을 조절합니다.
- 밀식 재배를 피하고 통풍이 잘 되도록 관리합니다.
- 개화기와 과실 비대기에는 예방 위주의 약제 방제를 실시합니다.
- 병 발생 초기 등록 약제를 살포하면 확산 억제에 도움이 됩니다.""",
        "image": "gray_mold.png"
    },
    
    1: {
        "name": "흰가루병 (powdery_mildew)" ,
        "explain": "흰가루병 (powdery_mildew) 입니다.",
        "symptom": """🍃 증상: 흰가루병은 딸기의 잎, 새순, 꽃, 과실 등
지상부 전체에 발생할 수 있는 대표적인 병해입니다.

특히 새순과 과실 피해가 심하며,
잎 뒷면에 흰색 가루 형태의 곰팡이가 나타나는 것이 특징입니다.

증상이 심해지면 잎과 과실 표면이
밀가루를 뿌린 것처럼 하얗게 덮이며,
과실의 비대가 정상적으로 이루어지지 않을 수 있습니다.

꽃봉오리에 감염되면 꽃잎이 자홍색으로 변형되거나
생육이 불량해질 수 있습니다.""",
        "cause": """🦠 원인: 시설 내 온도 변화가 크고 통풍이 부족할 때 발생이 증가합니다.

질소질 비료 과다 사용,
건조한 환경과 불균형한 습도 조건에서도 발생하기 쉽습니다.

병원균은 병든 잎과 식물체에서 월동하며,
바람을 통해 빠르게 전염됩니다.""",
        "solution": """💊 해결책: 
- 병든 잎과 과실은 조기에 제거하여 전염을 차단합니다.
- 재배 후 잔재물을 정리하여 시설 위생을 유지합니다.
- 충분한 관수와 환기 관리로 급격한 건조 환경을 피합니다.
- 질소질 비료 과다 사용을 줄이고 균형 시비를 실시합니다.
- 초기 발생 시 약제를 예방적으로 살포합니다.
- '여홍', '여봉', '풍향', '춘향' 등 품종은
  특히 주기적인 예찰과 방제가 중요합니다.""",
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


if uploaded_file:

    image = Image.open(uploaded_file)

elif camera_image:

    image = Image.open(camera_image)

elif uploaded_video_file is not None:

    st.success("업로드 완료!")

    # 영상 재생
    st.video(uploaded_file)

    # 임시 파일로 저장
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.write("임시 저장 경로:")
    st.write(tfile.name)

    cap = cv2.VideoCapture(tfile.name)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    st.write(f"FPS: {fps}")



if uploaded_file or camera_image:
    st.divider()

    with st.spinner("AI가 병해충을 분석중입니다..."):

        @st.cache_resource
        def load_model():
            return YOLO("best.pt")
        
        model = load_model()
        
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
            # with st.expander("🎯 병해 상세 정보"):
                st.write(info["symptom"])
            
                st.write(info["cause"])
    
                st.write(info["solution"])
    
                st.write("🍓 병해 예시 이미지")
                st.image(info["image"])
                st.caption(info["name"])
        
        else:
            with col2:
                st.subheader("탐지된 병해충이 없습니다.")
                st.success("건강한 딸기로 보입니다 🍓")

if uploaded_video_file is not None:

    with st.spinner("AI가 병해충을 분석중입니다..."):
        @st.cache_resource
        def load_model():
            return YOLO("best.pt")
        
        model = load_model()

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
        
            if not ret:
                break
        
            # 1초마다 1프레임 저장
            if frame_count % int(fps) == 0:
        
                results = model(frame, conf=conf_threshold)
        
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
                    # with st.expander("🎯 병해 상세 정보"):
                        st.write(info["symptom"])
                    
                        st.write(info["cause"])
            
                        st.write(info["solution"])
            
                        st.write("🍓 병해 예시 이미지")
                        st.image(info["image"])
                        st.caption(info["name"])
                
                else:
                    with col2:
                        st.subheader("탐지된 병해충이 없습니다.")
                        st.success("건강한 딸기로 보입니다 🍓")

            frame_count += 1
        
        


st.markdown("---")

st.caption("YOLO 기반 딸기 병해충 진단 시스템")

