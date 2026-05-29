import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import subprocess
import time


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

    st.session_state.conf_threshold = conf_threshold

    tab1, tab2, tab3, tab4 = st.tabs([
    "이미지 분석",
    "영상 빠른 분석",
    "영상 정밀 분석",
    "실시간 분석"
])
    
    st.divider()
    st.write("원하는 분석 방식을 선택하세요.")
    colum1, colum2 = st.columns(2)

    # -----------------------------------
    # 이미지 분석 버튼
    # -----------------------------------

    with colum1:

        st.subheader("🖼 이미지 분석")

        st.write("딸기 이미지를 업로드하여 병해충을 탐지합니다.")

        st.info("정지 이미지를 빠르게 분석")

        if st.button(
            "이미지 분석",
            use_container_width=True
        ):

            st.session_state.page = "image"

            st.rerun()
            
        

    # -----------------------------------
    # 동영상 분석 버튼
    # -----------------------------------
    
    with colum2:

        st.subheader("🎥 동영상 분석")

        st.write("딸기 동영상을 업로드하여 병해충을 탐지합니다.")

        st.info("넓은 구역 병해 탐지에 추천")

        if st.button(
            "동영상 분석",
            use_container_width=True
        ):

            st.session_state.page = "video"

            st.rerun()

elif st.session_state.page == "image":

    st.title("🖼 이미지 병해충 분석")

    colum1, colum2 = st.columns(2)

    with colum1:
        
        uploaded_file = st.file_uploader("이미지 업로드")

    with colum2:

        camera_image = st.camera_input("사진 촬영")

    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file

        st.success("✅ 이미지 업로드 완료")

        # 결과 페이지로 이동
        st.session_state.page = "result"

        st.rerun()

    elif camera_image:
        st.session_state.uploaded_file = camera_image

        st.success("✅ 이미지 업로드 완료")

        # 결과 페이지로 이동
        st.session_state.page = "result"

        st.rerun()
        
# -----------------------------------
# 동영상 분석 페이지
# -----------------------------------

elif st.session_state.page == "video":

    st.title("🎥 동영상 병해충 분석")
    # 동영상 업로드
    uploaded_video_file = st.file_uploader(
        "동영상을 업로드하세요",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video_file is not None:

        st.session_state.uploaded_file = uploaded_video_file

        st.success("✅ 동영상 업로드 완료")

        # 결과 페이지로 이동
        st.session_state.page = "result"

        st.rerun()

# -----------------------------------
# 결과 페이지
# -----------------------------------

elif st.session_state.page == "result":

    st.title("📊 분석 결과")

    uploaded_file = st.session_state.uploaded_file
    conf_threshold = st.session_state.conf_threshold

    file_type = uploaded_file.type

    # 이미지인 경우
    if "image" in file_type:

        image = Image.open(uploaded_file)
    
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

    # 동영상인 경우
    elif "video" in file_type:
        
        # 임시 파일로 저장
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
    
        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # fps 오류 방지
        if fps == 0:
            fps = 30

        # -----------------------------
        # 결과 영상 저장 경로
        # -----------------------------

        temp_output = "temp_result.mp4"
    
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
        out = cv2.VideoWriter(
            temp_output,
            fourcc,
            fps,
            (width, height)
        )

        with st.spinner("AI가 병해충을 분석중입니다..."):
            @st.cache_resource
            def load_model():
                return YOLO("best.pt")
            
            model = load_model()
    
        # -----------------------------
        # 진행률 표시
        # -----------------------------
        progress_bar = st.progress(0)

        start_time = time.time()

        status_text = st.empty()

        preview_frame = st.empty()

        frame_idx = 0
    
        # -----------------------------
        # 프레임 처리
        # -----------------------------
        while cap.isOpened():
            ret, frame = cap.read()
    
            if not ret:
                break

            # YOLO 추론
            results = model(frame)

            # bbox 그려진 결과 프레임
            annotated_frame = results[0].plot()

            # 저장
            out.write(annotated_frame)

            frame_idx += 1
    
            progress = int(frame_idx / total_frames * 100)
            progress_bar.progress(progress)
        
            # -----------------------------
            # 시간 계산
            # -----------------------------
            elapsed_time = time.time() - start_time
        
            fps_processing = frame_idx / elapsed_time
        
            remaining_frames = total_frames - frame_idx
        
            remaining_time = remaining_frames / fps_processing
        
            # -----------------------------
            # 상태 표시
            # -----------------------------
            status_text.text(
                f"""
                처리 프레임: {frame_idx}/{total_frames}
                처리 FPS: {fps_processing:.2f}
                경과 시간: {elapsed_time:.1f}초
                남은 예상 시간: {remaining_time:.1f}초
                """
            )

            preview_frame.image(
                annotated_frame,
                channels="BGR"
            )
        # 종료
        cap.release()
        out.release()
    
        st.success("분석 완료!")
    
        # -----------------------------
        # H.264 변환
        # -----------------------------
        final_output = "final_result.mp4"
    
        command = [
            "ffmpeg",
            "-y",
            "-i",
            temp_output,
            "-vcodec",
            "libx264",
            "-acodec",
            "aac",
            final_output
        ]
    
        subprocess.run(command)
    
        st.success("영상 생성 완료!")    

    
        # -----------------------------
        # 결과 영상 표시
        # -----------------------------
        st.video(final_output)
    
        # -----------------------------
        # 다운로드 버튼
        # -----------------------------
        with open(final_output, "rb") as file:
            st.download_button(
                label="결과 영상 다운로드",
                data=file,
                file_name="result.mp4",
                mime="video/mp4"
            )
    
#     # 동영상인 경우
#     elif "video" in file_type:
        
#         # 임시 파일로 저장
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(uploaded_file.read())
    
#         cap = cv2.VideoCapture(tfile.name)
        
#         fps = cap.get(cv2.CAP_PROP_FPS)

#         with st.spinner("AI가 병해충을 분석중입니다..."):
#             @st.cache_resource
#             def load_model():
#                 return YOLO("best.pt")
            
#             model = load_model()
    
#             frame_count = 0
#             saved_count = 0
    
#             detection_counts = 0
    
#             detected_classes = set()
            
#             progress_bar = st.progress(0)
    
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#             while True:
#                 ret, frame = cap.read()
            
#                 if not ret:
#                     break
    
#                 # 진행률 표시
#                 progress = min(frame_count / total_frames, 1.0)
    
#                 progress_bar.progress(progress)
    
            
#                 # 1초마다 1프레임 저장
#                 if frame_count % int(fps) == 0:
            
#                     results = model(frame, conf=conf_threshold)
    
#                     plotted = results[0].plot()
            
#                     col1, col2 = st.columns(2)
                
#                     with col1:
#                         st.image(plotted)
                
#                     if len(results[0].boxes) > 0:
    
#                         detection_counts += 1
                
#                         best_idx = results[0].boxes.conf.argmax()
                    
#                         class_id = int(results[0].boxes.cls[best_idx])
    
#                         info = disease_info[class_id]
    
#                         detected_classes.add(class_id)
                    
#                         conf = float(results[0].boxes.conf[best_idx])
                        
#                         with col2:
#                             st.subheader(info["explain"])
                
#                             confidence = float(conf)
#                             st.progress(confidence)
#                             st.write(f"신뢰도: {conf:.2f}")
    
#                     else:
#                         with col2:
#                             st.subheader("탐지된 병해충이 없습니다.")
#                             st.success("건강한 딸기로 보입니다 🍓")
    
#                 frame_count += 1
    
#             progress_bar.empty()
    
#             # -----------------------------------
#             # 결과 출력
#             # -----------------------------------
    
#             st.header("📊 병해충 탐지 결과")

#             st.info(
#     f"현재 신뢰도 임계값 (Confidence Threshold): {conf_threshold}"
# )
    
#             if detection_counts == 0:
    
#                 st.success("✅ 병해충이 탐지되지 않았습니다.")
    
#             else:
#                 for class_id in detected_classes:
    
#                     info = disease_info[class_id]
#                     st.header(f"🩺 {info["name"]} 병해충 정보")
    
#                     with st.container(border=True):
#                         # with st.expander("🎯 병해 상세 정보"):
#                             st.write(info["symptom"])
                        
#                             st.write(info["cause"])
                
#                             st.write(info["solution"])
                
#                             st.write("🍓 병해 예시 이미지")
#                             st.image(info["image"])
#                             st.caption(info["name"])

    if st.button("🔙 처음으로"):

        st.session_state.page = "upload"

        st.session_state.uploaded_file = None

        st.rerun()


st.markdown("---")

st.caption("YOLO 기반 딸기 병해충 진단 시스템")

