import cv2
import streamlit as st

def show_disease_info(class_id):

    info = disease_info[class_id]

    st.header(f"🩺 {info['name']}  병해충 정보")

    with st.container(border=True):

        st.write(info["symptom"])
        st.write(info["cause"])
        st.write(info["solution"])

        st.write("🍓 병해 예시 이미지")

        st.image(info["image"])

        st.caption(info["name"])

def show_detection_result(results):
    detection = False
    class_id = None

    col1, col2 = st.columns(2)

    with col1:
        st.image(results[0].plot())
    
    with col2:
        if len(results[0].boxes) == 0:
            st.subheader("탐지된 병해충이 없습니다.")
            st.success("건강한 딸기로 보입니다 🍓")
    
        else:
            detection = True
            
            best_idx = results[0].boxes.conf.argmax()
        
            class_id = int(results[0].boxes.cls[best_idx])
        
            conf = float(results[0].boxes.conf[best_idx])
        
            info = disease_info[class_id]
    
            st.subheader(info["explain"])
    
            st.progress(conf)
    
            st.write(f"신뢰도: {conf:.2f}")

    return class_id, detection


def get_video_info(video_path):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    duration = total_frames / fps

    cap.release()

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration": duration
    }
