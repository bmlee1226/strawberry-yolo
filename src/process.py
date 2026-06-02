import streamlit as st
from PIL import Image
import cv2

import time
import tempfile

from src import utility
from src.disease_data import disease_info

def process_image(uploaded_file, model, conf_threshold):
  image = Image.open(uploaded_file)
  
  st.divider()
  
  result_list = []
  
  results = model(image, conf=conf_threshold)
  
  class_id, conf, detection = utility.parse_detection_result(results)
  result_list += [results, class_id, conf, detection]

  st.session_state.result_list = result_list

def process_fast_video(video_path, model, conf_threshold):
  
  video_info_dic = utility.get_video_info(video_path)
  
  cap = cv2.VideoCapture(video_path)
  
  frame_count = 0

  detection_frame_count = 0
  
  detected_classes = set()
  
  progress_bar = st.progress(0)

  result_list = []
  
  while True:
      ret, frame = cap.read()
  
      if not ret:
          break
  
      # 진행률 표시
      progress = min(frame_count / video_info_dic["total_frames"], 1.0)
  
      progress_bar.progress(progress)
  
  
      # 1초마다 1프레임 저장
      if frame_count % int(video_info_dic["fps"]) == 0:
  
          results = model(frame, conf=conf_threshold)
  
          class_id, conf, detection = utility.parse_detection_result(results)
          result_list += [[results, class_id, conf, detection]]
  
          if detection:
              detection_frame_count += 1
              detected_classes.add(class_id)
              
      frame_count += 1
  
  progress_bar.empty()
  cap.release()

  st.session_state.result_list = result_list
  st.session_state.detection_frame_count = detection_frame_count
  st.session_state.detected_classes = detected_classes
  st.session_state.conf_threshold = conf_threshold


def process_precise_video(video_path, model, conf_threshold):
  
  video_info_dic = utility.get_video_info(video_path)
  
  cap = cv2.VideoCapture(video_path)
  
  # -----------------------------
  # 결과 영상 저장 경로
  # -----------------------------
  
  temp_output = tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".mp4"
).name
  
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  
  out = cv2.VideoWriter(
      temp_output,
      fourcc,
      video_info_dic["fps"],
      (video_info_dic["width"], video_info_dic["height"])
  )
  
  # -----------------------------
  # 진행률 표시
  # -----------------------------
  progress_bar = st.progress(0)
  
  start_time = time.time()
  
  status_text = st.empty()
  
  preview_frame = st.empty()
  
  frame_idx = 0
  
  detection_frame_count = 0
  
  detected_classes = set()
  
  
  # -----------------------------
  # 프레임 처리
  # -----------------------------
  while cap.isOpened():
      ret, frame = cap.read()
  
      if not ret:
          break
  
      # YOLO 추론
      results = model(frame, conf=conf_threshold)
      class_id, conf, detection = utility.parse_detection_result(results)

      if class_id:
        detected_classes.add(class_id)
        detection_frame_count += 1
  
      # bbox 그려진 결과 프레임
      annotated_frame = results[0].plot()
  
      # 저장
      out.write(annotated_frame)
  
      frame_idx += 1
  
      progress = frame_idx / video_info_dic["total_frames"]
      progress_bar.progress(progress)
  
      # -----------------------------
      # 시간 계산
      # -----------------------------
      elapsed_time = time.time() - start_time
  
      fps_processing = frame_idx / max(elapsed_time, 0.001)
  
      remaining_frames = video_info_dic["total_frames"] - frame_idx
  
      remaining_time = remaining_frames / fps_processing
  
      # -----------------------------
      # 상태 표시
      # -----------------------------

      if frame_idx % 5 == 0:
        status_text.text(
            f"""
            처리 프레임: {frame_idx}/{video_info_dic["total_frames"]}
            처리 FPS: {fps_processing:.2f}
            경과 시간: {elapsed_time:.1f}초
            남은 예상 시간: {remaining_time:.1f}초
            """
        )
    
      if frame_idx % 10 == 0:
      
          preview_frame.image(
              annotated_frame,
              channels="BGR"
          )
        
  # 종료
  cap.release()
  out.release()

  st.session_state.detection_frame_count = detection_frame_count
  st.session_state.detected_classes = detected_classes
  st.session_state.temp_output = temp_output
  st.session_state.conf_threshold = conf_threshold
  
  st.success("분석 완료!")
  
  
