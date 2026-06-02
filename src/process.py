import streamlit as st
from PIL import Image
import cv2

from src import utility
from src.disease_data import disease_info

def process_image(uploaded_file, model, conf_threshold):
  image = Image.open(uploaded_file)
  
  st.divider()
  
  results = model(image, conf=conf_threshold)
  
  class_id, detection = utility.show_detection_result(results)
  
  if detection ==True:
      utility.show_disease_info(class_id)

def process_fast_video(video_path, model, conf_threshold):
  
  video_info_dic = utility.get_video_info(video_path)
  
  cap = cv2.VideoCapture(video_path)
  
  frame_count = 0
  saved_count = 0
  
  detection_counts = 0
  
  detected_classes = set()
  
  progress_bar = st.progress(0)
  
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
  
          class_id, detection = utility.show_detection_result(results)
  
          if detection ==True:
              detection_counts += 1
              detected_classes.add(class_id)
              
      frame_count += 1
  
  progress_bar.empty()
  
  # -----------------------------------
  # 결과 출력
  # -----------------------------------
  
  st.header("📊 병해충 탐지 결과")
  
  st.info(
  f"현재 신뢰도 임계값 (Confidence Threshold): {conf_threshold}"
  )
  
  if detection_counts == 0:
  
      st.success("✅ 병해충이 탐지되지 않았습니다.")
  
  else:
      for class_id in detected_classes:
          utility.show_disease_info(class_id)

def process_precise_video(video_path, model, conf_threshold):
  
  video_info_dic = utility.get_video_info(video_path)
  
  cap = cv2.VideoCapture(video_path)
  
  # -----------------------------
  # 결과 영상 저장 경로
  # -----------------------------
  
  temp_output = "temp_result.mp4"
  
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
  
  detection_counts = 0
  
  detected_classes = set()
  
  
  # -----------------------------
  # 프레임 처리
  # -----------------------------
  while cap.isOpened():
      ret, frame = cap.read()
  
      if not ret:
          break
  
      # YOLO 추론
      results = model(frame)
  
      if len(results[0].boxes) > 0:
  
          detection_counts += 1
  
          best_idx = results[0].boxes.conf.argmax()
      
          class_id = int(results[0].boxes.cls[best_idx])
  
          info = disease_info[class_id]
  
          detected_classes.add(class_id)
  
      # bbox 그려진 결과 프레임
      annotated_frame = results[0].plot()
  
      # 저장
      out.write(annotated_frame)
  
      frame_idx += 1
  
      progress = int(frame_idx / video_info_dic["total_frames"] * 100)
      progress_bar.progress(progress)
  
      # -----------------------------
      # 시간 계산
      # -----------------------------
      elapsed_time = time.time() - start_time
  
      fps_processing = frame_idx / elapsed_time
  
      remaining_frames = video_info_dic["total_frames"] - frame_idx
  
      remaining_time = remaining_frames / fps_processing
  
      # -----------------------------
      # 상태 표시
      # -----------------------------
      status_text.text(
          f"""
          처리 프레임: {frame_idx}/{video_info_dic["total_frames"]}
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
  
  
  # -----------------------------------
  # 결과 출력
  # -----------------------------------
  
  st.header("📊 병해충 탐지 결과")
  
  st.info(
  f"현재 신뢰도 임계값 (Confidence Threshold): {conf_threshold}"
  )
  
  if detection_counts == 0:
  
      st.success("✅ 병해충이 탐지되지 않았습니다.")
  
  else:
      for class_id in detected_classes:
          utility.show_disease_info(class_id)
  
