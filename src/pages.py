import streamlit as st
from ultralytics import YOLO

import tempfile
import subprocess

from src import process
from src import utility

with st.spinner("AI가 병해충을 분석중입니다..."):

    @st.cache_resource
    def load_model():
        return YOLO("best.pt")
    
    model = load_model()


def page_home():
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
  
          go_to("image")
          
      
  
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
  
          go_to("video")

def page_image():
  
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
      go_to("analysis")
  
  elif camera_image:
      st.session_state.uploaded_file = camera_image
  
      st.success("✅ 이미지 업로드 완료")
  
      # 결과 페이지로 이동
      go_to("analysis")

def page_video():
  
  st.title("🎥 동영상 병해충 분석")
  # 동영상 업로드
  uploaded_video_file = st.file_uploader(
      "동영상을 업로드하세요",
      type=["mp4", "avi", "mov"]
  )
  
  if uploaded_video_file is not None:
  
      video_bytes = uploaded_video_file.read()
      st.session_state.uploaded_file = uploaded_video_file
  
      st.success("✅ 동영상 업로드 완료")
  
      # -----------------------------
      # 임시 저장
      # -----------------------------
      tfile = tempfile.NamedTemporaryFile(
          delete=False,
          suffix=".mp4"
      )
  
      tfile.write(video_bytes)
      tfile.close()
  
      # session_state 저장
      video_path = tfile.name
      st.session_state.video_path = video_path
 
      # -----------------------------
      # 영상 정보 읽기
      # -----------------------------
  
      video_info_dic = utility.get_video_info(video_path)
  
      # -----------------------------
      # 영상 정보 표시
      # -----------------------------
      st.video(video_bytes)
  
      st.subheader("영상 정보")
  
      col1, col2, col3 = st.columns(3)
  
      with col1:
          st.metric("FPS", f"{video_info_dic['fps']:.1f}")
  
      with col2:
          st.metric("총 프레임", video_info_dic['total_frames'])
  
      with col3:
          st.metric("영상 길이", f"{video_info_dic['duration']:.1f}초")
  
      # -----------------------------
      # 예상 소요 시간 계산
      # -----------------------------
      # 예시 기준:
      # 빠른 분석 = 1초당 1프레임
      # 정밀 분석 = 모든 프레임
  
      fast_analysis_frames = int(video_info_dic["duration"])
  
      precise_analysis_frames = video_info_dic["total_frames"]
  
      # 대략적인 처리 속도 가정
      # GPU/모델에 따라 수정 가능
  
      FAST_FPS = 15
      PRECISE_FPS = 5
  
      fast_estimated_time = fast_analysis_frames / FAST_FPS
  
      precise_estimated_time = (
          precise_analysis_frames / PRECISE_FPS
      )
  
      # -----------------------------
      # 분석 방식 선택 UI
      # -----------------------------
      st.subheader("분석 방식 선택")
  
      col1, col2 = st.columns(2)
  
      # -----------------------------
      # 빠른 분석
      # -----------------------------
      with col1:
  
          st.info(
              f"""
              빠른 분석
              
              • 1초당 1프레임 분석
              • 긴 영상 빠른 확인용
              • 예상 시간: {fast_estimated_time:.1f}초
              """
          )
  
          if st.button(
              "빠른 분석 시작",
              use_container_width=True
          ):
  
              st.success("빠른 분석 시작!")
              st.session_state.analysis_type = "fast"
  
              # 결과 페이지로 이동
              go_to("analysis")
  
      # -----------------------------
      # 정밀 분석
      # -----------------------------
      with col2:
  
          st.warning(
              f"""
              정밀 분석
              
              • 모든 프레임 분석
              • 가장 정확한 결과
              • 결과 mp4 생성
              • 예상 시간: {precise_estimated_time:.1f}초
              """
          )
  
          if st.button(
              "정밀 분석 시작",
              use_container_width=True
          ):
  
              st.success("정밀 분석 시작!")
              st.session_state.analysis_type = "precise"
  
              # 결과 페이지로 이동
              go_to("analysis")

def page_analysis():
  st.title("📊 분석 중")
  
  uploaded_file = st.session_state.uploaded_file
  conf_threshold = st.session_state.conf_threshold
  
  file_type = uploaded_file.type
  
  # 이미지인 경우
  if "image" in file_type:
  
      process.process_image(uploaded_file, model, conf_threshold)
  
  # 동영상인 경우
  elif "video" in file_type:
  
      if st.session_state.analysis_type == "fast":
          
          video_path = st.session_state.video_path
          process.process_fast_video(video_path, model, conf_threshold)
  
                  
      elif st.session_state.analysis_type == "precise":
  
          video_path = st.session_state.video_path
          process.process_precise_video(video_path, model, conf_threshold)

  go_to("result")
  

def page_result():

    uploaded_file = st.session_state.uploaded_file
    file_type = uploaded_file.type
    
    if "image" in file_type:
        result_list = st.session_state.result_list
        result = result_list[0]
        
        utility.render_detection_result(result["annotated_frame"], result["class_id"], result["conf"], result["detection"])

        if result["detection"]:
            utility.show_disease_info(result["class_id"])
    
    elif "video" in file_type:
        if st.session_state.analysis_type == "fast":
            result_list = st.session_state.result_list
            
            for result in result_list:
                utility.render_detection_result(result["annotated_frame"], result["class_id"], result["conf"], result["detection"])
            
            # -----------------------------------
            # 결과 출력
            # -----------------------------------
            
            st.header("📊 병해충 탐지 결과")
            
            st.info(
            f"현재 신뢰도 임계값 (Confidence Threshold): {st.session_state.conf_threshold}"
            )
            
            if st.session_state.detection_frame_count == 0:
            
              st.success("✅ 병해충이 탐지되지 않았습니다.")
            
            else:
              for class_id in st.session_state.detected_classes:
                  utility.show_disease_info(class_id)
                  
        elif st.session_state.analysis_type == "precise":
    
          # -----------------------------
          # H.264 변환
          # -----------------------------
          final_output = tempfile.NamedTemporaryFile(
              delete=False,
              suffix=".mp4"
          ).name
          
          command = [
              "ffmpeg",
              "-y",
              "-i",
              st.session_state.temp_output,
              "-vcodec",
              "libx264",
              "-acodec",
              "aac",
              final_output
          ]
        
          try:
              subprocess.run(
                  command,
                  check=True
              )
          
          except Exception as e:
          
              st.error(f"영상 변환 실패: {e}")
          
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
          f"현재 신뢰도 임계값 (Confidence Threshold): {st.session_state.conf_threshold}"
          )
          
          if st.session_state.detection_frame_count == 0:
          
              st.success("✅ 병해충이 탐지되지 않았습니다.")
          
          else:
              for class_id in st.session_state.detected_classes:
                  utility.show_disease_info(class_id)
        
    if st.button("🔙 처음으로"):
    
      st.session_state.uploaded_file = None
      
      go_to("home")
  
      


def go_to(page):

    st.session_state.page = page

    st.rerun()
