from dataclasses import dataclass
import numpy as np

@dataclass 
class DetectionResult: 
  class_id: int | None 
  conf: float | None 
  detection: bool 
  annotated_frame: np.ndarray | None

@dataclass
class VideoInfo:

    fps: float
    width: int
    height: int
    total_frames: int
    duration: float

@dataclass
class ImageAnalysisResult:
    result_list: list[DetectionResult]

@dataclass
class FastVideoAnalysisResult:

    result_list: list[DetectionResult]

    detection_frame_count: int

    detected_classes: set[int]

    conf_threshold: float

@dataclass
class PreciseVideoAnalysisResult:

    detection_frame_count: int

    detected_classes: set[int]

    conf_threshold: float

    temp_output : str
  
    final_output : str
