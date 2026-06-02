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
class AnalysisResult:
    result_list: list[DetectionResult] = field(default_factory=list)
    detected_classes: set[int] = field(default_factory=set)
    detection_frame_count: int = 0
    conf_threshold: float = 0.3
    temp_output : str = ''
    final_output: str | None = None
