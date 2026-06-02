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
