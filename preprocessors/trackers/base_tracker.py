import numpy as np
from abc import ABC, abstractmethod

class BaseTracker(ABC):
    """
    Base class for 2D human bounding box preprocessors.
    All subclasses must implement the `track` method.
    """
    def __init__(self, device='cuda:0'):
        self.device = device

    @abstractmethod
    def track(self, video_path: str) -> np.ndarray:
        """
        Processes a video and returns human bounding boxes.

        Args:
            video_path (str): Path to the input video.

        Returns:
            np.ndarray: Bounding boxes of shape (F, 4) where F is the number of frames.
                        Each row contains [x1, y1, x2, y2].
                        If no detection is found for a frame, the row is [-1.0, -1.0, -1.0, -1.0].
        """
        pass
