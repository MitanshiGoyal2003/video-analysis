import cv2
from pathlib import Path
from typing import Optional,Tuple


class PreprocessingValidator:
    """Validation logic for preprocessing operations"""
    
    @staticmethod
    def validate_frame_interval(interval: int) -> tuple[bool, Optional[str]]:
        """Validate frame extraction interval"""
        if not isinstance(interval, int):
            return False, "Frame interval must be an integer"
        if interval < 1:
            return False, "Frame interval must be >= 1"
        if interval > 300:  # ~10 seconds at 30fps
            return False, "Frame interval too large (max: 300)"
        return True, None
    
    @staticmethod
    def validate_jpeg_quality(quality: int) -> tuple[bool, Optional[str]]:
        """Validate JPEG quality parameter"""
        if not isinstance(quality, int):
            return False, "Quality must be an integer"
        if not 1 <= quality <= 100:
            return False, "Quality must be between 1 and 100"
        return True, None
    
    @staticmethod
    def can_open_video(video_path: str) -> tuple[bool, Optional[str]]:
        """Check if OpenCV can open the video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, f"Cannot open video with OpenCV: {video_path}"
        
        # Try to read first frame
        ret, _ = cap.read()
        cap.release()
        
        if not ret:
            return False, "Video file appears to be corrupted or empty"
        
        return True, None
