import cv2
from pathlib import Path
from typing import List
from src.preprocessing.validators import PreprocessingValidator


class FrameExtractor:
    """Handles extraction of frames from video files"""
    
    def __init__(self, logger):
        """
        Initialize frame extractor
        
        Args:
            logger: Logger instance for logging operations
        """
        self.logger = logger
    
    def extract(self, 
                video_path: str,
                output_dir: Path,
                frame_interval: int = 30,
                quality: int = 95) -> List[str]:
        """
        Extract frames from video at specified intervals
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save extracted frames
            frame_interval: Extract 1 frame every N frames
            quality: JPEG compression quality (1-100)
            
        Returns:
            List of paths to extracted frame files
            
        Raises:
            ValueError: If video cannot be opened or parameters are invalid
        """
        # Validate inputs
        valid, error = PreprocessingValidator.validate_frame_interval(frame_interval)
        if not valid:
            raise ValueError(error)
        
        valid, error = PreprocessingValidator.validate_jpeg_quality(quality)
        if not valid:
            raise ValueError(error)
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        frame_paths = []
        frame_count = 0
        saved_count = 0
        
        self.logger.info(f"Starting frame extraction (interval: {frame_interval})")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Save frame at specified intervals
                if frame_count % frame_interval == 0:
                    frame_filename = f"frame_{saved_count:04d}.jpg"
                    frame_path = output_dir / frame_filename
                    
                    # Save with specified quality
                    success = cv2.imwrite(
                        str(frame_path), 
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, quality]
                    )
                    
                    if success:
                        frame_paths.append(str(frame_path))
                        saved_count += 1
                    else:
                        self.logger.warning(f"Failed to save frame {saved_count}")
                
                frame_count += 1
            
            self.logger.info(
                f"Extracted {saved_count} frames from {frame_count} total frames"
            )
            
        finally:
            cap.release()
        
        return frame_paths