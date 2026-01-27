
from pathlib import Path
from typing import Dict, Optional
from src.utils.logger import setup_logger
from src.preprocessing.frame_extractor import FrameExtractor
from src.preprocessing.audio_extractor import AudioExtractor
from src.preprocessing.validators import PreprocessingValidator
from src.utils.file_utils import FileUtils


class VideoProcessor:
    """
    Main orchestrator for video preprocessing pipeline
    Coordinates frame extraction, audio extraction, and metadata generation
    """
    
    def __init__(self, output_base_dir: str = "data/processed"):
        """
        Initialize video processor
        
        Args:
            output_base_dir: Base directory for all processed outputs
        """
        self.output_base_dir = Path(output_base_dir)
        self.logger = setup_logger(__name__)
        
        # Initialize sub-components
        self.frame_extractor = FrameExtractor(self.logger)
        self.audio_extractor = AudioExtractor(self.logger)
        
        # Setup directory structure
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary output directories"""
        directories = ['frames', 'audio']
        for dir_name in directories:
            FileUtils.ensure_directory(self.output_base_dir / dir_name)
        
        self.logger.debug(f"Output directories created at: {self.output_base_dir}")
    
    def process(self,
                video_path: str,
                frame_interval: int = 30,
                extract_audio: bool = True,
                jpeg_quality: int = 95) -> Dict:
        """
        Complete preprocessing pipeline for a single video
        
        Args:
            video_path: Path to input video file
            frame_interval: Extract 1 frame every N frames (default: 30)
            extract_audio: Whether to extract audio (default: True)
            jpeg_quality: JPEG compression quality 1-100 (default: 95)
            
        Returns:
            Dictionary containing:
                - video_id: Unique identifier for this video
                - frame_paths: List of extracted frame file paths
                - audio_path: Path to extracted audio file (or None)
                - frame_count: Number of frames extracted
                - status: 'success' or 'partial' or 'failed'
                - errors: List of any errors encountered
                
        Raises:
            ValueError: If video file is invalid
            RuntimeError: If critical preprocessing step fails
        """
        self.logger.info("=" * 70)
        self.logger.info(f"Starting video preprocessing: {video_path}")
        self.logger.info("=" * 70)
        
        # Step 0: Validate input file
        self.logger.info("Step 0: Validating input file...")
        valid, error = FileUtils.validate_video_file(video_path)
        if not valid:
            self.logger.error(f"Validation failed: {error}")
            raise ValueError(error)
        
        valid, error = PreprocessingValidator.can_open_video(video_path)
        if not valid:
            self.logger.error(f"Video compatibility check failed: {error}")
            raise ValueError(error)
        
        self.logger.info("Input file validated successfully")
        
        # Generate unique video ID
        video_id = FileUtils.generate_unique_id(video_path)
        self.logger.info(f"Generated Video ID: {video_id}")
        
        # Initialize result tracking
        results = {
            'video_id': video_id,
            'frame_paths': [],
            'audio_path': None,
            'frame_count': 0,
            'status': 'success',
            'errors': []
        }
        
        # Step 1: Extract frames
        self.logger.info("\nStep 1: Extracting frames...")
        try:
            frame_dir = self.output_base_dir / 'frames' / video_id
            frame_paths = self.frame_extractor.extract(
                video_path=video_path,
                output_dir=frame_dir,
                frame_interval=frame_interval,
                quality=jpeg_quality
            )
            results['frame_paths'] = frame_paths
            results['frame_count'] = len(frame_paths)
            self.logger.info(f"Extracted {len(frame_paths)} frames")
            
        except Exception as e:
            error_msg = f"Frame extraction failed: {str(e)}"
            self.logger.error(error_msg)
            results['errors'].append(error_msg)
            results['status'] = 'partial'
        
        # Step 2: Extract audio
        if extract_audio:
            self.logger.info("\nStep 2: Extracting audio...")
            try:
                audio_path = self.output_base_dir / 'audio' / f"{video_id}.wav"
                audio_file = self.audio_extractor.extract(
                    video_path=video_path,
                    output_path=audio_path
                )
                results['audio_path'] = audio_file
                self.logger.info(f"Audio extracted successfully")
                
            except Exception as e:
                error_msg = f"Audio extraction failed: {str(e)}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
                results['status'] = 'partial'
        else:
            self.logger.info("\nStep 2: Audio extraction skipped (disabled)")
        
        # Final status
        if results['errors']:
            if not results['frame_paths'] and not results['audio_path']:
                results['status'] = 'failed'
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info(f"Preprocessing complete - Status: {results['status'].upper()}")
        self.logger.info(f"  Frames: {results['frame_count']}")
        self.logger.info(f"  Audio: {'[OK]' if results['audio_path'] else '[FAIL]'}")
        if results['errors']:
            self.logger.warning(f"  Errors: {len(results['errors'])}")
        self.logger.info("=" * 70)
        
        return results
