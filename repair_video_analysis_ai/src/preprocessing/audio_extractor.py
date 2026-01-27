import subprocess
import shutil
from pathlib import Path




class AudioExtractor:
    """Handles extraction of audio from video files"""
    
    def __init__(self, logger):
        """
        Initialize audio extractor
        
        Args:
            logger: Logger instance for logging operations
        """
        self.logger = logger
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        if not shutil.which('ffmpeg'):
            raise RuntimeError(
                "ffmpeg not found in system PATH. Please install ffmpeg."
            )


    
    def extract(self, 
                video_path: str,
                output_path: Path,
                sample_rate: int = 16000,
                channels: int = 1) -> str:
        """
        Extract audio from video and convert to WAV format
        
        Args:
            video_path: Path to input video file
            output_path: Path for output WAV file
            sample_rate: Audio sample rate in Hz (16000 optimal for Whisper)
            channels: Number of audio channels (1=mono, 2=stereo)
            
        Returns:
            Path to extracted WAV file
            
        Raises:
            RuntimeError: If audio extraction fails
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Extracting audio (sample_rate: {sample_rate}Hz, channels: {channels})")
        
        # Build ffmpeg command
        command = [
            'ffmpeg',
            '-i', video_path,           # Input file
            '-vn',                       # No video
            '-acodec', 'pcm_s16le',     # PCM 16-bit encoding
            '-ar', str(sample_rate),    # Sample rate
            '-ac', str(channels),       # Audio channels
            '-y',                        # Overwrite output file
            str(output_path)
        ]
        
        try:
            # Run ffmpeg with output suppression
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Verify output file was created
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError("Audio file was not created or is empty")
            
            self.logger.info(f"Audio extracted successfully: {output_path}")
            return str(output_path)
            
        except subprocess.CalledProcessError as e:
            error_msg = f"ffmpeg failed: {e.stderr}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)