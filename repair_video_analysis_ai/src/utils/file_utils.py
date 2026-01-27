
import os
import hashlib
from pathlib import Path
from typing import Optional
from datetime import datetime


class FileUtils:
    """Utility functions for file operations and validation"""
    
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    MAX_FILE_SIZE_MB = 500  # Maximum video file size in MB
    
    @staticmethod
    def validate_video_file(file_path: str) -> tuple[bool, Optional[str]]:
        """
        Validate if the video file exists and is valid
        
        Args:
            file_path: Path to video file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            return False, f"File does not exist: {file_path}"
        
        # Check if it's a file (not a directory)
        if not path.is_file():
            return False, f"Path is not a file: {file_path}"
        
        # Check file extension
        if path.suffix.lower() not in FileUtils.SUPPORTED_VIDEO_FORMATS:
            return False, (f"Unsupported format: {path.suffix}. "
                          f"Supported: {', '.join(FileUtils.SUPPORTED_VIDEO_FORMATS)}")
        
        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > FileUtils.MAX_FILE_SIZE_MB:
            return False, (f"File too large: {file_size_mb:.2f}MB. "
                          f"Maximum allowed: {FileUtils.MAX_FILE_SIZE_MB}MB")
        
        # Check if file is readable
        if not os.access(file_path, os.R_OK):
            return False, f"File is not readable: {file_path}"
        
        return True, None
    
    @staticmethod
    def generate_unique_id(file_path: str, prefix: str = "SR") -> str:
        """
        Generate unique ID for a file based on content hash and timestamp
        
        Args:
            file_path: Path to file
            prefix: Prefix for the ID (default: SR for Service Request)
            
        Returns:
            Unique identifier string
        """
        # Read first 1MB for hash (faster than reading entire file)
        with open(file_path, 'rb') as f:
            chunk = f.read(1024 * 1024)
            file_hash = hashlib.md5(chunk).hexdigest()[:8]
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{prefix}_{timestamp}_{file_hash}"
    
    @staticmethod
    def ensure_directory(directory: str) -> Path:
        """
        Create directory if it doesn't exist
        
        Args:
            directory: Directory path
            
        Returns:
            Path object of the directory
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        return path
