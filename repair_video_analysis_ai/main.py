
from src.preprocessing.video_processor import VideoProcessor

def main():
    """Example usage of the video preprocessing module"""
    
    # Initialize processor
    processor = VideoProcessor(output_base_dir="data/processed")
    
    # Example: Process a single video
    video_path =r"C:\Users\Lenovo\Desktop\video-analysis\repair_video_analysis_ai\data\raw\260119-000023_service_video_1768803391562.mp4"  # Replace with your video path
    
    try:
        results = processor.process(
            video_path=video_path,
            frame_interval=30,      # Extract 1 frame per second (for 30fps video)
            extract_audio=True,     # Extract audio
            jpeg_quality=95         # High quality frames
        )
        
        # Access results
        print(f"\n Processing Results:")
        print(f"  Video ID: {results['video_id']}")
        print(f"  Frames: {results['frame_count']}")
        print(f"  Audio: {results['audio_path']}")
        print(f"  Status: {results['status']}")
        
        if results['errors']:
            print(f"\nErrors encountered:")
            for error in results['errors']:
                print(f"  - {error}")
        
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()