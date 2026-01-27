"""
Vision Module Main Script
Runs complete vision analysis: YOLO detection + damage classification
"""

import sys
from pathlib import Path

# Get script directory and add to path
script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir))

from src.utils.logger import setup_logger
from src.vision.vision_pipeline import VisionPipeline


def main():
    """
    Main execution function for vision analysis
    Analyzes preprocessed frames with YOLO + damage classifier
    """
    
    logger = setup_logger(__name__)
    
    logger.info("Vision Module - Complete Analysis")
    logger.info("=" * 70)

    # Manually specify video ID
    VIDEO_ID = "SR_20260123_154910_d96affd0"
    logger.info(f"Using VIDEO_ID: {VIDEO_ID}")

    # Build paths relative to script directory
    FRAMES_DIR = script_dir.parent / "data" / "processed" / "frames" / VIDEO_ID
    YOLO_MODEL = script_dir / "models" / "yolo" / "spare_parts_v1.pt"
    DAMAGE_MODEL = script_dir / "models" / "damage_classifier" / "damage_classifier_best.pt"
    OUTPUT_DIR = script_dir.parent / "data" / "processed" / "vision"
        
    logger.info(f"Script directory: {script_dir}")
    logger.info(f"Looking for frames in: {FRAMES_DIR}")
    logger.info("")
    
    # Verify frames directory exists
    if not FRAMES_DIR.exists():
        logger.error(f"Frames directory not found: {FRAMES_DIR}")
        logger.error("Please run preprocessing (main.py) first!")
        logger.error("")
        logger.error("Expected structure:")
        logger.error(f"  {script_dir / 'data' / 'processed' / 'frames' / VIDEO_ID}")
        return
    
    # Get all frame paths
    frame_paths = sorted([str(p) for p in FRAMES_DIR.glob("*.jpg")])
    
    if not frame_paths:
        logger.error(f"No frames found in {FRAMES_DIR}")
        return
    
    logger.info(f"Found {len(frame_paths)} frames for video {VIDEO_ID}")
    logger.info("")
    
    # Check if models exist
    logger.info("Checking models...")
    if not YOLO_MODEL.exists():
        logger.error(f"YOLO model not found: {YOLO_MODEL}")
        return
    logger.info(f"YOLO model: {YOLO_MODEL}")
    
    if not DAMAGE_MODEL.exists():
        logger.warning(f"Damage model not found: {DAMAGE_MODEL}")
        logger.warning("Running in YOLO-only mode")
        DAMAGE_MODEL = None
    else:
        logger.info(f"Damage model: {DAMAGE_MODEL}")
    
    logger.info("")
    
    # Initialize vision pipeline with both models
    logger.info("Initializing vision pipeline...")
    
    try:
        pipeline = VisionPipeline(
            yolo_model_path=str(YOLO_MODEL),
            damage_model_path=str(DAMAGE_MODEL) if DAMAGE_MODEL else None,
            output_dir=str(OUTPUT_DIR),
            logger=logger
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return
    
    # Run complete vision analysis
    logger.info("Starting vision analysis...")
    logger.info("-" * 70)
    
    try:
        results = pipeline.analyze_frames(frame_paths, VIDEO_ID)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Display results summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("VISION ANALYSIS COMPLETE")
    logger.info("=" * 70)
    
    # YOLO detection summary
    logger.info(f"\nVideo ID: {results['video_id']}")
    logger.info(f"Total Frames Analyzed: {results['total_frames']}")
    logger.info(f"Frames with Objects: {results['summary']['frames_with_detections']}")
    logger.info(f"Total Objects Detected: {results['summary']['total_objects_detected']}")
    
    logger.info("\nDetections by Class:")
    if results['summary']['detections_by_class']:
        for class_name, count in sorted(results['summary']['detections_by_class'].items()):
            logger.info(f"  {class_name}: {count}")
    else:
        logger.info("  No objects detected")
    
    # Damage classification summary
    if results['summary']['damage_summary']:
        logger.info("\nDamage Classification Summary:")
        damage_summary = results['summary']['damage_summary']
        
        logger.info(f"  Total Frames Classified: {damage_summary['total_frames']}")
        logger.info("\n  Damage Distribution:")
        
        for damage_type, count in damage_summary['damage_counts'].items():
            percentage = damage_summary['damage_percentages'][damage_type]
            logger.info(f"    {damage_type.capitalize()}: {count} frames ({percentage}%)")
    
    # Show sample results from first few frames
    logger.info("\n" + "-" * 70)
    logger.info("Sample Results (First 3 Frames):")
    logger.info("-" * 70)
    
    sample_frames = list(results['yolo_detections'].keys())[:3]
    
    for frame_path in sample_frames:
        frame_name = Path(frame_path).name
        logger.info(f"\n{frame_name}:")
        
        # YOLO detections
        yolo_dets = results['yolo_detections'][frame_path]
        if yolo_dets:
            logger.info("  YOLO Detections:")
            for det in yolo_dets[:3]:  # Show max 3 detections
                logger.info(f"    - {det['class_name']} (confidence: {det['confidence']:.3f})")
        else:
            logger.info("  YOLO Detections: None")
        
        # Damage classification
        if frame_path in results['damage_classifications']:
            damage_preds = results['damage_classifications'][frame_path]
            if damage_preds:
                logger.info("  Damage Classification:")
                for pred in damage_preds:
                    logger.info(f"    - {pred['class']} (confidence: {pred['confidence']:.3f})")
    
    logger.info("\n" + "=" * 70)
    logger.info(f"Complete results saved to: {OUTPUT_DIR / f'{VIDEO_ID}_vision.json'}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()