"""
Vision Pipeline - Integrates YOLO detection and damage classification
Analyzes video frames to detect spare parts and classify damage
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from .yolo_detector import YOLODetector
from .damage_classifier import DamageClassifier


class VisionPipeline:
    """
    Complete vision analysis pipeline combining:
    1. YOLO object detection - identifies spare parts
    2. Damage classification - assesses damage type
    """
    
    def __init__(self,
                 yolo_model_path: Optional[str] = None,
                 damage_model_path: Optional[str] = None,
                 output_dir: str = "data/processed/vision",
                 logger=None):
        """
        Initialize vision pipeline with both detection and classification models
        
        Args:
            yolo_model_path: Path to trained YOLO model (.pt file)
            damage_model_path: Path to trained damage classifier (.pt file)
            output_dir: Directory to save analysis results
            logger: Logger instance
        """
        self.logger = logger
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize YOLO detector
        self.yolo_detector = YOLODetector(
            model_path=yolo_model_path,
            logger=logger
        )
        
        # Initialize damage classifier if model path provided
        self.damage_classifier = None
        if damage_model_path:
            try:
                self.damage_classifier = DamageClassifier(
                    model_path=damage_model_path,
                    logger=logger
                )
                if self.logger:
                    self.logger.info("Damage classifier loaded successfully")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Damage classifier not loaded: {e}")
                    self.logger.warning("Running in YOLO-only mode")
        else:
            if self.logger:
                self.logger.info("No damage model provided - YOLO detection only")
    
    def analyze_frames(self, 
                      frame_paths: List[str], 
                      video_id: str) -> Dict:
        """
        Analyze all frames with YOLO detection and optional damage classification
        
        Args:
            frame_paths: List of frame image paths
            video_id: Unique identifier for this video
            
        Returns:
            Complete analysis results with detections and classifications
        """
        if self.logger:
            self.logger.info(f"Starting vision analysis for video: {video_id}")
            self.logger.info(f"Total frames to process: {len(frame_paths)}")
        
        results = {
            'video_id': video_id,
            'total_frames': len(frame_paths),
            'yolo_detections': {},
            'damage_classifications': {},
            'summary': {
                'total_objects_detected': 0,
                'frames_with_detections': 0,
                'detections_by_class': {},
                'damage_summary': {}
            }
        }
        
        # Step 1: Run YOLO detection on all frames
        if self.logger:
            self.logger.info("Running YOLO object detection...")
        
        for frame_path in frame_paths:
            detections = self.yolo_detector.detect(frame_path)
            results['yolo_detections'][frame_path] = detections
            
            # Update summary statistics
            if detections:
                results['summary']['frames_with_detections'] += 1
                results['summary']['total_objects_detected'] += len(detections)
                
                for det in detections:
                    class_name = det['class_name']
                    if class_name not in results['summary']['detections_by_class']:
                        results['summary']['detections_by_class'][class_name] = 0
                    results['summary']['detections_by_class'][class_name] += 1
        
        if self.logger:
            self.logger.info(f"YOLO detection complete: {results['summary']['total_objects_detected']} objects found")
        
        # Step 2: Run damage classification if available
        if self.damage_classifier:
            if self.logger:
                self.logger.info("Running damage classification...")
            
            self._classify_damage(frame_paths, results)
            
            if self.logger:
                self.logger.info("Damage classification complete")
        
        # Save results to JSON file
        output_path = self.output_dir / f"{video_id}_vision.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        if self.logger:
            self.logger.info(f"Results saved to: {output_path}")
        
        return results
    
    def _classify_damage(self, frame_paths: List[str], results: Dict):
        """
        Run damage classification on all frames
        
        Args:
            frame_paths: List of frame paths
            results: Results dictionary to update with classifications
        """
        damage_counts = {'normal': 0, 'corrosion': 0, 'leakage': 0}
        
        for frame_path in frame_paths:
            try:
                # Classify damage in full frame
                predictions = self.damage_classifier.classify(frame_path, top_k=2)
                results['damage_classifications'][frame_path] = predictions
                
                # Update damage counts using top prediction
                if predictions and len(predictions) > 0:
                    top_class = predictions[0]['class']
                    damage_counts[top_class] += 1
                    
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Classification failed for {frame_path}: {e}")
                results['damage_classifications'][frame_path] = []
        
        # Calculate damage summary
        total = len(frame_paths)
        results['summary']['damage_summary'] = {
            'total_frames': total,
            'damage_counts': damage_counts,
            'damage_percentages': {
                cls: round((count / total) * 100, 2) if total > 0 else 0
                for cls, count in damage_counts.items()
            }
        }
    
    def analyze_single_frame(self, frame_path: str) -> Dict:
        """
        Analyze a single frame with both detection and classification
        
        Args:
            frame_path: Path to frame image
            
        Returns:
            Analysis results for this frame
        """
        result = {
            'frame_path': frame_path,
            'yolo_detections': [],
            'damage_classification': []
        }
        
        # YOLO detection
        result['yolo_detections'] = self.yolo_detector.detect(frame_path)
        
        # Damage classification if available
        if self.damage_classifier:
            try:
                result['damage_classification'] = self.damage_classifier.classify(
                    frame_path, 
                    top_k=2
                )
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Classification failed: {e}")
        
        return result