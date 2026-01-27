"""
YOLO-based object detection for spare parts identification
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
from ultralytics import YOLO


class YOLODetector:
    """YOLO-based object detector for identifying spare parts"""
    
    def __init__(self, 
                 model_path: str = None,
                 confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 logger=None):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to custom YOLO weights (.pt file)
                       If None, uses YOLOv8n pretrained on COCO (demo mode)
            confidence_threshold: Minimum confidence for detections (0-1)
            iou_threshold: IoU threshold for NMS (0-1)
            logger: Logger instance
        """
        self.logger = logger
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Load YOLO model
        self._load_model(model_path)
        
        if self.logger:
            self.logger.info(f"YOLO detector initialized (confidence={confidence_threshold})")
    
    def _load_model(self, model_path: str = None):
        """Load YOLO model weights"""
        try:
            if model_path and Path(model_path).exists():
                # Load custom trained model
                self.model = YOLO(model_path)
                if self.logger:
                    self.logger.info(f"Loaded custom YOLO model: {model_path}")
            else:
                # Load pretrained YOLOv8 nano (demo mode)
                self.model = YOLO('yolov8n.pt')
                if self.logger:
                    self.logger.info("Loaded YOLOv8n pretrained (COCO dataset)")
                    self.logger.warning("[WARNING] Using demo model - train on spare parts for accuracy")
        
        except Exception as e:
            error_msg = f"Failed to load YOLO model: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def detect(self, image_path: str) -> List[Dict]:
        """
        Detect objects in a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of detections with class_name, confidence, bbox
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Parse results
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            result = results[0]
            img_height, img_width = result.orig_shape
            
            for box in result.boxes:
                bbox = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                
                detection = {
                    'class_name': class_name,
                    'confidence': round(confidence, 3),
                    'bbox': bbox.tolist()
                }
                detections.append(detection)
        
        return detections
    
    def detect_batch(self, image_paths: List[str]) -> Dict[str, List[Dict]]:
        """Detect objects in multiple images"""
        results = {}
        for img_path in image_paths:
            try:
                detections = self.detect(img_path)
                results[img_path] = detections
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Detection failed for {img_path}: {e}")
                results[img_path] = []
        return results