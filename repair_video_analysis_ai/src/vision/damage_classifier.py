"""
Damage Classifier Module
Classifies spare parts damage into three categories: normal, corrosion, leakage
Uses ResNet18 CNN trained on damage classification dataset
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional


class DamageClassifier:
    """
    CNN-based classifier for damage detection in spare parts.
    Identifies three damage types: normal, corrosion, leakage
    """
    
    # Damage categories - must match training order
    DAMAGE_CLASSES = ['corrosion', 'leakage', 'normal']
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 logger=None):
        """
        Initialize damage classifier with trained model
        
        Args:
            model_path: Path to trained model checkpoint (.pt file)
            device: 'cuda', 'cpu', or None for auto-detect
            logger: Logger instance for tracking
        """
        self.logger = logger
        
        # Determine device for inference
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load the trained model
        self._load_model(model_path)
        
        # Define image preprocessing pipeline
        # Same transforms used during training validation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        if self.logger:
            self.logger.info(f"Damage classifier initialized on {self.device}")
    
    def _load_model(self, model_path: Optional[str] = None):
        """
        Load trained ResNet18 model from checkpoint
        
        Args:
            model_path: Path to model checkpoint saved during training
        """
        # Create ResNet18 architecture
        self.model = models.resnet18(pretrained=False)
        
        # Replace final fully connected layer for 3 classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, len(self.DAMAGE_CLASSES))
        
        # Load trained weights if provided
        if model_path and Path(model_path).exists():
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Handle both direct state dict and checkpoint dict
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                if self.logger:
                    self.logger.info(f"Loaded trained damage classifier from: {model_path}")
                    
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to load model weights: {e}")
                raise RuntimeError(f"Could not load model from {model_path}: {e}")
        else:
            error_msg = f"Model file not found: {model_path}"
            if self.logger:
                self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
    
    def classify(self, image_path: str, top_k: int = 2) -> List[Dict]:
        """
        Classify damage type in a single image
        
        Args:
            image_path: Path to image file (frame or cropped region)
            top_k: Number of top predictions to return (default: 2)
            
        Returns:
            List of predictions sorted by confidence
            Each prediction contains: {'class': str, 'confidence': float}
            
        Example:
            [
                {'class': 'corrosion', 'confidence': 0.892},
                {'class': 'leakage', 'confidence': 0.076}
            ]
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Get top K predictions
            top_probs, top_indices = torch.topk(
                probabilities, 
                min(top_k, len(self.DAMAGE_CLASSES))
            )
            
            # Format results
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                predictions.append({
                    'class': self.DAMAGE_CLASSES[idx.item()],
                    'confidence': round(prob.item(), 4)
                })
            
            return predictions
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Classification failed for {image_path}: {e}")
            raise RuntimeError(f"Classification error: {e}")
    
    def classify_batch(self, image_paths: List[str], top_k: int = 2) -> Dict[str, List[Dict]]:
        """
        Classify damage in multiple images
        
        Args:
            image_paths: List of image file paths
            top_k: Number of top predictions per image
            
        Returns:
            Dictionary mapping image paths to their predictions
            
        Example:
            {
                'frame_0000.jpg': [{'class': 'corrosion', 'confidence': 0.892}],
                'frame_0001.jpg': [{'class': 'normal', 'confidence': 0.956}]
            }
        """
        results = {}
        
        for img_path in image_paths:
            try:
                predictions = self.classify(img_path, top_k=top_k)
                results[img_path] = predictions
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to classify {img_path}: {e}")
                results[img_path] = []
        
        return results
    
    def get_damage_summary(self, predictions: List[Dict]) -> Dict:
        """
        Generate summary statistics from predictions
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Summary with counts and percentages by damage type
        """
        summary = {
            'total_predictions': len(predictions),
            'damage_counts': {cls: 0 for cls in self.DAMAGE_CLASSES},
            'damage_percentages': {}
        }
        
        if not predictions:
            return summary
        
        # Count each damage type (using top prediction)
        for pred in predictions:
            if pred and len(pred) > 0:
                top_class = pred[0]['class']
                summary['damage_counts'][top_class] += 1
        
        # Calculate percentages
        total = summary['total_predictions']
        summary['damage_percentages'] = {
            cls: round((count / total) * 100, 2)
            for cls, count in summary['damage_counts'].items()
        }
        
        return summary