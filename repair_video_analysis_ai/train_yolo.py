"""
Custom YOLO Training Script for Spare Parts Detection
Classes: capacitor, coil, motor
"""

from ultralytics import YOLO
import torch
from pathlib import Path


def train_yolo_model():
    """Train YOLOv8 on spare parts dataset"""
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print(f"Training Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*70}\n")
    
    # Load pretrained YOLOv8 nano model
    model = YOLO('yolov8n.pt')
    
    print("Starting training for spare parts detection...")
    print("Classes: capacitor, coil, motor")
    print(f"{'='*70}\n")
    
    # Train the model
    results = model.train(
        # Dataset configuration
        # Point directly to data.yaml inside your dataset folder
        data='dataset/video-analysis-dataset-roboflow.yolov8/data.yaml',
        
        # Training parameters
        epochs=100,              # Training epochs (adjust based on results)
        imgsz=640,              # Image size
        batch=16,               # Batch size (reduce to 8 if GPU memory error)
        
        # Model settings
        patience=50,            # Early stopping patience
        save=True,              # Save checkpoints
        save_period=10,         # Save every N epochs
        
        # Optimization
        optimizer='AdamW',      # Optimizer
        lr0=0.01,              # Initial learning rate
        lrf=0.01,              # Final learning rate
        momentum=0.937,         # Momentum
        weight_decay=0.0005,    # Weight decay
        
        # Data augmentation (important for small datasets)
        hsv_h=0.015,           # Hue augmentation
        hsv_s=0.7,             # Saturation augmentation  
        hsv_v=0.4,             # Value augmentation
        degrees=15,            # Rotation (±15 degrees)
        translate=0.1,         # Translation
        scale=0.5,             # Scale
        flipud=0.0,            # Vertical flip (not needed for spare parts)
        fliplr=0.5,            # Horizontal flip
        mosaic=1.0,            # Mosaic augmentation
        mixup=0.0,             # Mixup augmentation
        copy_paste=0.0,        # Copy-paste augmentation
        
        # Validation
        val=True,              # Validate during training
        plots=True,            # Save training plots
        
        # Output
        project='runs/detect',  # Project directory
        name='spare_parts_v1',   # Experiment name
        exist_ok=False,        # Don't overwrite existing
        
        # Performance
        workers=8,             # Dataloader workers
        device=device,         # Training device
        amp=True,              # Automatic Mixed Precision
        cos_lr=True,           # Cosine learning rate scheduler
        close_mosaic=10,       # Disable mosaic last 10 epochs
        
        # Resume (if training interrupted)
        resume=False,
        
        # Verbose
        verbose=True,
    )
    
    # Training complete
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Best model: {results.save_dir}/weights/best.pt")
    print(f"Last model: {results.save_dir}/weights/last.pt")
    print(f"Results saved to: {results.save_dir}")
    print(f"{'='*70}\n")
    
    # Copy best model to models folder
    import shutil
    best_model_src = Path(results.save_dir) / "weights" / "best.pt"
    models_dir = Path("models/yolo")
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_dst = models_dir / "spare_parts_v1.pt"
    
    if best_model_src.exists():
        shutil.copy(best_model_src, best_model_dst)
        print(f"✅ Model copied to: {best_model_dst}")
        print(f"{'='*70}\n")
    
    return results


def validate_model(model_path='models/yolo/spare_parts_v1.pt'):
    """Validate trained model on test set"""
    
    print(f"\n{'='*70}")
    print("VALIDATING MODEL")
    print(f"{'='*70}\n")
    
    model = YOLO(model_path)
    
    # Run validation
    metrics = model.val(
        data='dataset/video-analysis-dataset-roboflow.yolov8/data.yaml',
        split='test',          # Use test set
        imgsz=640,
        batch=16,
        conf=0.25,             # Confidence threshold
        iou=0.45,              # IoU threshold for NMS
        device='cuda' if torch.cuda.is_available() else 'cpu',
        plots=True,            # Save validation plots
    )
    
    # Print metrics
    print(f"\n{'='*70}")
    print("VALIDATION METRICS")
    print(f"{'='*70}")
    print(f"mAP50 (IoU=0.5):        {metrics.box.map50:.3f}")
    print(f"mAP50-95 (IoU=0.5:0.95): {metrics.box.map:.3f}")
    print(f"Precision:               {metrics.box.mp:.3f}")
    print(f"Recall:                  {metrics.box.mr:.3f}")
    print(f"{'='*70}\n")
    
    # Per-class metrics
    if hasattr(metrics.box, 'maps'):
        print("Per-Class mAP50:")
        class_names = ['capacitor', 'coil', 'motor']
        for i, class_name in enumerate(class_names):
            if i < len(metrics.box.ap50):
                print(f"  {class_name}: {metrics.box.ap50[i]:.3f}")
        print(f"{'='*70}\n")
    
    return metrics


def test_inference(model_path='models/yolo/spare_parts_v1.pt',
                   test_dir='data/processed/frames/SR_20260107_113041_39c80ade'):
    """Test model on actual frames"""
    
    print(f"\n{'='*70}")
    print("TESTING INFERENCE ON FRAMES")
    print(f"{'='*70}\n")
    
    model = YOLO(model_path)
    
    # Get test frames
    test_frames = sorted(Path(test_dir).glob('*.jpg'))[:5]  # Test on first 5 frames
    
    if not test_frames:
        print(f"No frames found in {test_dir}")
        return
    
    print(f"Testing on {len(test_frames)} frames from {test_dir}\n")
    
    # Run inference
    for frame_path in test_frames:
        results = model.predict(
            source=str(frame_path),
            conf=0.25,
            iou=0.45,
            save=True,           # Save annotated images
            save_txt=False,
            verbose=False,
        )
        
        # Print detections
        print(f"Frame: {frame_path.name}")
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = results[0].names[cls]
                print(f"  → {name}: {conf:.3f}")
        else:
            print("  → No detections")
        print()
    
    print(f"Annotated images saved to: runs/detect/predict/")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    
    # STEP 1: Train the model
    print("\n" + "="*70)
    print("STEP 1: TRAINING YOLO MODEL")
    print("="*70)
    train_yolo_model()
    
    # STEP 2: Validate on test set
    print("\n" + "="*70)
    print("STEP 2: VALIDATION")
    print("="*70)
    validate_model()
    
    # STEP 3: Test inference on your actual frames
    print("\n" + "="*70)
    print("STEP 3: TEST INFERENCE")
    print("="*70)
    test_inference()
    
    print("\n" + "="*70)
    print("ALL DONE! ✅")
    print("="*70)
    print("\nNext steps:")
    print("1. Check results in: runs/detect/spare_parts_v1/")
    print("2. View training plots: runs/detect/spare_parts_v1/results.png")
    print("3. Your model is saved in: models/yolo/spare_parts_v1.pt")
    print("4. Update main_vision.py to use: models/yolo/spare_parts_v1.pt")
    print("="*70 + "\n")