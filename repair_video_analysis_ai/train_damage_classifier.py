"""
Damage Classifier Training Script
Trains a ResNet18 model to classify spare parts damage into three categories:
- normal: No visible damage
- corrosion: Rust, oxidation, metal degradation
- leakage: Water stains, refrigerant leaks, wet areas
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from pathlib import Path
import json
import time
from datetime import datetime


class DamageClassifierTrainer:
    """
    Handles the complete training pipeline for damage classification.
    Uses transfer learning with a pretrained ResNet18 model.
    """
    
    def __init__(self, data_dir, output_dir="models/damage_classifier", num_classes=3):
        """
        Initialize the trainer with dataset and output directories.
        
        Args:
            data_dir: Path to dataset with train/valid/test splits
            output_dir: Where to save the trained model
            num_classes: Number of damage classes (default: 3)
        """
        # Convert to absolute paths
        self.data_dir = Path(data_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify dataset exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")
        
        self.num_classes = num_classes
        self.class_names = ['corrosion', 'leakage', 'normal']
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on: {self.device}")
        
        # Define image transformations
        self._setup_transforms()
        
        # Load datasets
        self._load_datasets()
        
        # Initialize model
        self._create_model()
    
    def _setup_transforms(self):
        """
        Define image preprocessing and augmentation transforms.
        Training uses augmentation, validation/test use only normalization.
        """
        # Training transforms with data augmentation
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Validation and test transforms without augmentation
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _load_datasets(self):
        """Load train, validation, and test datasets from directory structure."""
        print("\nLoading datasets...")
        
        # Load training data
        train_dir = self.data_dir / 'train'
        if not train_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        
        print(f"Train directory: {train_dir}")
        
        self.train_dataset = datasets.ImageFolder(train_dir, transform=self.train_transform)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Load validation data
        val_dir = self.data_dir / 'valid'
        if not val_dir.exists():
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")
        
        print(f"Valid directory: {val_dir}")
        
        self.val_dataset = datasets.ImageFolder(val_dir, transform=self.val_transform)
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Load test data
        test_dir = self.data_dir / 'test'
        if test_dir.exists():
            print(f"Test directory: {test_dir}")
            self.test_dataset = datasets.ImageFolder(test_dir, transform=self.val_transform)
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=16,
                shuffle=False,
                num_workers=2
            )
        else:
            self.test_loader = None
        
        # Print dataset statistics
        print(f"\nTraining images: {len(self.train_dataset)}")
        print(f"Validation images: {len(self.val_dataset)}")
        if self.test_loader:
            print(f"Test images: {len(self.test_dataset)}")
        
        # Print class distribution
        print("\nClass distribution in training set:")
        for class_name, class_idx in self.train_dataset.class_to_idx.items():
            count = sum(1 for _, label in self.train_dataset.samples if label == class_idx)
            print(f"  {class_name}: {count} images")
    
    def _create_model(self):
        """
        Create ResNet18 model with pretrained ImageNet weights.
        Replace final layer to match number of damage classes.
        """
        print("\nInitializing ResNet18 model...")
        
        # Load pretrained ResNet18 with updated syntax
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Replace final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=3, 
            factor=0.5
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def _train_epoch(self):
        """
        Train the model for one epoch.
        
        Returns:
            Average loss and accuracy for the epoch
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(self.train_dataset)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def _validate(self):
        """
        Validate the model on validation set.
        
        Returns:
            Average loss and accuracy for validation set
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Track statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_dataset)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs=25, early_stopping_patience=5):
        """
        Main training loop with early stopping.
        
        Args:
            num_epochs: Maximum number of epochs to train
            early_stopping_patience: Stop if no improvement for N epochs
            
        Returns:
            Path to best model and training history
        """
        print("\n" + "="*70)
        print(f"Starting Training - {num_epochs} epochs maximum")
        print("="*70 + "\n")
        
        best_val_acc = 0.0
        best_model_path = None
        patience_counter = 0
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self._train_epoch()
            
            # Validate
            val_loss, val_acc = self._validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print progress
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                best_model_path = self.output_dir / "damage_classifier_best.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'class_names': self.class_names,
                }, best_model_path)
                
                print(f"  Best model saved (Val Acc: {val_acc:.4f})")
            else:
                patience_counter += 1
                
            print()
            
            # Early stopping check
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                print(f"No improvement for {early_stopping_patience} consecutive epochs")
                break
        
        total_time = time.time() - start_time
        
        print("="*70)
        print(f"Training Complete in {total_time/60:.1f} minutes")
        print(f"Best Validation Accuracy: {best_val_acc:.4f}")
        print("="*70 + "\n")
        
        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved to: {history_path}")
        
        return best_model_path, history
    
    def test(self, model_path=None):
        """
        Evaluate model on test set.
        
        Args:
            model_path: Path to model weights (if None, uses current model)
            
        Returns:
            Test accuracy
        """
        if self.test_loader is None:
            print("No test dataset available")
            return None
        
        # Load best model if path provided
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from: {model_path}")
        
        self.model.eval()
        correct = 0
        total = 0
        
        # Per-class accuracy tracking
        class_correct = {i: 0 for i in range(self.num_classes)}
        class_total = {i: 0 for i in range(self.num_classes)}
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Track per-class accuracy
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1
        
        # Overall accuracy
        test_acc = correct / total
        
        print("\n" + "="*70)
        print("Test Set Results")
        print("="*70)
        print(f"Overall Accuracy: {test_acc:.4f} ({correct}/{total})")
        print("\nPer-Class Accuracy:")
        for i, class_name in enumerate(self.class_names):
            if class_total[i] > 0:
                acc = class_correct[i] / class_total[i]
                print(f"  {class_name}: {acc:.4f} ({class_correct[i]}/{class_total[i]})")
        print("="*70 + "\n")
        
        return test_acc


def main():
    """Main training function."""
    
    # Get script directory and set paths relative to it
    script_dir = Path(__file__).parent.resolve()
    
    # Configuration
    DATA_DIR = script_dir / "dataset" / "damage_classification"
    OUTPUT_DIR = script_dir / "models" / "damage_classifier"
    NUM_EPOCHS = 25
    EARLY_STOPPING_PATIENCE = 5
    
    print("Damage Classification Training")
    print("="*70)
    print(f"Script directory: {script_dir}")
    print(f"Dataset: {DATA_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Check if dataset exists
    if not DATA_DIR.exists():
        print(f"\nERROR: Dataset directory not found!")
        print(f"Expected location: {DATA_DIR}")
        print("\nPlease ensure your dataset is in the correct location:")
        print("  dataset/damage_classification/train/")
        print("  dataset/damage_classification/valid/")
        print("  dataset/damage_classification/test/")
        return
    
    print("="*70)
    
    # Initialize trainer
    try:
        trainer = DamageClassifierTrainer(
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            num_classes=3
        )
    except Exception as e:
        print(f"\nERROR: Failed to initialize trainer: {e}")
        return
    
    # Train model
    best_model_path, history = trainer.train(
        num_epochs=NUM_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE
    )
    
    # Test model
    if trainer.test_loader:
        trainer.test(model_path=best_model_path)
    
    print(f"\nTraining complete!")
    print(f"Best model saved to: {best_model_path}")
    print(f"Use this model path in your vision pipeline.")


if __name__ == "__main__":
    main()