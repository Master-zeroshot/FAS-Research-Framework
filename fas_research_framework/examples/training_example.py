"""
Training example for FAS-Research-Framework

This example demonstrates how to train a model using the framework.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import numpy as np

# Import FAS-Research-Framework modules
import fas_research_framework as frf
from fas_research_framework.core.models import mobilenetv4_small
from fas_research_framework.core.losses import AMSoftmaxLoss
from fas_research_framework.core.training import Trainer


def create_dummy_dataset(num_samples=1000, num_classes=2, img_size=224):
    """Create a dummy dataset for demonstration purposes."""
    # Generate random images
    images = torch.randn(num_samples, 3, img_size, img_size)
    
    # Generate random labels
    labels = torch.randint(0, num_classes, (num_samples,))
    
    # Create dataset
    dataset = TensorDataset(images, labels)
    return dataset


def main():
    """Main training function."""
    print("FAS-Research-Framework - Training Example")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Create model
    print("\n1. Creating MobileNetV4-Small model...")
    model = mobilenetv4_small(num_classes=2)
    model = model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 2. Create dummy dataset
    print("\n2. Creating dummy dataset...")
    train_dataset = create_dummy_dataset(num_samples=1000, num_classes=2)
    val_dataset = create_dummy_dataset(num_samples=200, num_classes=2)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # 3. Define loss function
    print("\n3. Setting up loss function...")
    criterion = AMSoftmaxLoss(num_classes=2, embedding_size=1280)
    print("AM-Softmax loss function configured")
    
    # 4. Create optimizer
    print("\n4. Setting up optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("Adam optimizer configured")
    
    # 5. Create trainer
    print("\n5. Creating trainer...")
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir="./checkpoints"
    )
    print("Trainer created")
    
    # 6. Training configuration
    print("\n6. Training configuration:")
    print(f"  - Epochs: 5")
    print(f"  - Batch size: 32")
    print(f"  - Learning rate: 0.001")
    print(f"  - Device: {device}")
    
    # 7. Start training
    print("\n7. Starting training...")
    try:
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=5,
            save_best=True,
            verbose=True
        )
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed with error: {e}")
        print("This is expected in the example since we're using dummy data")
    
    print("\n" + "=" * 50)
    print("Training example completed!")
    print("For real training, replace the dummy dataset with actual data.")


if __name__ == "__main__":
    main()
