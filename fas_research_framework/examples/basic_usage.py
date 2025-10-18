"""
Basic usage example for FAS-Research-Framework

This example demonstrates how to use the framework for basic face anti-spoofing tasks.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# Import FAS-Research-Framework modules
import fas_research_framework as frf
from fas_research_framework.core.models import mobilenetv4_large
from fas_research_framework.core.datasets import CelebASpoofDataset
from fas_research_framework.core.losses import AMSoftmaxLoss
from fas_research_framework.core.training import Trainer


def main():
    """Main function demonstrating basic usage."""
    print("FAS-Research-Framework - Basic Usage Example")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load a model
    print("\n1. Loading MobileNetV4-Large model...")
    model = mobilenetv4_large(pretrained=True, num_classes=2)
    model = model.to(device)
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 2. Define data transformations
    print("\n2. Setting up data transformations...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Create dataset (example with dummy data)
    print("\n3. Creating dataset...")
    # Note: In practice, you would provide the actual dataset path
    # dataset = CelebASpoofDataset(root="path/to/celeba_spoof", transform=transform)
    print("Dataset created (using dummy data for this example)")
    
    # 4. Define loss function
    print("\n4. Setting up loss function...")
    criterion = AMSoftmaxLoss(num_classes=2, embedding_size=1280)
    print("AM-Softmax loss function configured")
    
    # 5. Create optimizer
    print("\n5. Setting up optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("Adam optimizer configured")
    
    # 6. Model summary
    print("\n6. Model Summary:")
    print(f"  - Model: MobileNetV4-Large")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Device: {device}")
    print(f"  - Loss: AM-Softmax")
    print(f"  - Optimizer: Adam")
    
    # 7. Example forward pass
    print("\n7. Testing model forward pass...")
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        output = model(dummy_input)
        print(f"  - Input shape: {dummy_input.shape}")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Output values: {output.cpu().numpy()}")
    
    print("\n" + "=" * 50)
    print("Basic usage example completed successfully!")
    print("For more advanced examples, check the examples/ directory.")


if __name__ == "__main__":
    main()
