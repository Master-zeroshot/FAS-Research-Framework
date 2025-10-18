#!/usr/bin/env python3
"""
Test script for MobileNetV4 integration
"""

import os
import sys

import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import mobilenetv4_large, mobilenetv4_medium, mobilenetv4_small
from utils import read_py_config


def test_mobilenetv4_models():
    """Test MobileNetV4 model creation and forward pass"""

    print("Testing MobileNetV4 models...")

    # Test parameters
    test_params = {
        "width_mult": 1.0,
        "prob_dropout": 0.2,
        "type_dropout": "gaussian",
        "mu": 0.5,
        "sigma": 0.3,
        "embeding_dim": 1024,
        "prob_dropout_linear": 0.3,
        "theta": 0,
        "multi_heads": True,
    }

    # Test input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 224, 224)

    # Test MobileNetV4-Small
    print("\n1. Testing MobileNetV4-Small...")
    try:
        model_small = mobilenetv4_small(**test_params)
        model_small.eval()

        with torch.no_grad():
            features = model_small(input_tensor)
            print(f"   OK MobileNetV4-Small forward pass successful")
            print(f"   OK Output shape: {features.shape}")

            # Test multi-head output
            if test_params["multi_heads"]:
                logits = model_small.make_logits(features, all=True)
                print(f"   OK Multi-head output: {len(logits)} heads")
                for i, head in enumerate(logits):
                    print(f"   OK Head {i} shape: {head.shape}")

    except Exception as e:
        print(f"   X MobileNetV4-Small failed: {e}")
        return False

    # Test MobileNetV4-Medium
    print("\n2. Testing MobileNetV4-Medium...")
    try:
        test_params["embeding_dim"] = 1152
        model_medium = mobilenetv4_medium(**test_params)
        model_medium.eval()

        with torch.no_grad():
            features = model_medium(input_tensor)
            print(f"   OK MobileNetV4-Medium forward pass successful")
            print(f"   OK Output shape: {features.shape}")

    except Exception as e:
        print(f"   X MobileNetV4-Medium failed: {e}")
        return False

    # Test MobileNetV4-Large
    print("\n3. Testing MobileNetV4-Large...")
    try:
        test_params["embeding_dim"] = 1280
        model_large = mobilenetv4_large(**test_params)
        model_large.eval()

        with torch.no_grad():
            features = model_large(input_tensor)
            print(f"   OK MobileNetV4-Large forward pass successful")
            print(f"   OK Output shape: {features.shape}")

    except Exception as e:
        print(f"   X MobileNetV4-Large failed: {e}")
        return False

    print("\nOK All MobileNetV4 models tested successfully!")
    return True


def test_configuration_integration():
    """Test MobileNetV4 configuration integration"""

    print("\nTesting MobileNetV4 configuration integration...")

    # Test if configuration files exist
    config_files = [
        "configs/config_mobilenetv4_small.py",
        "configs/config_mobilenetv4_medium.py",
        "configs/config_mobilenetv4_large.py",
    ]

    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"   OK {config_file} exists")
            try:
                config = read_py_config(config_file)
                print(f"   OK {config_file} loads successfully")
                print(f"   OK Model type: {config.model.model_type}")
                print(f"   OK Model size: {config.model.model_size}")
                print(f"   OK Embedding dim: {config.model.embeding_dim}")
            except Exception as e:
                print(f"   X {config_file} failed to load: {e}")
                return False
        else:
            print(f"   X {config_file} not found")
            return False

    print("\nOK All MobileNetV4 configurations tested successfully!")
    return True


def test_model_building():
    """Test MobileNetV4 integration with model building pipeline"""

    print("\nTesting MobileNetV4 model building pipeline...")

    try:
        from utils import build_model

        # Test configuration for MobileNetV4-Small
        test_config = type(
            "Config",
            (),
            {
                "model": type(
                    "Model",
                    (),
                    {
                        "model_type": "Mobilenet4",
                        "model_size": "small",
                        "width_mult": 1.0,
                        "pretrained": False,
                        "embeding_dim": 1024,
                        "imagenet_weights": None,
                    },
                )(),
                "dropout": type(
                    "Dropout",
                    (),
                    {
                        "prob_dropout": 0.2,
                        "type": "gaussian",
                        "mu": 0.5,
                        "sigma": 0.3,
                        "classifier": 0.3,
                    },
                )(),
                "conv_cd": type("ConvCD", (), {"theta": 0})(),
                "multi_task_learning": True,
            },
        )

        # Build model
        model = build_model(test_config, device="cpu", strict=False, mode="train")
        print("   OK MobileNetV4 model building successful")

        # Test forward pass
        input_tensor = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = model(input_tensor)
            print(f"   OK Forward pass successful, output shape: {features.shape}")

    except Exception as e:
        print(f"   X MobileNetV4 model building failed: {e}")
        return False

    print("\nOK MobileNetV4 model building pipeline tested successfully!")
    return True


def main():
    """Run all MobileNetV4 tests"""

    print("=" * 60)
    print("MobileNetV4 Integration Test Suite")
    print("=" * 60)

    tests = [
        test_mobilenetv4_models,
        test_configuration_integration,
        test_model_building,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test {test.__name__} failed with exception: {e}")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("SUCCESS: All MobileNetV4 integration tests passed!")
        print("MobileNetV4 is ready for use in the face anti-spoofing project.")
    else:
        print("X Some tests failed. Please check the implementation.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
