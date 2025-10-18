#!/usr/bin/env python3
"""
Check if pretrained models are available and provide guidance.
"""

import os
from pathlib import Path


def check_pretrained_models():
    """Check which pretrained models are available"""
    pretrained_dir = Path("pretrained")

    if not pretrained_dir.exists():
        print("X Pretrained directory does not exist")
        return False

    # List of expected pretrained models
    expected_models = {
        "mobilenetv2-c5e733a8.pth": "MobileNetV2 ImageNet weights",
        "mobilenetv3-large-1cd25616.pth": "MobileNetV3-Large ImageNet weights",
        "mobilenetv3-small-55df8e1f.pth": "MobileNetV3-Small ImageNet weights",
        "mobilenetv4_small_imagenet.pth.tar": "MobileNetV4-Small ImageNet weights",
        "mobilenetv4_medium_imagenet.pth.tar": "MobileNetV4-Medium ImageNet weights",
        "mobilenetv4_large_imagenet.pth.tar": "MobileNetV4-Large ImageNet weights",
    }

    available_models = []
    missing_models = []

    for model_file, description in expected_models.items():
        model_path = pretrained_dir / model_file
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            available_models.append((model_file, description, f"{size_mb:.1f}MB"))
        else:
            missing_models.append((model_file, description))

    print("Pretrained Models Status")
    print("=" * 50)

    if available_models:
        print("OK Available models:")
        for model_file, description, size in available_models:
            print(f"   [FILE] {model_file} ({size}) - {description}")
        print()

    if missing_models:
        print("X Missing models:")
        for model_file, description in missing_models:
            print(f"   [FILE] {model_file} - {description}")
        print()

    if missing_models:
        print("TIP: To download missing models:")
        print("   python download_pretrained_models.py --list")
        print("   python download_pretrained_models.py --models <model_name>")
        print()
        print("TIP: To train without pretrained weights:")
        print("   Set 'pretrained=False' in your config files")
        print()

    return len(available_models) > 0


def check_config_pretrained_settings():
    """Check if config files are set up for missing pretrained models"""
    config_files = [
        "configs/config_large.py",
        "configs/config_small.py",
        "configs/config_mobilenetv4_large.py",
        "configs/config_mobilenetv4_medium.py",
        "configs/config_mobilenetv4_small.py",
    ]

    print("Configuration Files Status")
    print("=" * 50)

    for config_file in config_files:
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                content = f.read()
                if "pretrained=False" in content:
                    print(f"OK {config_file} - Set to train from scratch")
                elif "pretrained=True" in content:
                    print(f"! {config_file} - Requires pretrained weights")
                else:
                    print(f"? {config_file} - Unknown pretrained setting")
        else:
            print(f"X {config_file} - File not found")


def main():
    print("Face Anti-Spoofing - Pretrained Models Checker")
    print("=" * 60)
    print()

    has_models = check_pretrained_models()
    print()
    check_config_pretrained_settings()

    print()
    print("Summary:")
    if has_models:
        print("OK Some pretrained models are available")
        print("   You can start training with pretrained weights")
    else:
        print("! No pretrained models found")
        print("   You can either:")
        print("   1. Download pretrained models using download_pretrained_models.py")
        print("   2. Train from scratch (configs are already set up for this)")

    print()
    print("For more information, see SETUP_GUIDE.md")


if __name__ == "__main__":
    main()
