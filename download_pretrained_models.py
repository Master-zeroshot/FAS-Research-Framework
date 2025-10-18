#!/usr/bin/env python3
"""
Download pretrained models for the face anti-spoofing project.
This script downloads the necessary pretrained models that are too large for git.
"""

import argparse
import hashlib
import os
import urllib.error
import urllib.request
from pathlib import Path

# Model URLs and checksums
PRETRAINED_MODELS = {
    # MobileNetV2
    "mobilenetv2-c5e733a8.pth": {
        "url": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
        "size": "13.6MB",
        "description": "MobileNetV2 ImageNet pretrained weights",
    },
    # MobileNetV3
    "mobilenetv3-large-1cd25616.pth": {
        "url": "https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth",
        "size": "20.9MB",
        "description": "MobileNetV3-Large ImageNet pretrained weights",
    },
    "mobilenetv3-small-55df8e1f.pth": {
        "url": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
        "size": "9.4MB",
        "description": "MobileNetV3-Small ImageNet pretrained weights",
    },
    # MobileNetV4 (using ImageNet pretrained weights as base)
    "mobilenetv4_small_imagenet.pth.tar": {
        "url": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
        "size": "9.4MB",
        "description": "MobileNetV4-Small ImageNet pretrained weights (using MobileNetV3-Small as base)",
        "note": "MobileNetV4 uses MobileNetV3 ImageNet weights as initialization",
    },
    "mobilenetv4_medium_imagenet.pth.tar": {
        "url": "https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth",
        "size": "20.9MB",
        "description": "MobileNetV4-Medium ImageNet pretrained weights (using MobileNetV3-Large as base)",
        "note": "MobileNetV4 uses MobileNetV3 ImageNet weights as initialization",
    },
    "mobilenetv4_large_imagenet.pth.tar": {
        "url": "https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth",
        "size": "20.9MB",
        "description": "MobileNetV4-Large ImageNet pretrained weights (using MobileNetV3-Large as base)",
        "note": "MobileNetV4 uses MobileNetV3 ImageNet weights as initialization",
    },
}


def download_file(url, filepath, description):
    """Download a file with progress indication"""
    print(f"Downloading {description}...")
    print(f"URL: {url}")
    print(f"Destination: {filepath}")

    try:

        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) / total_size)
                print(
                    f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)",
                    end="",
                )

        urllib.request.urlretrieve(url, filepath, progress_hook)
        print(f"\nOK Successfully downloaded {filepath}")
        return True

    except urllib.error.URLError as e:
        print(f"\nX Failed to download {url}: {e}")
        return False
    except Exception as e:
        print(f"\nX Error downloading {url}: {e}")
        return False


def create_pretrained_directory():
    """Create the pretrained directory if it doesn't exist"""
    pretrained_dir = Path("pretrained")
    pretrained_dir.mkdir(exist_ok=True)
    return pretrained_dir


def list_available_models():
    """List all available pretrained models"""
    print("Available pretrained models:")
    print("=" * 60)

    for filename, info in PRETRAINED_MODELS.items():
        print(f"[FILE] {filename}")
        print(f"   Description: {info['description']}")
        print(f"   Size: {info['size']}")
        if "note" in info:
            print(f"   Note: {info['note']}")
        print()


def download_models(model_names=None, all_models=False):
    """Download specified models or all models"""
    pretrained_dir = create_pretrained_directory()

    if all_models:
        models_to_download = list(PRETRAINED_MODELS.keys())
    elif model_names:
        models_to_download = model_names
    else:
        print(
            "No models specified. Use --all or --models to specify which models to download."
        )
        return

    print(f"Downloading {len(models_to_download)} model(s)...")
    print("=" * 60)

    success_count = 0
    for model_name in models_to_download:
        if model_name not in PRETRAINED_MODELS:
            print(f"X Unknown model: {model_name}")
            continue

        model_info = PRETRAINED_MODELS[model_name]
        filepath = pretrained_dir / model_name

        # Check if file already exists
        if filepath.exists():
            print(f">> {model_name} already exists, skipping...")
            success_count += 1
            continue

        if download_file(model_info["url"], filepath, model_info["description"]):
            success_count += 1

    print("\n" + "=" * 60)
    print(
        f"Download complete: {success_count}/{len(models_to_download)} models downloaded successfully"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained models for face anti-spoofing"
    )
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--models", nargs="+", help="Download specific models")

    args = parser.parse_args()

    if args.list:
        list_available_models()
        return

    if args.all:
        download_models(all_models=True)
    elif args.models:
        download_models(model_names=args.models)
    else:
        print("Face Anti-Spoofing Pretrained Models Downloader")
        print("=" * 50)
        print()
        print("Usage:")
        print(
            "  python download_pretrained_models.py --list                    # List available models"
        )
        print(
            "  python download_pretrained_models.py --all                     # Download all models"
        )
        print(
            "  python download_pretrained_models.py --models model1 model2    # Download specific models"
        )
        print()
        print("Examples:")
        print(
            "  python download_pretrained_models.py --models mobilenetv2-c5e733a8.pth"
        )
        print(
            "  python download_pretrained_models.py --models mobilenetv3-large-1cd25616.pth mobilenetv3-small-55df8e1f.pth"
        )
        print()
        print("Note: Some custom models may not be available for download yet.")
        print("You may need to train them yourself or obtain them from other sources.")


if __name__ == "__main__":
    main()
