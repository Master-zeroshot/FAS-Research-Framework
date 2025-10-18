"""
Centralized path configuration for the face anti-spoofing project.
This module contains all hardcoded paths and makes them configurable.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Dataset paths
DATASET_PATHS = {
    "celeba_spoof": "../CelebA-Spoof-zips/CelebA_Spoof",
    "lcc_fasd": "../LCC_FASD",
    "lcc_fasd_cropped": "./LCC_FASDcropped",
    "casia": "./CASIA",
    "celeba_spoof_local": "./CelebA_Spoof",
}

# Model and checkpoint paths
MODEL_PATHS = {
    "pretrained_dir": "./pretrained",
    "checkpoint_dir": "./checkpoints",
    "model_file": "./model.pkl",
}

# Logging paths
LOG_PATHS = {
    "app_log": "./logs/app.log",
    "logs_dir": "./logs",
    "logs_xgb": "./logs_xgb",
    "logs_mobilenetv4_small": "./logs_mobilenetv4_small",
    "logs_mobilenetv4_medium": "./logs_mobilenetv4_medium",
    "logs_mobilenetv4_large": "./logs_mobilenetv4_large",
}

# Pretrained model paths
PRETRAINED_MODEL_PATHS = {
    "mobilenetv2": "./pretrained/mobilenetv2-c5e733a8.pth",
    "mobilenetv3_small": "./pretrained/mobilenetv3-small-55df8e1f.pth",
    "mobilenetv3_large": "./pretrained/mobilenetv3-large-1cd25616.pth",
    "mobilenetv4_small": "./pretrained/mobilenetv4_small_imagenet.pth.tar",
    "mobilenetv4_medium": "./pretrained/mobilenetv4_medium_imagenet.pth.tar",
    "mobilenetv4_large": "./pretrained/mobilenetv4_large_imagenet.pth.tar",
    "mnv3_large_custom": "./pretrained/mnv3_large_224_gausian_80_20_epoch20.pth.tar",
}

# Output paths
OUTPUT_PATHS = {
    "results_dir": "./results",
    "visualizations_dir": "./visualizations",
    "curves_dir": "./curves",
}


def get_dataset_path(dataset_name: str) -> str:
    """Get the path for a specific dataset."""
    return DATASET_PATHS.get(dataset_name, "")


def get_model_path(model_name: str) -> str:
    """Get the path for a specific pretrained model."""
    return PRETRAINED_MODEL_PATHS.get(model_name, "")


def get_log_path(log_type: str) -> str:
    """Get the path for a specific log file."""
    return LOG_PATHS.get(log_type, "./logs/app.log")


def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        MODEL_PATHS["pretrained_dir"],
        MODEL_PATHS["checkpoint_dir"],
        LOG_PATHS["logs_dir"],
        OUTPUT_PATHS["results_dir"],
        OUTPUT_PATHS["visualizations_dir"],
        OUTPUT_PATHS["curves_dir"],
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


# Magic numbers and constants
class Constants:
    """Centralized constants and magic numbers."""

    # Image dimensions
    DEFAULT_IMAGE_SIZE = 224
    DEFAULT_BATCH_SIZE = 50

    # Model dimensions
    MOBILENETV2_EMBEDDING_DIM = 1280
    MOBILENETV3_SMALL_EMBEDDING_DIM = 1024
    MOBILENETV3_LARGE_EMBEDDING_DIM = 1280
    MOBILENETV4_SMALL_EMBEDDING_DIM = 1024
    MOBILENETV4_MEDIUM_EMBEDDING_DIM = 1152
    MOBILENETV4_LARGE_EMBEDDING_DIM = 1280

    # Training parameters
    DEFAULT_LEARNING_RATE = 0.005
    DEFAULT_MOMENTUM = 0.9
    DEFAULT_WEIGHT_DECAY = 5e-4

    # Data loading
    DEFAULT_NUM_WORKERS = 8
    DEFAULT_PIN_MEMORY = True

    # Evaluation metrics
    DEFAULT_FPR_THRESHOLD = 0.1
    DEFAULT_EER_THRESHOLD = 0.5
