"""
Core modules for FAS-Research-Framework

This package contains the core functionality including:
- models: Model architectures (MobileNetV2/V3/V4, EfficientNet, ViT, ResNet)
- datasets: Dataset implementations and utilities
- losses: Loss functions for training
- training: Training utilities and trainers
- evaluation: Evaluation metrics and protocols
"""

from . import models
from . import datasets
from . import losses
from . import training
from . import evaluation

__all__ = ["models", "datasets", "losses", "training", "evaluation"]
