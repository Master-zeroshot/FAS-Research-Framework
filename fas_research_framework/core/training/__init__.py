"""
Training modules for FAS-Research-Framework

This module contains training utilities and trainers:
- Main training script
- Trainer class
- Training utilities
"""

from .train import *
from .trainer import *

__all__ = [
    "main",
    "Trainer"
]
