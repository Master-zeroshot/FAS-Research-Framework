"""
Loss functions for FAS-Research-Framework

This module contains loss function implementations:
- AM-Softmax loss
- Cross-entropy loss
- Binary cross-entropy loss
- SVM loss
"""

from .am_softmax import *

__all__ = [
    "AMSoftmaxLoss",
    "CrossEntropyLoss",
    "BCELoss",
    "SVMLoss"
]