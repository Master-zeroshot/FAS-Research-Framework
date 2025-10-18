"""
FAS-Research-Framework: A Comprehensive Face Anti-Spoofing Research Platform

This package provides a state-of-the-art research framework for face anti-spoofing,
including multiple model architectures, advanced training techniques, comprehensive
evaluation frameworks, and rich visualization tools.

Author: Aseyed Mostafa
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Aseyed Mostafa"
__email__ = "aseyedmostafaho@gmail.com"
__license__ = "MIT"

# Core modules
try:
    from .core import models, datasets, losses, training, evaluation
except ImportError as e:
    print(f"Warning: Could not import core modules: {e}")
    models = datasets = losses = training = evaluation = None

# Research modules
try:
    from .research import advanced_training, visualization, analysis
except ImportError as e:
    print(f"Warning: Could not import research modules: {e}")
    advanced_training = visualization = analysis = None

# Tools and utilities
try:
    from .tools import *
except ImportError as e:
    print(f"Warning: Could not import tools: {e}")

# Package metadata
__all__ = [
    "models",
    "datasets", 
    "losses",
    "training",
    "evaluation",
    "advanced_training",
    "visualization",
    "analysis",
    "__version__",
    "__author__",
    "__email__",
    "__license__"
]
