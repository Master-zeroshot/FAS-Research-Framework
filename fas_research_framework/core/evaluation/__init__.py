"""
Evaluation modules for FAS-Research-Framework

This module contains evaluation utilities and protocols:
- Evaluation protocol
- Metrics calculation
- Performance analysis
"""

from .eval_protocol import *

__all__ = [
    "evaluate_model",
    "calculate_metrics",
    "generate_report"
]
