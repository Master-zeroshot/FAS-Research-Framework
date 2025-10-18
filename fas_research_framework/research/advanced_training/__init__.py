"""
Advanced training techniques for FAS-Research-Framework

This module contains advanced training methods:
- Adversarial training (FGSM, PGD, C&W)
- Knowledge distillation (Logit, Feature, Attention)
- Self-supervised learning (SimCLR, BYOL)
- Curriculum learning
- Meta-learning (MAML, Prototypical Networks)
"""

from .adversarial_training import *
from .knowledge_distillation import *
from .self_supervised_learning import *
from .curriculum_learning import *
from .meta_learning import *

__all__ = [
    "AdversarialTrainer",
    "KnowledgeDistillationTrainer",
    "SelfSupervisedTrainer",
    "CurriculumLearningTrainer",
    "MetaLearningTrainer"
]