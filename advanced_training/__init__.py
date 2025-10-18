"""
Advanced Training Techniques for Face Anti-Spoofing

This module provides advanced training techniques including adversarial training,
knowledge distillation, self-supervised learning, curriculum learning, and meta-learning.
"""

from .adversarial_training import (
    AdversarialTrainer, AdversarialTrainingConfig, AdversarialTrainingScheduler,
    FGSMAttack, PGDAttack, CWAttack, create_adversarial_trainer,
    adversarial_training_step, evaluate_adversarial_robustness
)

from .knowledge_distillation import (
    DistillationLoss, FeatureDistillationLoss, AttentionDistillationLoss,
    KnowledgeDistillationTrainer, ProgressiveDistillation, MultiTeacherDistillation,
    DistillationConfig, create_distillation_trainer, distillation_training_step
)

from .self_supervised_learning import (
    ContrastiveLoss, SimCLRLoss, BYOLLoss, DataAugmentation,
    SelfSupervisedTrainer, SelfSupervisedConfig, create_self_supervised_trainer,
    self_supervised_training_step
)

from .curriculum_learning import (
    DifficultyEstimator, LossBasedDifficulty, ConfidenceBasedDifficulty, GradientBasedDifficulty,
    CurriculumScheduler, LinearCurriculum, ExponentialCurriculum, CosineCurriculum,
    CurriculumTrainer, AdaptiveCurriculum, CurriculumConfig, create_curriculum_trainer,
    curriculum_training_step
)

from .meta_learning import (
    MAML, PrototypicalNetworks, RelationNetworks, MetaLearningTrainer,
    MetaLearningConfig, create_meta_learning_trainer, meta_learning_training_step
)

__all__ = [
    # Adversarial Training
    'AdversarialTrainer', 'AdversarialTrainingConfig', 'AdversarialTrainingScheduler',
    'FGSMAttack', 'PGDAttack', 'CWAttack', 'create_adversarial_trainer',
    'adversarial_training_step', 'evaluate_adversarial_robustness',
    
    # Knowledge Distillation
    'DistillationLoss', 'FeatureDistillationLoss', 'AttentionDistillationLoss',
    'KnowledgeDistillationTrainer', 'ProgressiveDistillation', 'MultiTeacherDistillation',
    'DistillationConfig', 'create_distillation_trainer', 'distillation_training_step',
    
    # Self-Supervised Learning
    'ContrastiveLoss', 'SimCLRLoss', 'BYOLLoss', 'DataAugmentation',
    'SelfSupervisedTrainer', 'SelfSupervisedConfig', 'create_self_supervised_trainer',
    'self_supervised_training_step',
    
    # Curriculum Learning
    'DifficultyEstimator', 'LossBasedDifficulty', 'ConfidenceBasedDifficulty', 'GradientBasedDifficulty',
    'CurriculumScheduler', 'LinearCurriculum', 'ExponentialCurriculum', 'CosineCurriculum',
    'CurriculumTrainer', 'AdaptiveCurriculum', 'CurriculumConfig', 'create_curriculum_trainer',
    'curriculum_training_step',
    
    # Meta-Learning
    'MAML', 'PrototypicalNetworks', 'RelationNetworks', 'MetaLearningTrainer',
    'MetaLearningConfig', 'create_meta_learning_trainer', 'meta_learning_training_step'
]
