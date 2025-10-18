"""
Knowledge Distillation Implementation for Face Anti-Spoofing

This module implements knowledge distillation techniques to transfer knowledge
from a large teacher model to a smaller student model for face anti-spoofing tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable
from abc import ABC, abstractmethod

from utils.logging_config import get_logger

logger = get_logger(__name__)


class DistillationLoss(nn.Module):
    """Knowledge distillation loss function."""
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.7, 
                 beta: float = 0.3, reduction: str = 'mean'):
        """
        Initialize distillation loss.
        
        Args:
            temperature: Temperature for softmax
            alpha: Weight for distillation loss
            beta: Weight for student loss
            reduction: Reduction method for loss
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute distillation loss.
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            targets: Ground truth labels
            
        Returns:
            Distillation loss
        """
        # Softmax with temperature
        student_soft = F.softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # Distillation loss (KL divergence)
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            teacher_soft,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Student loss (cross entropy)
        student_loss = F.cross_entropy(student_logits, targets)
        
        # Total loss
        total_loss = self.alpha * distillation_loss + self.beta * student_loss
        
        return total_loss


class FeatureDistillationLoss(nn.Module):
    """Feature-based knowledge distillation loss."""
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, 
                 feature_layers: List[str] = None):
        """
        Initialize feature distillation loss.
        
        Args:
            alpha: Weight for feature distillation
            beta: Weight for logit distillation
            feature_layers: List of layer names for feature distillation
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.feature_layers = feature_layers or ['features']
    
    def forward(self, student_features: Dict[str, torch.Tensor], 
                teacher_features: Dict[str, torch.Tensor],
                student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute feature distillation loss.
        
        Args:
            student_features: Student model features
            teacher_features: Teacher model features
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            targets: Ground truth labels
            
        Returns:
            Feature distillation loss
        """
        # Logit distillation
        logit_loss = F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction='batchmean'
        )
        
        # Feature distillation
        feature_loss = 0.0
        for layer_name in self.feature_layers:
            if layer_name in student_features and layer_name in teacher_features:
                student_feat = student_features[layer_name]
                teacher_feat = teacher_features[layer_name]
                
                # L2 loss between features
                feature_loss += F.mse_loss(student_feat, teacher_feat)
        
        # Total loss
        total_loss = self.alpha * feature_loss + self.beta * logit_loss
        
        return total_loss


class AttentionDistillationLoss(nn.Module):
    """Attention-based knowledge distillation loss."""
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        """
        Initialize attention distillation loss.
        
        Args:
            alpha: Weight for attention distillation
            beta: Weight for logit distillation
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, student_attention: torch.Tensor, 
                teacher_attention: torch.Tensor,
                student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute attention distillation loss.
        
        Args:
            student_attention: Student model attention maps
            teacher_attention: Teacher model attention maps
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            targets: Ground truth labels
            
        Returns:
            Attention distillation loss
        """
        # Logit distillation
        logit_loss = F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction='batchmean'
        )
        
        # Attention distillation (L2 loss)
        attention_loss = F.mse_loss(student_attention, teacher_attention)
        
        # Total loss
        total_loss = self.alpha * attention_loss + self.beta * logit_loss
        
        return total_loss


class KnowledgeDistillationTrainer:
    """Knowledge distillation trainer for face anti-spoofing."""
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module,
                 device: str = "cuda", temperature: float = 3.0,
                 alpha: float = 0.7, beta: float = 0.3):
        """
        Initialize knowledge distillation trainer.
        
        Args:
            teacher_model: Teacher model
            student_model: Student model
            device: Device to run training on
            temperature: Temperature for distillation
            alpha: Weight for distillation loss
            beta: Weight for student loss
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.device = device
        
        # Set models to appropriate mode
        self.teacher_model.eval()
        self.student_model.train()
        
        # Initialize loss function
        self.distillation_loss = DistillationLoss(temperature, alpha, beta)
        
        logger.info("Initialized knowledge distillation trainer")
    
    def distillation_step(self, x: torch.Tensor, y: torch.Tensor,
                         optimizer: optim.Optimizer) -> Dict[str, float]:
        """
        Perform one distillation training step.
        
        Args:
            x: Input images
            y: Target labels
            optimizer: Optimizer for student model
            
        Returns:
            Dictionary with loss values
        """
        # Zero gradients
        optimizer.zero_grad()
        
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_logits = self.teacher_model(x)
        
        # Student forward pass
        student_logits = self.student_model(x)
        
        # Compute distillation loss
        loss = self.distillation_loss(student_logits, teacher_logits, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            pred = student_logits.argmax(dim=1)
            accuracy = (pred == y).float().mean().item()
        
        return {
            'distillation_loss': loss.item(),
            'student_accuracy': accuracy
        }
    
    def evaluate_student(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate student model performance.
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.student_model.eval()
        
        total_samples = 0
        correct_predictions = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                
                # Student prediction
                student_logits = self.student_model(x)
                pred = student_logits.argmax(dim=1)
                
                # Teacher prediction
                teacher_logits = self.teacher_model(x)
                
                # Compute loss
                loss = self.distillation_loss(student_logits, teacher_logits, y)
                
                # Update metrics
                correct_predictions += (pred == y).sum().item()
                total_loss += loss.item()
                total_samples += x.size(0)
        
        accuracy = correct_predictions / total_samples
        avg_loss = total_loss / len(dataloader)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss
        }


class ProgressiveDistillation:
    """Progressive knowledge distillation."""
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module,
                 device: str = "cuda"):
        """
        Initialize progressive distillation.
        
        Args:
            teacher_model: Teacher model
            student_model: Student model
            device: Device to run training on
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.device = device
        
        logger.info("Initialized progressive distillation")
    
    def progressive_distillation_step(self, x: torch.Tensor, y: torch.Tensor,
                                    optimizer: optim.Optimizer, 
                                    temperature: float = 3.0,
                                    alpha: float = 0.7, beta: float = 0.3) -> Dict[str, float]:
        """
        Perform progressive distillation step.
        
        Args:
            x: Input images
            y: Target labels
            optimizer: Optimizer
            temperature: Temperature for distillation
            alpha: Weight for distillation loss
            beta: Weight for student loss
            
        Returns:
            Dictionary with loss values
        """
        # Zero gradients
        optimizer.zero_grad()
        
        # Teacher forward pass
        with torch.no_grad():
            teacher_logits = self.teacher_model(x)
        
        # Student forward pass
        student_logits = self.student_model(x)
        
        # Progressive distillation loss
        distillation_loss = DistillationLoss(temperature, alpha, beta)
        loss = distillation_loss(student_logits, teacher_logits, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return {
            'progressive_loss': loss.item(),
            'temperature': temperature,
            'alpha': alpha,
            'beta': beta
        }


class MultiTeacherDistillation:
    """Multi-teacher knowledge distillation."""
    
    def __init__(self, teacher_models: List[nn.Module], student_model: nn.Module,
                 device: str = "cuda", teacher_weights: List[float] = None):
        """
        Initialize multi-teacher distillation.
        
        Args:
            teacher_models: List of teacher models
            student_model: Student model
            device: Device to run training on
            teacher_weights: Weights for each teacher
        """
        self.teacher_models = teacher_models
        self.student_model = student_model
        self.device = device
        self.teacher_weights = teacher_weights or [1.0] * len(teacher_models)
        
        # Set models to appropriate mode
        for teacher in self.teacher_models:
            teacher.eval()
        self.student_model.train()
        
        logger.info(f"Initialized multi-teacher distillation with {len(teacher_models)} teachers")
    
    def multi_teacher_step(self, x: torch.Tensor, y: torch.Tensor,
                          optimizer: optim.Optimizer, temperature: float = 3.0,
                          alpha: float = 0.7, beta: float = 0.3) -> Dict[str, float]:
        """
        Perform multi-teacher distillation step.
        
        Args:
            x: Input images
            y: Target labels
            optimizer: Optimizer
            temperature: Temperature for distillation
            alpha: Weight for distillation loss
            beta: Weight for student loss
            
        Returns:
            Dictionary with loss values
        """
        # Zero gradients
        optimizer.zero_grad()
        
        # Get teacher predictions
        teacher_logits_list = []
        with torch.no_grad():
            for teacher in self.teacher_models:
                teacher_logits = teacher(x)
                teacher_logits_list.append(teacher_logits)
        
        # Average teacher predictions
        teacher_logits = torch.stack(teacher_logits_list, dim=0)
        teacher_weights = torch.tensor(self.teacher_weights, device=self.device).view(-1, 1, 1)
        teacher_logits = (teacher_logits * teacher_weights).sum(dim=0) / teacher_weights.sum()
        
        # Student forward pass
        student_logits = self.student_model(x)
        
        # Multi-teacher distillation loss
        distillation_loss = DistillationLoss(temperature, alpha, beta)
        loss = distillation_loss(student_logits, teacher_logits, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return {
            'multi_teacher_loss': loss.item(),
            'num_teachers': len(self.teacher_models),
            'temperature': temperature
        }


class DistillationConfig:
    """Configuration for knowledge distillation."""
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.7, beta: float = 0.3,
                 progressive: bool = False, multi_teacher: bool = False,
                 feature_distillation: bool = False, attention_distillation: bool = False):
        """
        Initialize distillation configuration.
        
        Args:
            temperature: Temperature for distillation
            alpha: Weight for distillation loss
            beta: Weight for student loss
            progressive: Whether to use progressive distillation
            multi_teacher: Whether to use multi-teacher distillation
            feature_distillation: Whether to use feature distillation
            attention_distillation: Whether to use attention distillation
        """
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.progressive = progressive
        self.multi_teacher = multi_teacher
        self.feature_distillation = feature_distillation
        self.attention_distillation = attention_distillation


def create_distillation_trainer(teacher_model: nn.Module, student_model: nn.Module,
                               config: DistillationConfig, device: str = "cuda"):
    """
    Create knowledge distillation trainer.
    
    Args:
        teacher_model: Teacher model
        student_model: Student model
        config: Distillation configuration
        device: Device to run training on
        
    Returns:
        Distillation trainer instance
    """
    if config.multi_teacher:
        return MultiTeacherDistillation([teacher_model], student_model, device)
    elif config.progressive:
        return ProgressiveDistillation(teacher_model, student_model, device)
    else:
        return KnowledgeDistillationTrainer(teacher_model, student_model, device)


def distillation_training_step(teacher_model: nn.Module, student_model: nn.Module,
                               x: torch.Tensor, y: torch.Tensor, optimizer: optim.Optimizer,
                               config: DistillationConfig, device: str = "cuda") -> Dict[str, float]:
    """
    Perform one distillation training step.
    
    Args:
        teacher_model: Teacher model
        student_model: Student model
        x: Input images
        y: Target labels
        optimizer: Optimizer
        config: Distillation configuration
        device: Device to run training on
        
    Returns:
        Dictionary with loss values
    """
    trainer = create_distillation_trainer(teacher_model, student_model, config, device)
    return trainer.distillation_step(x, y, optimizer)
