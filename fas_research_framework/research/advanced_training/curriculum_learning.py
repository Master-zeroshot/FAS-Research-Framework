"""
Curriculum Learning Implementation for Face Anti-Spoofing

This module implements curriculum learning techniques to train models
on progressively more difficult examples for face anti-spoofing tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable
from abc import ABC, abstractmethod
import random

from utils.logging_config import get_logger

logger = get_logger(__name__)


class DifficultyEstimator(ABC):
    """Base class for difficulty estimation."""
    
    @abstractmethod
    def estimate_difficulty(self, x: torch.Tensor, y: torch.Tensor, 
                           model: nn.Module) -> torch.Tensor:
        """Estimate difficulty of examples."""
        pass


class LossBasedDifficulty(DifficultyEstimator):
    """Difficulty estimation based on loss."""
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize loss-based difficulty estimator.
        
        Args:
            temperature: Temperature for difficulty computation
        """
        self.temperature = temperature
    
    def estimate_difficulty(self, x: torch.Tensor, y: torch.Tensor, 
                           model: nn.Module) -> torch.Tensor:
        """Estimate difficulty based on loss."""
        with torch.no_grad():
            output = model(x)
            loss = F.cross_entropy(output, y, reduction='none')
            difficulty = loss / self.temperature
        
        return difficulty


class ConfidenceBasedDifficulty(DifficultyEstimator):
    """Difficulty estimation based on model confidence."""
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize confidence-based difficulty estimator.
        
        Args:
            temperature: Temperature for difficulty computation
        """
        self.temperature = temperature
    
    def estimate_difficulty(self, x: torch.Tensor, y: torch.Tensor, 
                           model: nn.Module) -> torch.Tensor:
        """Estimate difficulty based on confidence."""
        with torch.no_grad():
            output = model(x)
            probs = F.softmax(output, dim=1)
            confidence = probs.max(dim=1)[0]
            difficulty = 1.0 - confidence
        
        return difficulty


class GradientBasedDifficulty(DifficultyEstimator):
    """Difficulty estimation based on gradients."""
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize gradient-based difficulty estimator.
        
        Args:
            temperature: Temperature for difficulty computation
        """
        self.temperature = temperature
    
    def estimate_difficulty(self, x: torch.Tensor, y: torch.Tensor, 
                           model: nn.Module) -> torch.Tensor:
        """Estimate difficulty based on gradients."""
        x.requires_grad_(True)
        
        output = model(x)
        loss = F.cross_entropy(output, y, reduction='none')
        
        # Compute gradients
        gradients = torch.autograd.grad(
            loss.sum(), x, retain_graph=True, create_graph=True
        )[0]
        
        # Compute gradient magnitude
        gradient_magnitude = torch.norm(gradients.view(gradients.size(0), -1), dim=1)
        difficulty = gradient_magnitude / self.temperature
        
        return difficulty


class CurriculumScheduler(ABC):
    """Base class for curriculum scheduling."""
    
    @abstractmethod
    def get_curriculum_ratio(self, epoch: int, total_epochs: int) -> float:
        """Get curriculum ratio for current epoch."""
        pass


class LinearCurriculum(CurriculumScheduler):
    """Linear curriculum scheduling."""
    
    def __init__(self, start_ratio: float = 0.1, end_ratio: float = 1.0):
        """
        Initialize linear curriculum.
        
        Args:
            start_ratio: Starting curriculum ratio
            end_ratio: Ending curriculum ratio
        """
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
    
    def get_curriculum_ratio(self, epoch: int, total_epochs: int) -> float:
        """Get linear curriculum ratio."""
        progress = epoch / total_epochs
        ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * progress
        return min(ratio, 1.0)


class ExponentialCurriculum(CurriculumScheduler):
    """Exponential curriculum scheduling."""
    
    def __init__(self, start_ratio: float = 0.1, end_ratio: float = 1.0, 
                 gamma: float = 2.0):
        """
        Initialize exponential curriculum.
        
        Args:
            start_ratio: Starting curriculum ratio
            end_ratio: Ending curriculum ratio
            gamma: Exponential factor
        """
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.gamma = gamma
    
    def get_curriculum_ratio(self, epoch: int, total_epochs: int) -> float:
        """Get exponential curriculum ratio."""
        progress = epoch / total_epochs
        ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * (progress ** self.gamma)
        return min(ratio, 1.0)


class CosineCurriculum(CurriculumScheduler):
    """Cosine curriculum scheduling."""
    
    def __init__(self, start_ratio: float = 0.1, end_ratio: float = 1.0):
        """
        Initialize cosine curriculum.
        
        Args:
            start_ratio: Starting curriculum ratio
            end_ratio: Ending curriculum ratio
        """
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
    
    def get_curriculum_ratio(self, epoch: int, total_epochs: int) -> float:
        """Get cosine curriculum ratio."""
        progress = epoch / total_epochs
        ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * (1 - np.cos(np.pi * progress)) / 2
        return min(ratio, 1.0)


class CurriculumTrainer:
    """Curriculum learning trainer for face anti-spoofing."""
    
    def __init__(self, model: nn.Module, difficulty_estimator: DifficultyEstimator,
                 curriculum_scheduler: CurriculumScheduler, device: str = "cuda"):
        """
        Initialize curriculum trainer.
        
        Args:
            model: Model to train
            difficulty_estimator: Difficulty estimation method
            curriculum_scheduler: Curriculum scheduling method
            device: Device to run training on
        """
        self.model = model
        self.difficulty_estimator = difficulty_estimator
        self.curriculum_scheduler = curriculum_scheduler
        self.device = device
        
        logger.info("Initialized curriculum learning trainer")
    
    def get_curriculum_batch(self, x: torch.Tensor, y: torch.Tensor, 
                            epoch: int, total_epochs: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get curriculum batch based on current epoch.
        
        Args:
            x: Input images
            y: Target labels
            epoch: Current epoch
            total_epochs: Total number of epochs
            
        Returns:
            Tuple of (curriculum_x, curriculum_y)
        """
        # Get curriculum ratio
        curriculum_ratio = self.curriculum_scheduler.get_curriculum_ratio(epoch, total_epochs)
        
        # Estimate difficulty
        difficulty = self.difficulty_estimator.estimate_difficulty(x, y, self.model)
        
        # Select examples based on curriculum
        num_samples = int(len(x) * curriculum_ratio)
        
        if num_samples == 0:
            return x, y
        
        # Sort by difficulty and select easiest examples
        sorted_indices = torch.argsort(difficulty)
        selected_indices = sorted_indices[:num_samples]
        
        curriculum_x = x[selected_indices]
        curriculum_y = y[selected_indices]
        
        return curriculum_x, curriculum_y
    
    def training_step(self, x: torch.Tensor, y: torch.Tensor, 
                     optimizer: optim.Optimizer, epoch: int, 
                     total_epochs: int) -> Dict[str, float]:
        """
        Perform one curriculum training step.
        
        Args:
            x: Input images
            y: Target labels
            optimizer: Optimizer
            epoch: Current epoch
            total_epochs: Total number of epochs
            
        Returns:
            Dictionary with loss values
        """
        # Get curriculum batch
        curriculum_x, curriculum_y = self.get_curriculum_batch(x, y, epoch, total_epochs)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = self.model(curriculum_x)
        loss = F.cross_entropy(output, curriculum_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            pred = output.argmax(dim=1)
            accuracy = (pred == curriculum_y).float().mean().item()
        
        # Get curriculum ratio
        curriculum_ratio = self.curriculum_scheduler.get_curriculum_ratio(epoch, total_epochs)
        
        return {
            'curriculum_loss': loss.item(),
            'curriculum_accuracy': accuracy,
            'curriculum_ratio': curriculum_ratio,
            'batch_size': curriculum_x.size(0)
        }
    
    def evaluate_curriculum_progress(self, dataloader: torch.utils.data.DataLoader,
                                   epoch: int, total_epochs: int) -> Dict[str, float]:
        """
        Evaluate curriculum learning progress.
        
        Args:
            dataloader: Data loader for evaluation
            epoch: Current epoch
            total_epochs: Total number of epochs
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        all_difficulties = []
        all_accuracies = []
        
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                
                # Estimate difficulty
                difficulty = self.difficulty_estimator.estimate_difficulty(x, y, self.model)
                all_difficulties.append(difficulty.cpu())
                
                # Compute accuracy
                output = self.model(x)
                pred = output.argmax(dim=1)
                accuracy = (pred == y).float()
                all_accuracies.append(accuracy.cpu())
        
        # Concatenate all difficulties and accuracies
        all_difficulties = torch.cat(all_difficulties, dim=0)
        all_accuracies = torch.cat(all_accuracies, dim=0)
        
        # Compute curriculum metrics
        curriculum_ratio = self.curriculum_scheduler.get_curriculum_ratio(epoch, total_epochs)
        num_easy = int(len(all_difficulties) * curriculum_ratio)
        
        if num_easy > 0:
            easy_difficulties = all_difficulties[:num_easy]
            easy_accuracies = all_accuracies[:num_easy]
            
            easy_accuracy = easy_accuracies.mean().item()
            easy_difficulty = easy_difficulties.mean().item()
        else:
            easy_accuracy = 0.0
            easy_difficulty = 0.0
        
        overall_accuracy = all_accuracies.mean().item()
        overall_difficulty = all_difficulties.mean().item()
        
        return {
            'curriculum_ratio': curriculum_ratio,
            'easy_accuracy': easy_accuracy,
            'easy_difficulty': easy_difficulty,
            'overall_accuracy': overall_accuracy,
            'overall_difficulty': overall_difficulty
        }


class AdaptiveCurriculum:
    """Adaptive curriculum learning."""
    
    def __init__(self, model: nn.Module, difficulty_estimator: DifficultyEstimator,
                 device: str = "cuda", adaptation_rate: float = 0.1):
        """
        Initialize adaptive curriculum.
        
        Args:
            model: Model to train
            difficulty_estimator: Difficulty estimation method
            device: Device to run training on
            adaptation_rate: Rate of curriculum adaptation
        """
        self.model = model
        self.difficulty_estimator = difficulty_estimator
        self.device = device
        self.adaptation_rate = adaptation_rate
        
        # Initialize curriculum parameters
        self.curriculum_ratio = 0.1
        self.performance_history = []
        
        logger.info("Initialized adaptive curriculum learning")
    
    def adapt_curriculum(self, performance: float) -> float:
        """
        Adapt curriculum based on performance.
        
        Args:
            performance: Current performance metric
            
        Returns:
            Updated curriculum ratio
        """
        # Update performance history
        self.performance_history.append(performance)
        
        # Keep only recent history
        if len(self.performance_history) > 10:
            self.performance_history = self.performance_history[-10:]
        
        # Adapt curriculum based on performance trend
        if len(self.performance_history) >= 3:
            recent_trend = np.mean(self.performance_history[-3:]) - np.mean(self.performance_history[-6:-3])
            
            if recent_trend > 0.01:  # Performance improving
                self.curriculum_ratio = min(self.curriculum_ratio + self.adaptation_rate, 1.0)
            elif recent_trend < -0.01:  # Performance decreasing
                self.curriculum_ratio = max(self.curriculum_ratio - self.adaptation_rate, 0.1)
        
        return self.curriculum_ratio
    
    def get_adaptive_batch(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get adaptive curriculum batch.
        
        Args:
            x: Input images
            y: Target labels
            
        Returns:
            Tuple of (curriculum_x, curriculum_y)
        """
        # Estimate difficulty
        difficulty = self.difficulty_estimator.estimate_difficulty(x, y, self.model)
        
        # Select examples based on current curriculum ratio
        num_samples = int(len(x) * self.curriculum_ratio)
        
        if num_samples == 0:
            return x, y
        
        # Sort by difficulty and select easiest examples
        sorted_indices = torch.argsort(difficulty)
        selected_indices = sorted_indices[:num_samples]
        
        curriculum_x = x[selected_indices]
        curriculum_y = y[selected_indices]
        
        return curriculum_x, curriculum_y


class CurriculumConfig:
    """Configuration for curriculum learning."""
    
    def __init__(self, difficulty_method: str = "loss", curriculum_method: str = "linear",
                 start_ratio: float = 0.1, end_ratio: float = 1.0, 
                 adaptation_rate: float = 0.1):
        """
        Initialize curriculum configuration.
        
        Args:
            difficulty_method: Method for difficulty estimation
            curriculum_method: Method for curriculum scheduling
            start_ratio: Starting curriculum ratio
            end_ratio: Ending curriculum ratio
            adaptation_rate: Rate of curriculum adaptation
        """
        self.difficulty_method = difficulty_method
        self.curriculum_method = curriculum_method
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.adaptation_rate = adaptation_rate


def create_curriculum_trainer(model: nn.Module, config: CurriculumConfig,
                            device: str = "cuda") -> CurriculumTrainer:
    """
    Create curriculum trainer.
    
    Args:
        model: Model to train
        config: Curriculum configuration
        device: Device to run training on
        
    Returns:
        Curriculum trainer instance
    """
    # Create difficulty estimator
    if config.difficulty_method == "loss":
        difficulty_estimator = LossBasedDifficulty()
    elif config.difficulty_method == "confidence":
        difficulty_estimator = ConfidenceBasedDifficulty()
    elif config.difficulty_method == "gradient":
        difficulty_estimator = GradientBasedDifficulty()
    else:
        raise ValueError(f"Unknown difficulty method: {config.difficulty_method}")
    
    # Create curriculum scheduler
    if config.curriculum_method == "linear":
        curriculum_scheduler = LinearCurriculum(config.start_ratio, config.end_ratio)
    elif config.curriculum_method == "exponential":
        curriculum_scheduler = ExponentialCurriculum(config.start_ratio, config.end_ratio)
    elif config.curriculum_method == "cosine":
        curriculum_scheduler = CosineCurriculum(config.start_ratio, config.end_ratio)
    else:
        raise ValueError(f"Unknown curriculum method: {config.curriculum_method}")
    
    return CurriculumTrainer(model, difficulty_estimator, curriculum_scheduler, device)


def curriculum_training_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                           optimizer: optim.Optimizer, epoch: int, total_epochs: int,
                           config: CurriculumConfig, device: str = "cuda") -> Dict[str, float]:
    """
    Perform one curriculum training step.
    
    Args:
        model: Model to train
        x: Input images
        y: Target labels
        optimizer: Optimizer
        epoch: Current epoch
        total_epochs: Total number of epochs
        config: Curriculum configuration
        device: Device to run training on
        
    Returns:
        Dictionary with loss values
    """
    trainer = create_curriculum_trainer(model, config, device)
    return trainer.training_step(x, y, optimizer, epoch, total_epochs)
