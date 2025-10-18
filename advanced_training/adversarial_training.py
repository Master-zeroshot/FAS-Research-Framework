"""
Adversarial Training Implementation for Face Anti-Spoofing

This module implements adversarial training techniques to improve model robustness
against adversarial attacks and improve generalization on face anti-spoofing tasks.
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


class AdversarialAttack(ABC):
    """Base class for adversarial attacks."""
    
    def __init__(self, model: nn.Module, epsilon: float = 0.03, 
                 alpha: float = 0.01, num_steps: int = 10):
        """
        Initialize adversarial attack.
        
        Args:
            model: Target model to attack
            epsilon: Maximum perturbation magnitude
            alpha: Step size for iterative attacks
            num_steps: Number of attack steps
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
    
    @abstractmethod
    def generate_perturbation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate adversarial perturbation."""
        pass


class FGSMAttack(AdversarialAttack):
    """Fast Gradient Sign Method (FGSM) attack."""
    
    def __init__(self, model: nn.Module, epsilon: float = 0.03):
        super().__init__(model, epsilon)
    
    def generate_perturbation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate FGSM perturbation."""
        x.requires_grad_(True)
        
        # Forward pass
        output = self.model(x)
        loss = F.cross_entropy(output, y)
        
        # Compute gradients
        grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
        
        # Generate perturbation
        perturbation = self.epsilon * grad.sign()
        
        return perturbation


class PGDAttack(AdversarialAttack):
    """Projected Gradient Descent (PGD) attack."""
    
    def __init__(self, model: nn.Module, epsilon: float = 0.03, 
                 alpha: float = 0.01, num_steps: int = 10):
        super().__init__(model, epsilon, alpha, num_steps)
    
    def generate_perturbation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate PGD perturbation."""
        # Initialize perturbation
        perturbation = torch.zeros_like(x)
        
        for _ in range(self.num_steps):
            perturbation.requires_grad_(True)
            
            # Forward pass with perturbation
            x_adv = x + perturbation
            output = self.model(x_adv)
            loss = F.cross_entropy(output, y)
            
            # Compute gradients
            grad = torch.autograd.grad(loss, perturbation, retain_graph=False, create_graph=False)[0]
            
            # Update perturbation
            perturbation = perturbation + self.alpha * grad.sign()
            perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
        
        return perturbation


class CWAttack(AdversarialAttack):
    """Carlini & Wagner (C&W) attack."""
    
    def __init__(self, model: nn.Module, epsilon: float = 0.03, 
                 confidence: float = 0.0, num_steps: int = 10):
        super().__init__(model, epsilon)
        self.confidence = confidence
        self.num_steps = num_steps
    
    def generate_perturbation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate C&W perturbation."""
        # Initialize perturbation
        perturbation = torch.zeros_like(x, requires_grad=True)
        optimizer = optim.Adam([perturbation], lr=0.01)
        
        for _ in range(self.num_steps):
            # Forward pass
            x_adv = x + perturbation
            output = self.model(x_adv)
            
            # C&W loss
            target_logits = output.gather(1, y.unsqueeze(1))
            max_logits = output.gather(1, (output - torch.eye(output.size(1), device=output.device)[y] * 1000).argmax(1).unsqueeze(1))
            
            loss = torch.clamp(max_logits - target_logits + self.confidence, min=0.0).mean()
            
            # Update perturbation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Project to epsilon ball
            perturbation.data = torch.clamp(perturbation.data, -self.epsilon, self.epsilon)
        
        return perturbation.detach()


class AdversarialTrainer:
    """Adversarial training implementation for face anti-spoofing."""
    
    def __init__(self, model: nn.Module, device: str = "cuda", 
                 attack_type: str = "fgsm", epsilon: float = 0.03,
                 alpha: float = 0.01, num_steps: int = 10):
        """
        Initialize adversarial trainer.
        
        Args:
            model: Model to train
            device: Device to run training on
            attack_type: Type of adversarial attack ('fgsm', 'pgd', 'cw')
            epsilon: Maximum perturbation magnitude
            alpha: Step size for iterative attacks
            num_steps: Number of attack steps
        """
        self.model = model
        self.device = device
        
        # Initialize attack
        if attack_type == "fgsm":
            self.attack = FGSMAttack(model, epsilon)
        elif attack_type == "pgd":
            self.attack = PGDAttack(model, epsilon, alpha, num_steps)
        elif attack_type == "cw":
            self.attack = CWAttack(model, epsilon, alpha, num_steps)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        self.attack_type = attack_type
        self.epsilon = epsilon
        
        logger.info(f"Initialized adversarial trainer with {attack_type} attack")
    
    def adversarial_loss(self, x: torch.Tensor, y: torch.Tensor, 
                        lambda_adv: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adversarial training loss.
        
        Args:
            x: Input images
            y: Target labels
            lambda_adv: Weight for adversarial loss
            
        Returns:
            Tuple of (total_loss, adversarial_loss)
        """
        # Standard forward pass
        x_clean = x.clone()
        output_clean = self.model(x_clean)
        loss_clean = F.cross_entropy(output_clean, y)
        
        # Generate adversarial examples
        with torch.no_grad():
            perturbation = self.attack.generate_perturbation(x, y)
        
        x_adv = x + perturbation
        x_adv = torch.clamp(x_adv, 0, 1)  # Ensure valid pixel values
        
        # Adversarial forward pass
        output_adv = self.model(x_adv)
        loss_adv = F.cross_entropy(output_adv, y)
        
        # Total loss
        total_loss = loss_clean + lambda_adv * loss_adv
        
        return total_loss, loss_adv
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor, 
                   optimizer: optim.Optimizer, lambda_adv: float = 0.5) -> Dict[str, float]:
        """
        Perform one adversarial training step.
        
        Args:
            x: Input images
            y: Target labels
            optimizer: Optimizer
            lambda_adv: Weight for adversarial loss
            
        Returns:
            Dictionary with loss values
        """
        self.model.train()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Compute adversarial loss
        total_loss, adv_loss = self.adversarial_loss(x, y, lambda_adv)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'adversarial_loss': adv_loss.item(),
            'lambda_adv': lambda_adv
        }
    
    def evaluate_robustness(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate model robustness against adversarial attacks.
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            Dictionary with robustness metrics
        """
        self.model.eval()
        
        total_samples = 0
        clean_correct = 0
        adv_correct = 0
        
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            
            # Clean accuracy
            with torch.no_grad():
                output_clean = self.model(x)
                pred_clean = output_clean.argmax(dim=1)
                clean_correct += (pred_clean == y).sum().item()
            
            # Adversarial accuracy
            with torch.no_grad():
                perturbation = self.attack.generate_perturbation(x, y)
                x_adv = x + perturbation
                x_adv = torch.clamp(x_adv, 0, 1)
                
                output_adv = self.model(x_adv)
                pred_adv = output_adv.argmax(dim=1)
                adv_correct += (pred_adv == y).sum().item()
            
            total_samples += x.size(0)
        
        clean_accuracy = clean_correct / total_samples
        adv_accuracy = adv_correct / total_samples
        robustness = adv_accuracy / clean_accuracy if clean_accuracy > 0 else 0
        
        return {
            'clean_accuracy': clean_accuracy,
            'adversarial_accuracy': adv_accuracy,
            'robustness_ratio': robustness
        }


class AdversarialTrainingConfig:
    """Configuration for adversarial training."""
    
    def __init__(self, attack_type: str = "fgsm", epsilon: float = 0.03,
                 alpha: float = 0.01, num_steps: int = 10, lambda_adv: float = 0.5,
                 warmup_epochs: int = 5, schedule_lambda: bool = True):
        """
        Initialize adversarial training configuration.
        
        Args:
            attack_type: Type of adversarial attack
            epsilon: Maximum perturbation magnitude
            alpha: Step size for iterative attacks
            num_steps: Number of attack steps
            lambda_adv: Weight for adversarial loss
            warmup_epochs: Number of warmup epochs
            schedule_lambda: Whether to schedule lambda_adv
        """
        self.attack_type = attack_type
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.lambda_adv = lambda_adv
        self.warmup_epochs = warmup_epochs
        self.schedule_lambda = schedule_lambda
    
    def get_lambda_adv(self, epoch: int, total_epochs: int) -> float:
        """Get lambda_adv for current epoch."""
        if not self.schedule_lambda:
            return self.lambda_adv
        
        # Linear warmup
        if epoch < self.warmup_epochs:
            return self.lambda_adv * (epoch / self.warmup_epochs)
        
        # Cosine annealing
        progress = (epoch - self.warmup_epochs) / (total_epochs - self.warmup_epochs)
        return self.lambda_adv * (1 + np.cos(np.pi * progress)) / 2


class AdversarialTrainingScheduler:
    """Scheduler for adversarial training parameters."""
    
    def __init__(self, config: AdversarialTrainingConfig):
        self.config = config
    
    def get_epsilon(self, epoch: int, total_epochs: int) -> float:
        """Get epsilon for current epoch."""
        # Gradually increase epsilon
        progress = epoch / total_epochs
        return self.config.epsilon * (0.5 + 0.5 * progress)
    
    def get_alpha(self, epoch: int, total_epochs: int) -> float:
        """Get alpha for current epoch."""
        # Adjust alpha based on epsilon
        epsilon = self.get_epsilon(epoch, total_epochs)
        return min(self.config.alpha, epsilon / self.config.num_steps)
    
    def get_lambda_adv(self, epoch: int, total_epochs: int) -> float:
        """Get lambda_adv for current epoch."""
        return self.config.get_lambda_adv(epoch, total_epochs)


def create_adversarial_trainer(model: nn.Module, config: AdversarialTrainingConfig,
                             device: str = "cuda") -> AdversarialTrainer:
    """
    Create adversarial trainer with given configuration.
    
    Args:
        model: Model to train
        config: Adversarial training configuration
        device: Device to run training on
        
    Returns:
        Adversarial trainer instance
    """
    return AdversarialTrainer(
        model=model,
        device=device,
        attack_type=config.attack_type,
        epsilon=config.epsilon,
        alpha=config.alpha,
        num_steps=config.num_steps
    )


def adversarial_training_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                            optimizer: optim.Optimizer, config: AdversarialTrainingConfig,
                            device: str = "cuda") -> Dict[str, float]:
    """
    Perform one adversarial training step.
    
    Args:
        model: Model to train
        x: Input images
        y: Target labels
        optimizer: Optimizer
        config: Adversarial training configuration
        device: Device to run training on
        
    Returns:
        Dictionary with loss values
    """
    trainer = create_adversarial_trainer(model, config, device)
    return trainer.train_step(x, y, optimizer, config.lambda_adv)


def evaluate_adversarial_robustness(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                                  attack_type: str = "fgsm", epsilon: float = 0.03,
                                  device: str = "cuda") -> Dict[str, float]:
    """
    Evaluate model robustness against adversarial attacks.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader for evaluation
        attack_type: Type of adversarial attack
        epsilon: Maximum perturbation magnitude
        device: Device to run evaluation on
        
    Returns:
        Dictionary with robustness metrics
    """
    config = AdversarialTrainingConfig(attack_type=attack_type, epsilon=epsilon)
    trainer = create_adversarial_trainer(model, config, device)
    return trainer.evaluate_robustness(dataloader)
