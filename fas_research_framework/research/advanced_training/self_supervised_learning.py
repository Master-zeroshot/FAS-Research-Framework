"""
Self-Supervised Learning Implementation for Face Anti-Spoofing

This module implements self-supervised learning techniques to learn representations
from unlabeled data for face anti-spoofing tasks.
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


class ContrastiveLoss(nn.Module):
    """Contrastive loss for self-supervised learning."""
    
    def __init__(self, temperature: float = 0.07, margin: float = 1.0):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature for similarity computation
            margin: Margin for negative pairs
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            features: Feature representations
            labels: Labels for positive/negative pairs
            
        Returns:
            Contrastive loss
        """
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(mask.size(0), device=mask.device)
        
        # Compute positive and negative similarities
        pos_sim = similarity_matrix * mask
        neg_sim = similarity_matrix * (1 - mask)
        
        # Compute loss
        pos_loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim) + 1e-8))
        pos_loss = pos_loss * mask
        pos_loss = pos_loss.sum() / (mask.sum() + 1e-8)
        
        return pos_loss


class SimCLRLoss(nn.Module):
    """SimCLR contrastive loss."""
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize SimCLR loss.
        
        Args:
            temperature: Temperature for similarity computation
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        Compute SimCLR loss.
        
        Args:
            features1: Features from first augmentation
            features2: Features from second augmentation
            
        Returns:
            SimCLR loss
        """
        # Normalize features
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        
        # Concatenate features
        features = torch.cat([features1, features2], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create labels for positive pairs
        batch_size = features1.size(0)
        labels = torch.arange(batch_size, device=features.device)
        labels = torch.cat([labels, labels], dim=0)
        
        # Create mask for positive pairs
        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
        
        # Remove diagonal
        mask = mask - torch.eye(mask.size(0), device=mask.device)
        
        # Compute loss
        exp_sim = torch.exp(similarity_matrix)
        pos_sim = exp_sim * mask
        neg_sim = exp_sim * (1 - mask)
        
        loss = -torch.log(pos_sim.sum(1) / (pos_sim.sum(1) + neg_sim.sum(1) + 1e-8))
        loss = loss.mean()
        
        return loss


class BYOLLoss(nn.Module):
    """BYOL (Bootstrap Your Own Latent) loss."""
    
    def __init__(self, momentum: float = 0.999):
        """
        Initialize BYOL loss.
        
        Args:
            momentum: Momentum for target network updates
        """
        super().__init__()
        self.momentum = momentum
    
    def forward(self, online_features: torch.Tensor, 
                target_features: torch.Tensor) -> torch.Tensor:
        """
        Compute BYOL loss.
        
        Args:
            online_features: Features from online network
            target_features: Features from target network
            
        Returns:
            BYOL loss
        """
        # Normalize features
        online_features = F.normalize(online_features, dim=1)
        target_features = F.normalize(target_features, dim=1)
        
        # Compute similarity
        similarity = torch.sum(online_features * target_features, dim=1)
        loss = 2 - 2 * similarity.mean()
        
        return loss


class DataAugmentation:
    """Data augmentation for self-supervised learning."""
    
    def __init__(self, image_size: int = 224):
        """
        Initialize data augmentation.
        
        Args:
            image_size: Size of input images
        """
        self.image_size = image_size
    
    def random_crop(self, x: torch.Tensor, scale: Tuple[float, float] = (0.8, 1.0)) -> torch.Tensor:
        """Random crop augmentation."""
        batch_size, channels, height, width = x.shape
        
        # Random scale
        scale = random.uniform(scale[0], scale[1])
        new_height = int(height * scale)
        new_width = int(width * scale)
        
        # Random crop
        top = random.randint(0, height - new_height)
        left = random.randint(0, width - new_width)
        
        cropped = x[:, :, top:top+new_height, left:left+new_width]
        
        # Resize to original size
        cropped = F.interpolate(cropped, size=(height, width), mode='bilinear', align_corners=False)
        
        return cropped
    
    def color_jitter(self, x: torch.Tensor, brightness: float = 0.4, 
                    contrast: float = 0.4, saturation: float = 0.4, 
                    hue: float = 0.1) -> torch.Tensor:
        """Color jitter augmentation."""
        # Brightness
        if random.random() < 0.5:
            brightness_factor = random.uniform(1 - brightness, 1 + brightness)
            x = x * brightness_factor
        
        # Contrast
        if random.random() < 0.5:
            contrast_factor = random.uniform(1 - contrast, 1 + contrast)
            x = torch.clamp(x * contrast_factor, 0, 1)
        
        # Saturation
        if random.random() < 0.5:
            saturation_factor = random.uniform(1 - saturation, 1 + saturation)
            x = torch.clamp(x * saturation_factor, 0, 1)
        
        # Hue
        if random.random() < 0.5:
            hue_factor = random.uniform(-hue, hue)
            x = torch.clamp(x + hue_factor, 0, 1)
        
        return x
    
    def gaussian_blur(self, x: torch.Tensor, sigma: Tuple[float, float] = (0.1, 2.0)) -> torch.Tensor:
        """Gaussian blur augmentation."""
        if random.random() < 0.5:
            sigma = random.uniform(sigma[0], sigma[1])
            kernel_size = int(2 * np.ceil(2 * sigma) + 1)
            
            # Create Gaussian kernel
            kernel = self._gaussian_kernel(kernel_size, sigma)
            kernel = kernel.expand(x.size(1), 1, kernel_size, kernel_size)
            
            # Apply blur
            x = F.conv2d(x, kernel, padding=kernel_size//2, groups=x.size(1))
        
        return x
    
    def _gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Create Gaussian kernel."""
        x = torch.arange(kernel_size, dtype=torch.float32)
        x = x - kernel_size // 2
        x = x ** 2
        x = x / (2 * sigma ** 2)
        x = torch.exp(-x)
        x = x / x.sum()
        
        return x.view(1, 1, kernel_size, kernel_size)
    
    def augment(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random augmentations."""
        # First augmentation
        x1 = x.clone()
        x1 = self.random_crop(x1)
        x1 = self.color_jitter(x1)
        x1 = self.gaussian_blur(x1)
        
        # Second augmentation
        x2 = x.clone()
        x2 = self.random_crop(x2)
        x2 = self.color_jitter(x2)
        x2 = self.gaussian_blur(x2)
        
        return x1, x2


class SelfSupervisedTrainer:
    """Self-supervised learning trainer."""
    
    def __init__(self, model: nn.Module, device: str = "cuda", 
                 method: str = "simclr", temperature: float = 0.07):
        """
        Initialize self-supervised trainer.
        
        Args:
            model: Model to train
            device: Device to run training on
            method: Self-supervised learning method ('simclr', 'byol', 'contrastive')
            temperature: Temperature for contrastive learning
        """
        self.model = model
        self.device = device
        self.method = method
        
        # Initialize loss function
        if method == "simclr":
            self.loss_fn = SimCLRLoss(temperature)
        elif method == "byol":
            self.loss_fn = BYOLLoss()
        elif method == "contrastive":
            self.loss_fn = ContrastiveLoss(temperature)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Initialize data augmentation
        self.augmentation = DataAugmentation()
        
        logger.info(f"Initialized self-supervised trainer with {method} method")
    
    def training_step(self, x: torch.Tensor, optimizer: optim.Optimizer) -> Dict[str, float]:
        """
        Perform one self-supervised training step.
        
        Args:
            x: Input images
            optimizer: Optimizer
            
        Returns:
            Dictionary with loss values
        """
        # Zero gradients
        optimizer.zero_grad()
        
        # Apply augmentations
        x1, x2 = self.augmentation.augment(x)
        
        # Forward pass
        features1 = self.model(x1)
        features2 = self.model(x2)
        
        # Compute loss
        if self.method == "simclr":
            loss = self.loss_fn(features1, features2)
        elif self.method == "byol":
            loss = self.loss_fn(features1, features2)
        elif self.method == "contrastive":
            # Create labels for contrastive learning
            batch_size = x.size(0)
            labels = torch.arange(batch_size, device=x.device)
            labels = torch.cat([labels, labels], dim=0)
            features = torch.cat([features1, features2], dim=0)
            loss = self.loss_fn(features, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return {
            'ssl_loss': loss.item(),
            'method': self.method
        }
    
    def evaluate_representations(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate learned representations.
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                
                # Extract features
                features = self.model(x)
                features = F.normalize(features, dim=1)
                
                all_features.append(features.cpu())
                all_labels.append(y.cpu())
        
        # Concatenate all features and labels
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute representation quality metrics
        metrics = self._compute_representation_metrics(all_features, all_labels)
        
        return metrics
    
    def _compute_representation_metrics(self, features: torch.Tensor, 
                                     labels: torch.Tensor) -> Dict[str, float]:
        """Compute representation quality metrics."""
        # Compute intra-class and inter-class distances
        unique_labels = torch.unique(labels)
        intra_class_distances = []
        inter_class_distances = []
        
        for label in unique_labels:
            # Intra-class distances
            class_features = features[labels == label]
            if class_features.size(0) > 1:
                pairwise_distances = torch.pdist(class_features)
                intra_class_distances.append(pairwise_distances.mean().item())
            
            # Inter-class distances
            other_features = features[labels != label]
            if other_features.size(0) > 0:
                inter_distances = torch.cdist(class_features, other_features)
                inter_class_distances.append(inter_distances.mean().item())
        
        # Compute metrics
        avg_intra_class = np.mean(intra_class_distances) if intra_class_distances else 0
        avg_inter_class = np.mean(inter_class_distances) if inter_class_distances else 0
        
        # Separation ratio
        separation_ratio = avg_inter_class / (avg_intra_class + 1e-8)
        
        return {
            'intra_class_distance': avg_intra_class,
            'inter_class_distance': avg_inter_class,
            'separation_ratio': separation_ratio
        }


class SelfSupervisedConfig:
    """Configuration for self-supervised learning."""
    
    def __init__(self, method: str = "simclr", temperature: float = 0.07,
                 learning_rate: float = 0.001, weight_decay: float = 1e-4,
                 epochs: int = 100, warmup_epochs: int = 10):
        """
        Initialize self-supervised configuration.
        
        Args:
            method: Self-supervised learning method
            temperature: Temperature for contrastive learning
            learning_rate: Learning rate
            weight_decay: Weight decay
            epochs: Number of training epochs
            warmup_epochs: Number of warmup epochs
        """
        self.method = method
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs


def create_self_supervised_trainer(model: nn.Module, config: SelfSupervisedConfig,
                                 device: str = "cuda") -> SelfSupervisedTrainer:
    """
    Create self-supervised trainer.
    
    Args:
        model: Model to train
        config: Self-supervised configuration
        device: Device to run training on
        
    Returns:
        Self-supervised trainer instance
    """
    return SelfSupervisedTrainer(
        model=model,
        device=device,
        method=config.method,
        temperature=config.temperature
    )


def self_supervised_training_step(model: nn.Module, x: torch.Tensor,
                                optimizer: optim.Optimizer, config: SelfSupervisedConfig,
                                device: str = "cuda") -> Dict[str, float]:
    """
    Perform one self-supervised training step.
    
    Args:
        model: Model to train
        x: Input images
        optimizer: Optimizer
        config: Self-supervised configuration
        device: Device to run training on
        
    Returns:
        Dictionary with loss values
    """
    trainer = create_self_supervised_trainer(model, config, device)
    return trainer.training_step(x, optimizer)
