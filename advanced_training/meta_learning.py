"""
Meta-Learning Implementation for Face Anti-Spoofing

This module implements meta-learning techniques to enable models to quickly
adapt to new tasks and domains for face anti-spoofing applications.
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


class MAML(nn.Module):
    """Model-Agnostic Meta-Learning (MAML) implementation."""
    
    def __init__(self, model: nn.Module, inner_lr: float = 0.01, 
                 meta_lr: float = 0.001, num_inner_steps: int = 5):
        """
        Initialize MAML.
        
        Args:
            model: Base model
            inner_lr: Learning rate for inner loop
            meta_lr: Learning rate for meta loop
            num_inner_steps: Number of inner loop steps
        """
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        
        # Initialize meta-optimizer
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)
        
        logger.info("Initialized MAML meta-learner")
    
    def inner_loop(self, support_x: torch.Tensor, support_y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform inner loop adaptation.
        
        Args:
            support_x: Support set inputs
            support_y: Support set labels
            
        Returns:
            Dictionary with adapted parameters
        """
        # Clone model parameters
        adapted_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Inner loop optimization
        for step in range(self.num_inner_steps):
            # Forward pass with adapted parameters
            output = self._forward_with_params(support_x, adapted_params)
            loss = F.cross_entropy(output, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, adapted_params.values(), create_graph=True)
            
            # Update parameters
            for (name, param), grad in zip(adapted_params.items(), grads):
                adapted_params[name] = param - self.inner_lr * grad
        
        return adapted_params
    
    def _forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with given parameters."""
        # This is a simplified version - in practice, you'd need to implement
        # a more sophisticated parameter replacement mechanism
        return self.model(x)
    
    def meta_step(self, support_x: torch.Tensor, support_y: torch.Tensor,
                  query_x: torch.Tensor, query_y: torch.Tensor) -> Dict[str, float]:
        """
        Perform one meta-learning step.
        
        Args:
            support_x: Support set inputs
            support_y: Support set labels
            query_x: Query set inputs
            query_y: Query set labels
            
        Returns:
            Dictionary with loss values
        """
        # Inner loop adaptation
        adapted_params = self.inner_loop(support_x, support_y)
        
        # Query set evaluation
        query_output = self._forward_with_params(query_x, adapted_params)
        meta_loss = F.cross_entropy(query_output, query_y)
        
        # Meta-optimization
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return {
            'meta_loss': meta_loss.item(),
            'inner_steps': self.num_inner_steps
        }


class PrototypicalNetworks(nn.Module):
    """Prototypical Networks for few-shot learning."""
    
    def __init__(self, model: nn.Module, embedding_dim: int = 128):
        """
        Initialize Prototypical Networks.
        
        Args:
            model: Feature extractor model
            embedding_dim: Dimension of embeddings
        """
        super().__init__()
        self.model = model
        self.embedding_dim = embedding_dim
        
        logger.info("Initialized Prototypical Networks")
    
    def compute_prototypes(self, support_x: torch.Tensor, support_y: torch.Tensor) -> torch.Tensor:
        """
        Compute class prototypes from support set.
        
        Args:
            support_x: Support set inputs
            support_y: Support set labels
            
        Returns:
            Class prototypes
        """
        # Extract features
        support_features = self.model(support_x)
        support_features = F.normalize(support_features, dim=1)
        
        # Compute prototypes
        unique_labels = torch.unique(support_y)
        prototypes = []
        
        for label in unique_labels:
            class_features = support_features[support_y == label]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes, dim=0)
        return prototypes
    
    def forward(self, support_x: torch.Tensor, support_y: torch.Tensor,
                query_x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for few-shot classification.
        
        Args:
            support_x: Support set inputs
            support_y: Support set labels
            query_x: Query set inputs
            
        Returns:
            Query set predictions
        """
        # Compute prototypes
        prototypes = self.compute_prototypes(support_x, support_y)
        
        # Extract query features
        query_features = self.model(query_x)
        query_features = F.normalize(query_features, dim=1)
        
        # Compute distances to prototypes
        distances = torch.cdist(query_features, prototypes)
        
        # Convert distances to probabilities
        logits = -distances
        return logits
    
    def training_step(self, support_x: torch.Tensor, support_y: torch.Tensor,
                     query_x: torch.Tensor, query_y: torch.Tensor,
                     optimizer: optim.Optimizer) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            support_x: Support set inputs
            support_y: Support set labels
            query_x: Query set inputs
            query_y: Query set labels
            optimizer: Optimizer
            
        Returns:
            Dictionary with loss values
        """
        # Forward pass
        logits = self.forward(support_x, support_y, query_x)
        loss = F.cross_entropy(logits, query_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            accuracy = (pred == query_y).float().mean().item()
        
        return {
            'prototypical_loss': loss.item(),
            'prototypical_accuracy': accuracy
        }


class RelationNetworks(nn.Module):
    """Relation Networks for few-shot learning."""
    
    def __init__(self, model: nn.Module, relation_dim: int = 128):
        """
        Initialize Relation Networks.
        
        Args:
            model: Feature extractor model
            relation_dim: Dimension of relation features
        """
        super().__init__()
        self.model = model
        self.relation_dim = relation_dim
        
        # Relation network
        self.relation_net = nn.Sequential(
            nn.Linear(relation_dim * 2, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, 1),
            nn.Sigmoid()
        )
        
        logger.info("Initialized Relation Networks")
    
    def forward(self, support_x: torch.Tensor, support_y: torch.Tensor,
                query_x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for few-shot classification.
        
        Args:
            support_x: Support set inputs
            support_y: Support set labels
            query_x: Query set inputs
            
        Returns:
            Query set predictions
        """
        # Extract features
        support_features = self.model(support_x)
        query_features = self.model(query_x)
        
        # Compute class prototypes
        unique_labels = torch.unique(support_y)
        prototypes = []
        
        for label in unique_labels:
            class_features = support_features[support_y == label]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes, dim=0)
        
        # Compute relations
        batch_size = query_x.size(0)
        num_classes = prototypes.size(0)
        
        query_features_expanded = query_features.unsqueeze(1).expand(-1, num_classes, -1)
        prototypes_expanded = prototypes.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate features
        relation_features = torch.cat([query_features_expanded, prototypes_expanded], dim=2)
        relation_features = relation_features.view(-1, relation_features.size(-1))
        
        # Compute relations
        relations = self.relation_net(relation_features)
        relations = relations.view(batch_size, num_classes)
        
        return relations
    
    def training_step(self, support_x: torch.Tensor, support_y: torch.Tensor,
                     query_x: torch.Tensor, query_y: torch.Tensor,
                     optimizer: optim.Optimizer) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            support_x: Support set inputs
            support_y: Support set labels
            query_x: Query set inputs
            query_y: Query set labels
            optimizer: Optimizer
            
        Returns:
            Dictionary with loss values
        """
        # Forward pass
        relations = self.forward(support_x, support_y, query_x)
        loss = F.cross_entropy(relations, query_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            pred = relations.argmax(dim=1)
            accuracy = (pred == query_y).float().mean().item()
        
        return {
            'relation_loss': loss.item(),
            'relation_accuracy': accuracy
        }


class MetaLearningTrainer:
    """Meta-learning trainer for face anti-spoofing."""
    
    def __init__(self, model: nn.Module, method: str = "maml", 
                 device: str = "cuda", **kwargs):
        """
        Initialize meta-learning trainer.
        
        Args:
            model: Base model
            method: Meta-learning method ('maml', 'prototypical', 'relation')
            device: Device to run training on
            **kwargs: Additional arguments for specific methods
        """
        self.model = model
        self.device = device
        self.method = method
        
        # Initialize meta-learner
        if method == "maml":
            self.meta_learner = MAML(model, **kwargs)
        elif method == "prototypical":
            self.meta_learner = PrototypicalNetworks(model, **kwargs)
        elif method == "relation":
            self.meta_learner = RelationNetworks(model, **kwargs)
        else:
            raise ValueError(f"Unknown meta-learning method: {method}")
        
        logger.info(f"Initialized meta-learning trainer with {method} method")
    
    def create_episode(self, x: torch.Tensor, y: torch.Tensor, 
                      num_support: int = 5, num_query: int = 15) -> Dict[str, torch.Tensor]:
        """
        Create few-shot learning episode.
        
        Args:
            x: Input images
            y: Target labels
            num_support: Number of support examples per class
            num_query: Number of query examples per class
            
        Returns:
            Dictionary with episode data
        """
        unique_labels = torch.unique(y)
        support_x, support_y = [], []
        query_x, query_y = [], []
        
        for label in unique_labels:
            # Get examples for this class
            class_indices = (y == label).nonzero().squeeze(1)
            class_examples = x[class_indices]
            class_labels = y[class_indices]
            
            # Shuffle examples
            perm = torch.randperm(len(class_examples))
            class_examples = class_examples[perm]
            class_labels = class_labels[perm]
            
            # Split into support and query
            support_x.append(class_examples[:num_support])
            support_y.append(class_labels[:num_support])
            query_x.append(class_examples[num_support:num_support+num_query])
            query_y.append(class_labels[num_support:num_support+num_query])
        
        # Concatenate all examples
        support_x = torch.cat(support_x, dim=0)
        support_y = torch.cat(support_y, dim=0)
        query_x = torch.cat(query_x, dim=0)
        query_y = torch.cat(query_y, dim=0)
        
        return {
            'support_x': support_x,
            'support_y': support_y,
            'query_x': query_x,
            'query_y': query_y
        }
    
    def training_step(self, x: torch.Tensor, y: torch.Tensor,
                     optimizer: optim.Optimizer, num_support: int = 5,
                     num_query: int = 15) -> Dict[str, float]:
        """
        Perform one meta-learning training step.
        
        Args:
            x: Input images
            y: Target labels
            optimizer: Optimizer
            num_support: Number of support examples per class
            num_query: Number of query examples per class
            
        Returns:
            Dictionary with loss values
        """
        # Create episode
        episode = self.create_episode(x, y, num_support, num_query)
        
        # Move to device
        support_x = episode['support_x'].to(self.device)
        support_y = episode['support_y'].to(self.device)
        query_x = episode['query_x'].to(self.device)
        query_y = episode['query_y'].to(self.device)
        
        # Perform meta-learning step
        if self.method == "maml":
            return self.meta_learner.meta_step(support_x, support_y, query_x, query_y)
        else:
            return self.meta_learner.training_step(support_x, support_y, query_x, query_y, optimizer)
    
    def evaluate_few_shot(self, dataloader: torch.utils.data.DataLoader,
                         num_support: int = 5, num_query: int = 15,
                         num_episodes: int = 100) -> Dict[str, float]:
        """
        Evaluate few-shot learning performance.
        
        Args:
            dataloader: Data loader for evaluation
            num_support: Number of support examples per class
            num_query: Number of query examples per class
            num_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        all_accuracies = []
        
        with torch.no_grad():
            for episode in range(num_episodes):
                # Get random batch
                x, y = next(iter(dataloader))
                x, y = x.to(self.device), y.to(self.device)
                
                # Create episode
                episode_data = self.create_episode(x, y, num_support, num_query)
                
                support_x = episode_data['support_x'].to(self.device)
                support_y = episode_data['support_y'].to(self.device)
                query_x = episode_data['query_x'].to(self.device)
                query_y = episode_data['query_y'].to(self.device)
                
                # Evaluate
                if self.method == "maml":
                    # MAML evaluation
                    adapted_params = self.meta_learner.inner_loop(support_x, support_y)
                    query_output = self.meta_learner._forward_with_params(query_x, adapted_params)
                else:
                    # Prototypical/Relation Networks evaluation
                    query_output = self.meta_learner.forward(support_x, support_y, query_x)
                
                # Compute accuracy
                pred = query_output.argmax(dim=1)
                accuracy = (pred == query_y).float().mean().item()
                all_accuracies.append(accuracy)
        
        mean_accuracy = np.mean(all_accuracies)
        std_accuracy = np.std(all_accuracies)
        
        return {
            'few_shot_accuracy': mean_accuracy,
            'few_shot_std': std_accuracy,
            'num_episodes': num_episodes,
            'num_support': num_support,
            'num_query': num_query
        }


class MetaLearningConfig:
    """Configuration for meta-learning."""
    
    def __init__(self, method: str = "maml", inner_lr: float = 0.01,
                 meta_lr: float = 0.001, num_inner_steps: int = 5,
                 num_support: int = 5, num_query: int = 15):
        """
        Initialize meta-learning configuration.
        
        Args:
            method: Meta-learning method
            inner_lr: Learning rate for inner loop
            meta_lr: Learning rate for meta loop
            num_inner_steps: Number of inner loop steps
            num_support: Number of support examples per class
            num_query: Number of query examples per class
        """
        self.method = method
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        self.num_support = num_support
        self.num_query = num_query


def create_meta_learning_trainer(model: nn.Module, config: MetaLearningConfig,
                               device: str = "cuda") -> MetaLearningTrainer:
    """
    Create meta-learning trainer.
    
    Args:
        model: Model to train
        config: Meta-learning configuration
        device: Device to run training on
        
    Returns:
        Meta-learning trainer instance
    """
    return MetaLearningTrainer(
        model=model,
        method=config.method,
        device=device,
        inner_lr=config.inner_lr,
        meta_lr=config.meta_lr,
        num_inner_steps=config.num_inner_steps
    )


def meta_learning_training_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                              optimizer: optim.Optimizer, config: MetaLearningConfig,
                              device: str = "cuda") -> Dict[str, float]:
    """
    Perform one meta-learning training step.
    
    Args:
        model: Model to train
        x: Input images
        y: Target labels
        optimizer: Optimizer
        config: Meta-learning configuration
        device: Device to run training on
        
    Returns:
        Dictionary with loss values
    """
    trainer = create_meta_learning_trainer(model, config, device)
    return trainer.training_step(x, y, optimizer, config.num_support, config.num_query)
