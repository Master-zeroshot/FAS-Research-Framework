"""
Feature Analysis for Face Anti-Spoofing

This module provides tools for analyzing and visualizing features
learned by face anti-spoofing models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import json

from utils.logging_config import get_logger

logger = get_logger(__name__)


class FeatureAnalyzer:
    """Analyzer for model features in face anti-spoofing."""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Initialize feature analyzer.
        
        Args:
            model: Model to analyze
            device: Device to run analysis on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        # Hook for capturing features
        self.features = {}
        self.hooks = []
        
        logger.info("Initialized feature analyzer")
    
    def register_feature_hooks(self, layer_names: List[str] = None) -> None:
        """
        Register hooks for capturing features.
        
        Args:
            layer_names: List of layer names to hook
        """
        if layer_names is None:
            # Default to common feature layer names
            layer_names = ['features', 'classifier', 'fc', 'linear']
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self.features[name] = output.detach().cpu()
            return hook
        
        # Register hooks for specified layers
        for name, module in self.model.named_modules():
            if any(layer_name in name.lower() for layer_name in layer_names):
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
                logger.info(f"Registered hook for layer: {name}")
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.features = {}
    
    def extract_features(self, dataloader, max_samples: int = 1000) -> Dict[str, Any]:
        """
        Extract features from dataset.
        
        Args:
            dataloader: Data loader for feature extraction
            max_samples: Maximum number of samples to process
            
        Returns:
            Dictionary with extracted features and labels
        """
        all_features = {}
        all_labels = []
        all_metadata = []
        
        sample_count = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if sample_count >= max_samples:
                    break
                
                if len(batch) == 3:
                    images, labels, metadata = batch
                else:
                    images, labels = batch
                    metadata = None
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                _ = self.model(images)
                
                # Store features
                for layer_name, features in self.features.items():
                    if layer_name not in all_features:
                        all_features[layer_name] = []
                    all_features[layer_name].append(features.cpu())
                
                # Store labels and metadata
                all_labels.extend(labels.cpu().numpy())
                if metadata:
                    all_metadata.extend(metadata)
                
                sample_count += images.size(0)
        
        # Concatenate features
        for layer_name in all_features:
            all_features[layer_name] = torch.cat(all_features[layer_name], dim=0)
        
        return {
            'features': all_features,
            'labels': np.array(all_labels),
            'metadata': all_metadata,
            'num_samples': sample_count
        }
    
    def analyze_feature_distribution(self, features: torch.Tensor, labels: np.ndarray) -> Dict[str, Any]:
        """
        Analyze feature distribution.
        
        Args:
            features: Feature tensor
            labels: Corresponding labels
            
        Returns:
            Dictionary with distribution analysis
        """
        features_np = features.numpy()
        
        # Basic statistics
        stats = {
            'mean': np.mean(features_np, axis=0),
            'std': np.std(features_np, axis=0),
            'min': np.min(features_np, axis=0),
            'max': np.max(features_np, axis=0),
            'shape': features_np.shape
        }
        
        # Class-wise statistics
        unique_labels = np.unique(labels)
        class_stats = {}
        
        for label in unique_labels:
            mask = labels == label
            class_features = features_np[mask]
            
            class_stats[label] = {
                'mean': np.mean(class_features, axis=0),
                'std': np.std(class_features, axis=0),
                'count': np.sum(mask)
            }
        
        # Feature correlation
        correlation_matrix = np.corrcoef(features_np.T)
        
        return {
            'overall_stats': stats,
            'class_stats': class_stats,
            'correlation_matrix': correlation_matrix
        }
    
    def compute_feature_similarity(self, features: torch.Tensor, labels: np.ndarray) -> Dict[str, Any]:
        """
        Compute feature similarity metrics.
        
        Args:
            features: Feature tensor
            labels: Corresponding labels
            
        Returns:
            Dictionary with similarity metrics
        """
        features_np = features.numpy()
        
        # Compute pairwise distances
        from scipy.spatial.distance import pdist, squareform
        distances = pdist(features_np, metric='euclidean')
        distance_matrix = squareform(distances)
        
        # Intra-class and inter-class distances
        unique_labels = np.unique(labels)
        intra_class_distances = []
        inter_class_distances = []
        
        for i, label_i in enumerate(unique_labels):
            for j, label_j in enumerate(unique_labels):
                mask_i = labels == label_i
                mask_j = labels == label_j
                
                if i == j:
                    # Intra-class distances
                    class_distances = distance_matrix[mask_i][:, mask_i]
                    # Remove diagonal (self-distances)
                    class_distances = class_distances[np.triu_indices_from(class_distances, k=1)]
                    intra_class_distances.extend(class_distances)
                else:
                    # Inter-class distances
                    inter_distances = distance_matrix[mask_i][:, mask_j]
                    inter_class_distances.extend(inter_distances.flatten())
        
        # Compute statistics
        similarity_stats = {
            'intra_class_mean': np.mean(intra_class_distances),
            'intra_class_std': np.std(intra_class_distances),
            'inter_class_mean': np.mean(inter_class_distances),
            'inter_class_std': np.std(inter_class_distances),
            'separation_ratio': np.mean(inter_class_distances) / (np.mean(intra_class_distances) + 1e-8)
        }
        
        return similarity_stats


class FeatureVisualizer:
    """Visualizer for model features."""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Initialize feature visualizer.
        
        Args:
            model: Model to visualize
            device: Device to run visualization on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        logger.info("Initialized feature visualizer")
    
    def create_feature_tsne(self, features: torch.Tensor, labels: np.ndarray,
                           perplexity: int = 30, n_components: int = 2) -> plt.Figure:
        """
        Create t-SNE visualization of features.
        
        Args:
            features: Feature tensor
            labels: Corresponding labels
            perplexity: t-SNE perplexity parameter
            n_components: Number of components for t-SNE
            
        Returns:
            Matplotlib figure
        """
        features_np = features.numpy()
        
        # Apply t-SNE
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        features_tsne = tsne.fit_transform(features_np)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(features_tsne[mask, 0], features_tsne[mask, 1], 
                      c=[colors[i]], label=f'Class {label}', alpha=0.7)
        
        ax.set_title('t-SNE Visualization of Features')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def create_feature_pca(self, features: torch.Tensor, labels: np.ndarray,
                          n_components: int = 2) -> plt.Figure:
        """
        Create PCA visualization of features.
        
        Args:
            features: Feature tensor
            labels: Corresponding labels
            n_components: Number of PCA components
            
        Returns:
            Matplotlib figure
        """
        features_np = features.numpy()
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        features_pca = pca.fit_transform(features_np)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(features_pca[mask, 0], features_pca[mask, 1], 
                      c=[colors[i]], label=f'Class {label}', alpha=0.7)
        
        ax.set_title('PCA Visualization of Features')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def create_feature_heatmap(self, features: torch.Tensor, labels: np.ndarray,
                              max_features: int = 50) -> plt.Figure:
        """
        Create heatmap of feature values.
        
        Args:
            features: Feature tensor
            labels: Corresponding labels
            max_features: Maximum number of features to display
            
        Returns:
            Matplotlib figure
        """
        features_np = features.numpy()
        
        # Select subset of features if too many
        if features_np.shape[1] > max_features:
            # Select features with highest variance
            feature_vars = np.var(features_np, axis=0)
            top_features = np.argsort(feature_vars)[-max_features:]
            features_np = features_np[:, top_features]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Sort by labels for better visualization
        sort_indices = np.argsort(labels)
        features_sorted = features_np[sort_indices]
        labels_sorted = labels[sort_indices]
        
        # Create heatmap
        im = ax.imshow(features_sorted.T, aspect='auto', cmap='viridis')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Feature Value')
        
        # Add labels
        ax.set_title('Feature Heatmap')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Features')
        
        # Add class boundaries
        unique_labels = np.unique(labels_sorted)
        for label in unique_labels:
            boundary = np.where(labels_sorted == label)[0]
            if len(boundary) > 0:
                ax.axvline(x=boundary[0], color='red', linestyle='--', alpha=0.7)
                ax.axvline(x=boundary[-1], color='red', linestyle='--', alpha=0.7)
        
        return fig
    
    def create_feature_distribution(self, features: torch.Tensor, labels: np.ndarray,
                                  feature_indices: List[int] = None) -> plt.Figure:
        """
        Create distribution plots for features.
        
        Args:
            features: Feature tensor
            labels: Corresponding labels
            feature_indices: Indices of features to plot
            
        Returns:
            Matplotlib figure
        """
        features_np = features.numpy()
        
        if feature_indices is None:
            # Select features with highest variance
            feature_vars = np.var(features_np, axis=0)
            feature_indices = np.argsort(feature_vars)[-9:]  # Top 9 features
        
        # Create subplots
        n_features = len(feature_indices)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, feature_idx in enumerate(feature_indices):
            ax = axes[i]
            
            # Plot distribution for each class
            for j, label in enumerate(unique_labels):
                mask = labels == label
                feature_values = features_np[mask, feature_idx]
                
                ax.hist(feature_values, bins=30, alpha=0.7, 
                       color=colors[j], label=f'Class {label}', density=True)
            
            ax.set_title(f'Feature {feature_idx} Distribution')
            ax.set_xlabel('Feature Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig


class FeatureClustering:
    """Clustering analysis for features."""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Initialize feature clustering.
        
        Args:
            model: Model to analyze
            device: Device to run analysis on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        logger.info("Initialized feature clustering")
    
    def cluster_features(self, features: torch.Tensor, labels: np.ndarray,
                        n_clusters: int = 2) -> Dict[str, Any]:
        """
        Cluster features and analyze results.
        
        Args:
            features: Feature tensor
            labels: Corresponding labels
            n_clusters: Number of clusters
            
        Returns:
            Dictionary with clustering results
        """
        features_np = features.numpy()
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_np)
        
        # Compute silhouette score
        silhouette_avg = silhouette_score(features_np, cluster_labels)
        
        # Analyze cluster purity
        cluster_purity = self._compute_cluster_purity(cluster_labels, labels)
        
        # Compute cluster statistics
        cluster_stats = self._compute_cluster_statistics(features_np, cluster_labels)
        
        return {
            'cluster_labels': cluster_labels,
            'silhouette_score': silhouette_avg,
            'cluster_purity': cluster_purity,
            'cluster_stats': cluster_stats,
            'n_clusters': n_clusters
        }
    
    def _compute_cluster_purity(self, cluster_labels: np.ndarray, true_labels: np.ndarray) -> float:
        """Compute cluster purity."""
        n_clusters = len(np.unique(cluster_labels))
        purity = 0.0
        
        for cluster in range(n_clusters):
            cluster_mask = cluster_labels == cluster
            if np.sum(cluster_mask) > 0:
                cluster_true_labels = true_labels[cluster_mask]
                most_common_label = np.bincount(cluster_true_labels).argmax()
                cluster_purity = np.sum(cluster_true_labels == most_common_label) / np.sum(cluster_mask)
                purity += cluster_purity * np.sum(cluster_mask)
        
        return purity / len(cluster_labels)
    
    def _compute_cluster_statistics(self, features: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Compute statistics for each cluster."""
        unique_clusters = np.unique(cluster_labels)
        cluster_stats = {}
        
        for cluster in unique_clusters:
            cluster_mask = cluster_labels == cluster
            cluster_features = features[cluster_mask]
            
            cluster_stats[cluster] = {
                'size': np.sum(cluster_mask),
                'mean': np.mean(cluster_features, axis=0),
                'std': np.std(cluster_features, axis=0),
                'centroid': np.mean(cluster_features, axis=0)
            }
        
        return cluster_stats
    
    def create_cluster_visualization(self, features: torch.Tensor, cluster_labels: np.ndarray,
                                   true_labels: np.ndarray) -> plt.Figure:
        """
        Create visualization of clustering results.
        
        Args:
            features: Feature tensor
            cluster_labels: Cluster assignments
            true_labels: True labels
            
        Returns:
            Matplotlib figure
        """
        # Apply t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42)
        features_tsne = tsne.fit_transform(features.numpy())
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot true labels
        unique_labels = np.unique(true_labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = true_labels == label
            ax1.scatter(features_tsne[mask, 0], features_tsne[mask, 1], 
                      c=[colors[i]], label=f'True Class {label}', alpha=0.7)
        
        ax1.set_title('True Labels')
        ax1.set_xlabel('t-SNE Component 1')
        ax1.set_ylabel('t-SNE Component 2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot cluster labels
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.Set2(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster in enumerate(unique_clusters):
            mask = cluster_labels == cluster
            ax2.scatter(features_tsne[mask, 0], features_tsne[mask, 1], 
                       c=[colors[i]], label=f'Cluster {cluster}', alpha=0.7)
        
        ax2.set_title('Cluster Assignments')
        ax2.set_xlabel('t-SNE Component 1')
        ax2.set_ylabel('t-SNE Component 2')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def analyze_features(model: nn.Module, dataloader, max_samples: int = 1000, 
                    device: str = "cuda") -> Dict[str, Any]:
    """
    Analyze features from a model.
    
    Args:
        model: Model to analyze
        dataloader: Data loader for analysis
        max_samples: Maximum number of samples to process
        device: Device to run analysis on
        
    Returns:
        Dictionary with feature analysis results
    """
    analyzer = FeatureAnalyzer(model, device)
    analyzer.register_feature_hooks()
    
    # Extract features
    feature_data = analyzer.extract_features(dataloader, max_samples)
    
    # Analyze features
    analysis_results = {}
    for layer_name, features in feature_data['features'].items():
        analysis_results[layer_name] = analyzer.analyze_feature_distribution(
            features, feature_data['labels']
        )
    
    analyzer.remove_hooks()
    
    return {
        'feature_data': feature_data,
        'analysis_results': analysis_results
    }


def visualize_features(model: nn.Module, dataloader, max_samples: int = 1000,
                     device: str = "cuda") -> Dict[str, plt.Figure]:
    """
    Create feature visualizations.
    
    Args:
        model: Model to visualize
        dataloader: Data loader for visualization
        max_samples: Maximum number of samples to process
        device: Device to run visualization on
        
    Returns:
        Dictionary with visualization figures
    """
    analyzer = FeatureAnalyzer(model, device)
    analyzer.register_feature_hooks()
    
    # Extract features
    feature_data = analyzer.extract_features(dataloader, max_samples)
    
    # Create visualizations
    visualizer = FeatureVisualizer(model, device)
    figures = {}
    
    for layer_name, features in feature_data['features'].items():
        # t-SNE visualization
        tsne_fig = visualizer.create_feature_tsne(features, feature_data['labels'])
        figures[f'{layer_name}_tsne'] = tsne_fig
        
        # PCA visualization
        pca_fig = visualizer.create_feature_pca(features, feature_data['labels'])
        figures[f'{layer_name}_pca'] = pca_fig
        
        # Feature heatmap
        heatmap_fig = visualizer.create_feature_heatmap(features, feature_data['labels'])
        figures[f'{layer_name}_heatmap'] = heatmap_fig
    
    analyzer.remove_hooks()
    
    return figures
