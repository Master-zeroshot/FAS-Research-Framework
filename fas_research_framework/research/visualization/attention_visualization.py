"""
Attention Visualization for Face Anti-Spoofing

This module provides tools for visualizing attention maps and understanding
model focus areas in face anti-spoofing tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import json

from utils.logging_config import get_logger

logger = get_logger(__name__)


class AttentionVisualizer:
    """Visualizer for attention maps in face anti-spoofing models."""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Initialize attention visualizer.
        
        Args:
            model: Model to visualize
            device: Device to run visualization on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        # Hook for capturing attention maps
        self.attention_maps = {}
        self.hooks = []
        
        logger.info("Initialized attention visualizer")
    
    def register_hooks(self, layer_names: List[str] = None) -> None:
        """
        Register hooks for capturing attention maps.
        
        Args:
            layer_names: List of layer names to hook
        """
        if layer_names is None:
            # Default to common attention layer names
            layer_names = ['attention', 'self_attention', 'multihead_attention']
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self.attention_maps[name] = output.detach().cpu()
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
        self.attention_maps = {}
    
    def visualize_attention(self, image: torch.Tensor, layer_name: str = None) -> Dict[str, np.ndarray]:
        """
        Visualize attention maps for an image.
        
        Args:
            image: Input image tensor
            layer_name: Specific layer to visualize
            
        Returns:
            Dictionary with attention visualizations
        """
        # Clear previous attention maps
        self.attention_maps = {}
        
        # Move image to device
        image = image.to(self.device)
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(image)
        
        # Generate visualizations
        visualizations = {}
        
        if layer_name:
            # Visualize specific layer
            if layer_name in self.attention_maps:
                attention_map = self.attention_maps[layer_name]
                visualizations[layer_name] = self._create_attention_visualization(attention_map, image)
        else:
            # Visualize all captured attention maps
            for name, attention_map in self.attention_maps.items():
                visualizations[name] = self._create_attention_visualization(attention_map, image)
        
        return visualizations
    
    def _create_attention_visualization(self, attention_map: torch.Tensor, 
                                     image: torch.Tensor) -> np.ndarray:
        """Create attention visualization from attention map."""
        # Convert to numpy
        attention_map = attention_map.squeeze().numpy()
        
        # Handle different attention map shapes
        if attention_map.ndim == 4:  # [batch, heads, height, width]
            # Average over heads
            attention_map = attention_map.mean(axis=1)
        
        if attention_map.ndim == 3:  # [batch, height, width]
            attention_map = attention_map[0]  # Take first batch
        
        # Normalize attention map
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        # Resize to match image size
        image_size = image.shape[-2:]
        attention_map = cv2.resize(attention_map, (image_size[1], image_size[0]))
        
        return attention_map
    
    def create_attention_overlay(self, image: torch.Tensor, attention_map: np.ndarray,
                               alpha: float = 0.6, colormap: str = 'jet') -> np.ndarray:
        """
        Create attention overlay on image.
        
        Args:
            image: Original image tensor
            attention_map: Attention map
            alpha: Transparency for overlay
            colormap: Colormap for attention visualization
            
        Returns:
            Overlay image
        """
        # Convert image to numpy
        if image.dim() == 4:
            image = image[0]  # Take first batch
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())
        
        # Apply colormap to attention map
        cmap = plt.cm.get_cmap(colormap)
        attention_colored = cmap(attention_map)[:, :, :3]  # Remove alpha channel
        
        # Create overlay
        overlay = alpha * attention_colored + (1 - alpha) * image
        
        return overlay
    
    def save_attention_visualization(self, image: torch.Tensor, attention_maps: Dict[str, np.ndarray],
                                   output_dir: str, filename: str = "attention_vis") -> None:
        """
        Save attention visualizations to files.
        
        Args:
            image: Original image tensor
            attention_maps: Dictionary of attention maps
            output_dir: Output directory
            filename: Base filename
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert image to numpy
        if image.dim() == 4:
            image = image[0]
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        
        # Create visualizations for each attention map
        for layer_name, attention_map in attention_maps.items():
            # Create overlay
            overlay = self.create_attention_overlay(image, attention_map)
            
            # Save overlay
            overlay_path = output_dir / f"{filename}_{layer_name}_overlay.png"
            plt.imsave(overlay_path, overlay)
            
            # Save attention map only
            attention_path = output_dir / f"{filename}_{layer_name}_attention.png"
            plt.imsave(attention_path, attention_map, cmap='jet')
            
            # Create side-by-side comparison
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(image_np)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Attention map
            im = axes[1].imshow(attention_map, cmap='jet')
            axes[1].set_title(f'Attention Map - {layer_name}')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1])
            
            # Overlay
            axes[2].imshow(overlay)
            axes[2].set_title('Attention Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            comparison_path = output_dir / f"{filename}_{layer_name}_comparison.png"
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved attention visualization for {layer_name}")


class AttentionAnalyzer:
    """Analyzer for attention patterns in face anti-spoofing models."""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Initialize attention analyzer.
        
        Args:
            model: Model to analyze
            device: Device to run analysis on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        logger.info("Initialized attention analyzer")
    
    def analyze_attention_patterns(self, dataloader, num_samples: int = 100) -> Dict[str, Any]:
        """
        Analyze attention patterns across multiple samples.
        
        Args:
            dataloader: Data loader for analysis
            num_samples: Number of samples to analyze
            
        Returns:
            Dictionary with attention analysis results
        """
        attention_stats = {
            'mean_attention': [],
            'attention_entropy': [],
            'attention_sparsity': [],
            'attention_consistency': []
        }
        
        sample_count = 0
        
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            images, labels, metadata = batch
            images = images.to(self.device)
            
            # Analyze each image in batch
            for i in range(images.size(0)):
                if sample_count >= num_samples:
                    break
                
                image = images[i:i+1]
                label = labels[i].item()
                
                # Get attention maps
                visualizer = AttentionVisualizer(self.model, self.device)
                visualizer.register_hooks()
                attention_maps = visualizer.visualize_attention(image)
                visualizer.remove_hooks()
                
                # Analyze attention patterns
                for layer_name, attention_map in attention_maps.items():
                    stats = self._analyze_single_attention_map(attention_map, label)
                    attention_stats['mean_attention'].append(stats['mean_attention'])
                    attention_stats['attention_entropy'].append(stats['entropy'])
                    attention_stats['attention_sparsity'].append(stats['sparsity'])
                
                sample_count += 1
        
        # Compute aggregate statistics
        aggregate_stats = self._compute_aggregate_statistics(attention_stats)
        
        return aggregate_stats
    
    def _analyze_single_attention_map(self, attention_map: np.ndarray, label: int) -> Dict[str, float]:
        """Analyze a single attention map."""
        # Mean attention
        mean_attention = attention_map.mean()
        
        # Attention entropy (measure of attention distribution)
        attention_flat = attention_map.flatten()
        attention_flat = attention_flat / (attention_flat.sum() + 1e-8)
        entropy = -np.sum(attention_flat * np.log(attention_flat + 1e-8))
        
        # Attention sparsity (measure of focus)
        sparsity = np.sum(attention_flat > 0.1) / len(attention_flat)
        
        return {
            'mean_attention': mean_attention,
            'entropy': entropy,
            'sparsity': sparsity
        }
    
    def _compute_aggregate_statistics(self, attention_stats: Dict[str, List[float]]) -> Dict[str, Any]:
        """Compute aggregate statistics from attention analysis."""
        aggregate_stats = {}
        
        for stat_name, values in attention_stats.items():
            if values:
                aggregate_stats[stat_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        return aggregate_stats
    
    def compare_attention_patterns(self, real_attention: List[np.ndarray], 
                                spoof_attention: List[np.ndarray]) -> Dict[str, Any]:
        """
        Compare attention patterns between real and spoof samples.
        
        Args:
            real_attention: List of attention maps for real samples
            spoof_attention: List of attention maps for spoof samples
            
        Returns:
            Dictionary with comparison results
        """
        # Analyze real samples
        real_stats = self._analyze_attention_group(real_attention)
        
        # Analyze spoof samples
        spoof_stats = self._analyze_attention_group(spoof_attention)
        
        # Compute differences
        differences = {}
        for stat_name in real_stats:
            differences[stat_name] = {
                'real_mean': real_stats[stat_name]['mean'],
                'spoof_mean': spoof_stats[stat_name]['mean'],
                'difference': real_stats[stat_name]['mean'] - spoof_stats[stat_name]['mean'],
                'relative_difference': (real_stats[stat_name]['mean'] - spoof_stats[stat_name]['mean']) / real_stats[stat_name]['mean'] if real_stats[stat_name]['mean'] != 0 else 0
            }
        
        return {
            'real_stats': real_stats,
            'spoof_stats': spoof_stats,
            'differences': differences
        }
    
    def _analyze_attention_group(self, attention_maps: List[np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Analyze a group of attention maps."""
        if not attention_maps:
            return {}
        
        # Compute statistics for each attention map
        stats_list = []
        for attention_map in attention_maps:
            stats = self._analyze_single_attention_map(attention_map, 0)  # Label doesn't matter for this analysis
            stats_list.append(stats)
        
        # Aggregate statistics
        aggregate_stats = {}
        for stat_name in stats_list[0].keys():
            values = [stats[stat_name] for stats in stats_list]
            aggregate_stats[stat_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        return aggregate_stats


class AttentionHeatmapGenerator:
    """Generator for attention heatmaps and visualizations."""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Initialize attention heatmap generator.
        
        Args:
            model: Model to generate heatmaps for
            device: Device to run generation on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        logger.info("Initialized attention heatmap generator")
    
    def generate_attention_heatmap(self, image: torch.Tensor, layer_name: str = None) -> np.ndarray:
        """
        Generate attention heatmap for an image.
        
        Args:
            image: Input image tensor
            layer_name: Specific layer to visualize
            
        Returns:
            Attention heatmap
        """
        visualizer = AttentionVisualizer(self.model, self.device)
        visualizer.register_hooks([layer_name] if layer_name else None)
        
        attention_maps = visualizer.visualize_attention(image, layer_name)
        visualizer.remove_hooks()
        
        if attention_maps:
            # Return the first (or only) attention map
            return list(attention_maps.values())[0]
        else:
            return None
    
    def create_attention_grid(self, images: torch.Tensor, labels: torch.Tensor,
                            layer_name: str = None, grid_size: Tuple[int, int] = (4, 4)) -> plt.Figure:
        """
        Create a grid of attention visualizations.
        
        Args:
            images: Batch of images
            labels: Corresponding labels
            layer_name: Layer to visualize
            grid_size: Size of the grid
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(20, 20))
        axes = axes.flatten()
        
        for i in range(min(len(images), len(axes))):
            image = images[i:i+1]
            label = labels[i].item()
            
            # Generate attention heatmap
            attention_map = self.generate_attention_heatmap(image, layer_name)
            
            if attention_map is not None:
                # Create overlay
                visualizer = AttentionVisualizer(self.model, self.device)
                overlay = visualizer.create_attention_overlay(image, attention_map)
                
                # Plot
                axes[i].imshow(overlay)
                axes[i].set_title(f'Label: {label}')
                axes[i].axis('off')
            else:
                # Plot original image if no attention map
                image_np = image[0].permute(1, 2, 0).cpu().numpy()
                axes[i].imshow(image_np)
                axes[i].set_title(f'Label: {label} (No Attention)')
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(images), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def save_attention_grid(self, images: torch.Tensor, labels: torch.Tensor,
                          output_path: str, layer_name: str = None, 
                          grid_size: Tuple[int, int] = (4, 4)) -> None:
        """
        Save attention grid to file.
        
        Args:
            images: Batch of images
            labels: Corresponding labels
            output_path: Output file path
            layer_name: Layer to visualize
            grid_size: Size of the grid
        """
        fig = self.create_attention_grid(images, labels, layer_name, grid_size)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved attention grid to {output_path}")


def visualize_attention(model: nn.Module, image: torch.Tensor, 
                        layer_name: str = None, device: str = "cuda") -> Dict[str, np.ndarray]:
    """
    Visualize attention maps for an image.
    
    Args:
        model: Model to visualize
        image: Input image tensor
        layer_name: Specific layer to visualize
        device: Device to run visualization on
        
    Returns:
        Dictionary with attention visualizations
    """
    visualizer = AttentionVisualizer(model, device)
    visualizer.register_hooks([layer_name] if layer_name else None)
    
    attention_maps = visualizer.visualize_attention(image, layer_name)
    visualizer.remove_hooks()
    
    return attention_maps


def analyze_attention_patterns(model: nn.Module, dataloader, 
                             num_samples: int = 100, device: str = "cuda") -> Dict[str, Any]:
    """
    Analyze attention patterns across multiple samples.
    
    Args:
        model: Model to analyze
        dataloader: Data loader for analysis
        num_samples: Number of samples to analyze
        device: Device to run analysis on
        
    Returns:
        Dictionary with attention analysis results
    """
    analyzer = AttentionAnalyzer(model, device)
    return analyzer.analyze_attention_patterns(dataloader, num_samples)
