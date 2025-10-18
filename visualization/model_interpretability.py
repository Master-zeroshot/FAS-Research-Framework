"""
Model Interpretability Tools for Face Anti-Spoofing

This module provides tools for interpreting and understanding
face anti-spoofing models through various interpretability techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import json

from utils.logging_config import get_logger

logger = get_logger(__name__)


class GradCAM:
    """Gradient-weighted Class Activation Mapping (GradCAM) for model interpretability."""
    
    def __init__(self, model: nn.Module, target_layer: str, device: str = "cuda"):
        """
        Initialize GradCAM.
        
        Args:
            model: Model to interpret
            target_layer: Target layer for GradCAM
            device: Device to run interpretation on
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.model.eval()
        
        # Find target layer
        self.target_module = None
        for name, module in self.model.named_modules():
            if name == target_layer:
                self.target_module = module
                break
        
        if self.target_module is None:
            raise ValueError(f"Target layer '{target_layer}' not found in model")
        
        # Hook for capturing gradients and activations
        self.gradients = None
        self.activations = None
        self.hook_g = None
        self.hook_a = None
        
        logger.info(f"Initialized GradCAM for layer: {target_layer}")
    
    def register_hooks(self):
        """Register hooks for capturing gradients and activations."""
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        self.hook_g = self.target_module.register_backward_hook(backward_hook)
        self.hook_a = self.target_module.register_forward_hook(forward_hook)
    
    def remove_hooks(self):
        """Remove registered hooks."""
        if self.hook_g:
            self.hook_g.remove()
        if self.hook_a:
            self.hook_a.remove()
    
    def generate_cam(self, image: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generate GradCAM for an image.
        
        Args:
            image: Input image tensor
            class_idx: Class index for which to generate CAM
            
        Returns:
            GradCAM heatmap
        """
        # Move image to device
        image = image.to(self.device)
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        
        # Register hooks
        self.register_hooks()
        
        # Forward pass
        output = self.model(image)
        
        # Get class index
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]  # Remove batch dimension
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        # Convert to numpy
        cam = cam.cpu().numpy()
        
        # Remove hooks
        self.remove_hooks()
        
        return cam
    
    def create_cam_overlay(self, image: torch.Tensor, cam: np.ndarray, 
                          alpha: float = 0.6, colormap: str = 'jet') -> np.ndarray:
        """
        Create CAM overlay on image.
        
        Args:
            image: Original image tensor
            cam: CAM heatmap
            alpha: Transparency for overlay
            colormap: Colormap for CAM visualization
            
        Returns:
            Overlay image
        """
        # Convert image to numpy
        if image.dim() == 4:
            image = image[0]  # Take first batch
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())
        
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
        
        # Apply colormap to CAM
        cmap = plt.cm.get_cmap(colormap)
        cam_colored = cmap(cam_resized)[:, :, :3]  # Remove alpha channel
        
        # Create overlay
        overlay = alpha * cam_colored + (1 - alpha) * image
        
        return overlay


class IntegratedGradients:
    """Integrated Gradients for model interpretability."""
    
    def __init__(self, model: nn.Module, device: str = "cuda", steps: int = 50):
        """
        Initialize Integrated Gradients.
        
        Args:
            model: Model to interpret
            device: Device to run interpretation on
            steps: Number of integration steps
        """
        self.model = model
        self.device = device
        self.steps = steps
        self.model.eval()
        
        logger.info(f"Initialized Integrated Gradients with {steps} steps")
    
    def generate_attributions(self, image: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generate integrated gradients attributions.
        
        Args:
            image: Input image tensor
            class_idx: Class index for which to generate attributions
            
        Returns:
            Attribution map
        """
        # Move image to device
        image = image.to(self.device)
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        
        # Create baseline (zero image)
        baseline = torch.zeros_like(image)
        
        # Generate interpolated images
        alphas = torch.linspace(0, 1, self.steps).to(self.device)
        interpolated_images = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (image - baseline)
            interpolated_images.append(interpolated)
        
        interpolated_images = torch.cat(interpolated_images, dim=0)
        
        # Compute gradients
        interpolated_images.requires_grad_(True)
        outputs = self.model(interpolated_images)
        
        if class_idx is None:
            class_idx = outputs.argmax(dim=1)[0].item()
        
        # Compute gradients for target class
        gradients = torch.autograd.grad(
            outputs[:, class_idx].sum(),
            interpolated_images,
            create_graph=True
        )[0]
        
        # Average gradients
        avg_gradients = torch.mean(gradients, dim=0)
        
        # Compute integrated gradients
        integrated_gradients = (image - baseline) * avg_gradients
        
        # Convert to numpy
        attributions = integrated_gradients[0].detach().cpu().numpy()
        
        return attributions
    
    def create_attribution_visualization(self, image: torch.Tensor, attributions: np.ndarray,
                                       percentile: float = 99) -> np.ndarray:
        """
        Create visualization of attributions.
        
        Args:
            image: Original image tensor
            attributions: Attribution map
            percentile: Percentile for thresholding
            
        Returns:
            Visualization image
        """
        # Convert image to numpy
        if image.dim() == 4:
            image = image[0]
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())
        
        # Compute attribution magnitude
        attribution_magnitude = np.abs(attributions).sum(axis=0)
        
        # Threshold attributions
        threshold = np.percentile(attribution_magnitude, percentile)
        attribution_magnitude = np.clip(attribution_magnitude, 0, threshold)
        attribution_magnitude = attribution_magnitude / attribution_magnitude.max()
        
        # Create visualization
        visualization = image.copy()
        for c in range(3):
            visualization[:, :, c] = image[:, :, c] * (1 - attribution_magnitude) + attribution_magnitude
        
        return visualization


class LIME:
    """Local Interpretable Model-agnostic Explanations (LIME) for model interpretability."""
    
    def __init__(self, model: nn.Module, device: str = "cuda", num_samples: int = 1000):
        """
        Initialize LIME.
        
        Args:
            model: Model to interpret
            device: Device to run interpretation on
            num_samples: Number of samples for LIME
        """
        self.model = model
        self.device = device
        self.num_samples = num_samples
        self.model.eval()
        
        logger.info(f"Initialized LIME with {num_samples} samples")
    
    def generate_explanations(self, image: torch.Tensor, class_idx: int = None) -> Dict[str, Any]:
        """
        Generate LIME explanations.
        
        Args:
            image: Input image tensor
            class_idx: Class index for which to generate explanations
            
        Returns:
            Dictionary with LIME explanations
        """
        # Move image to device
        image = image.to(self.device)
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        
        # Get original prediction
        with torch.no_grad():
            original_output = self.model(image)
            if class_idx is None:
                class_idx = original_output.argmax(dim=1).item()
            original_score = original_output[0, class_idx].item()
        
        # Generate perturbed samples
        perturbed_samples = self._generate_perturbed_samples(image)
        
        # Get predictions for perturbed samples
        with torch.no_grad():
            perturbed_outputs = self.model(perturbed_samples)
            perturbed_scores = perturbed_outputs[:, class_idx].cpu().numpy()
        
        # Compute feature importance
        feature_importance = self._compute_feature_importance(
            image, perturbed_samples, original_score, perturbed_scores
        )
        
        return {
            'original_score': original_score,
            'feature_importance': feature_importance,
            'num_samples': self.num_samples
        }
    
    def _generate_perturbed_samples(self, image: torch.Tensor) -> torch.Tensor:
        """Generate perturbed samples for LIME."""
        batch_size = image.size(0)
        image_size = image.size(2)
        
        # Generate random masks
        masks = torch.rand(self.num_samples, 1, image_size, image_size).to(self.device)
        
        # Create perturbed samples
        perturbed_samples = []
        for i in range(self.num_samples):
            mask = masks[i:i+1]
            perturbed = image * mask
            perturbed_samples.append(perturbed)
        
        return torch.cat(perturbed_samples, dim=0)
    
    def _compute_feature_importance(self, image: torch.Tensor, perturbed_samples: torch.Tensor,
                                   original_score: float, perturbed_scores: np.ndarray) -> np.ndarray:
        """Compute feature importance using linear regression."""
        # Compute feature values (mask values)
        feature_values = perturbed_samples[:, 0, :, :].cpu().numpy().reshape(self.num_samples, -1)
        
        # Compute target values (score differences)
        target_values = perturbed_scores - original_score
        
        # Fit linear regression
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        reg.fit(feature_values, target_values)
        
        # Get feature importance
        feature_importance = reg.coef_.reshape(image.size(2), image.size(3))
        
        return feature_importance


class ModelInterpretabilityAnalyzer:
    """Comprehensive model interpretability analyzer."""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Initialize model interpretability analyzer.
        
        Args:
            model: Model to analyze
            device: Device to run analysis on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        logger.info("Initialized model interpretability analyzer")
    
    def analyze_model_interpretability(self, image: torch.Tensor, target_layer: str = None) -> Dict[str, Any]:
        """
        Analyze model interpretability using multiple methods.
        
        Args:
            image: Input image tensor
            target_layer: Target layer for GradCAM
            
        Returns:
            Dictionary with interpretability analysis
        """
        results = {}
        
        # Get model prediction
        with torch.no_grad():
            output = self.model(image.to(self.device))
            predicted_class = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1)[0, predicted_class].item()
        
        results['prediction'] = {
            'predicted_class': predicted_class,
            'confidence': confidence
        }
        
        # GradCAM analysis
        if target_layer:
            try:
                gradcam = GradCAM(self.model, target_layer, self.device)
                cam = gradcam.generate_cam(image, predicted_class)
                results['gradcam'] = {
                    'cam': cam,
                    'target_layer': target_layer
                }
            except Exception as e:
                logger.warning(f"GradCAM analysis failed: {e}")
                results['gradcam'] = None
        
        # Integrated Gradients analysis
        try:
            ig = IntegratedGradients(self.model, self.device)
            attributions = ig.generate_attributions(image, predicted_class)
            results['integrated_gradients'] = {
                'attributions': attributions
            }
        except Exception as e:
            logger.warning(f"Integrated Gradients analysis failed: {e}")
            results['integrated_gradients'] = None
        
        # LIME analysis
        try:
            lime = LIME(self.model, self.device)
            lime_explanations = lime.generate_explanations(image, predicted_class)
            results['lime'] = lime_explanations
        except Exception as e:
            logger.warning(f"LIME analysis failed: {e}")
            results['lime'] = None
        
        return results
    
    def create_interpretability_visualization(self, image: torch.Tensor, 
                                           analysis_results: Dict[str, Any]) -> plt.Figure:
        """
        Create comprehensive interpretability visualization.
        
        Args:
            image: Original image tensor
            analysis_results: Results from interpretability analysis
            
        Returns:
            Matplotlib figure
        """
        # Convert image to numpy
        if image.dim() == 4:
            image = image[0]
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        
        # Determine number of subplots
        n_plots = 1  # Original image
        if analysis_results.get('gradcam'):
            n_plots += 1
        if analysis_results.get('integrated_gradients'):
            n_plots += 1
        if analysis_results.get('lime'):
            n_plots += 1
        
        # Create subplots
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Original image
        axes[plot_idx].imshow(image_np)
        axes[plot_idx].set_title('Original Image')
        axes[plot_idx].axis('off')
        plot_idx += 1
        
        # GradCAM
        if analysis_results.get('gradcam'):
            cam = analysis_results['gradcam']['cam']
            axes[plot_idx].imshow(cam, cmap='jet')
            axes[plot_idx].set_title('GradCAM')
            axes[plot_idx].axis('off')
            plot_idx += 1
        
        # Integrated Gradients
        if analysis_results.get('integrated_gradients'):
            attributions = analysis_results['integrated_gradients']['attributions']
            attribution_magnitude = np.abs(attributions).sum(axis=0)
            axes[plot_idx].imshow(attribution_magnitude, cmap='hot')
            axes[plot_idx].set_title('Integrated Gradients')
            axes[plot_idx].axis('off')
            plot_idx += 1
        
        # LIME
        if analysis_results.get('lime'):
            feature_importance = analysis_results['lime']['feature_importance']
            axes[plot_idx].imshow(feature_importance, cmap='viridis')
            axes[plot_idx].set_title('LIME Feature Importance')
            axes[plot_idx].axis('off')
            plot_idx += 1
        
        plt.tight_layout()
        return fig
    
    def save_interpretability_analysis(self, image: torch.Tensor, analysis_results: Dict[str, Any],
                                     output_path: str) -> None:
        """
        Save interpretability analysis results.
        
        Args:
            image: Original image tensor
            analysis_results: Results from interpretability analysis
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create visualization
        fig = self.create_interpretability_visualization(image, analysis_results)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Save analysis results
        results_path = output_path.with_suffix('.json')
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in analysis_results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, np.ndarray):
                            serializable_results[key][sub_key] = sub_value.tolist()
                        else:
                            serializable_results[key][sub_key] = sub_value
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved interpretability analysis to {output_path}")


def analyze_model_interpretability(model: nn.Module, image: torch.Tensor, 
                                 target_layer: str = None, device: str = "cuda") -> Dict[str, Any]:
    """
    Analyze model interpretability.
    
    Args:
        model: Model to analyze
        image: Input image tensor
        target_layer: Target layer for GradCAM
        device: Device to run analysis on
        
    Returns:
        Dictionary with interpretability analysis results
    """
    analyzer = ModelInterpretabilityAnalyzer(model, device)
    return analyzer.analyze_model_interpretability(image, target_layer)


def create_interpretability_visualization(model: nn.Module, image: torch.Tensor,
                                        target_layer: str = None, device: str = "cuda") -> plt.Figure:
    """
    Create interpretability visualization.
    
    Args:
        model: Model to visualize
        image: Input image tensor
        target_layer: Target layer for GradCAM
        device: Device to run visualization on
        
    Returns:
        Matplotlib figure
    """
    analyzer = ModelInterpretabilityAnalyzer(model, device)
    analysis_results = analyzer.analyze_model_interpretability(image, target_layer)
    return analyzer.create_interpretability_visualization(image, analysis_results)
