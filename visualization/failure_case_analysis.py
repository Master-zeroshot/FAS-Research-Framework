"""
Failure Case Analysis for Face Anti-Spoofing

This module provides tools for analyzing and understanding
failure cases in face anti-spoofing models.
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
from collections import defaultdict

from utils.logging_config import get_logger

logger = get_logger(__name__)


class FailureCaseAnalyzer:
    """Analyzer for failure cases in face anti-spoofing models."""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Initialize failure case analyzer.
        
        Args:
            model: Model to analyze
            device: Device to run analysis on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        logger.info("Initialized failure case analyzer")
    
    def identify_failure_cases(self, dataloader, confidence_threshold: float = 0.5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify failure cases in the dataset.
        
        Args:
            dataloader: Data loader for analysis
            confidence_threshold: Confidence threshold for failure detection
            
        Returns:
            Dictionary with failure cases categorized by type
        """
        failure_cases = {
            'false_positives': [],  # Real classified as spoof
            'false_negatives': [],  # Spoof classified as real
            'low_confidence': [],   # Low confidence predictions
            'high_confidence_errors': []  # High confidence but wrong
        }
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if len(batch) == 3:
                    images, labels, metadata = batch
                else:
                    images, labels = batch
                    metadata = None
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Get predictions
                outputs = self.model(images)
                predictions = outputs.argmax(dim=1)
                confidences = F.softmax(outputs, dim=1).max(dim=1)[0]
                
                # Analyze each sample in batch
                for i in range(images.size(0)):
                    true_label = labels[i].item()
                    predicted_label = predictions[i].item()
                    confidence = confidences[i].item()
                    
                    # Create failure case record
                    failure_case = {
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'true_label': true_label,
                        'predicted_label': predicted_label,
                        'confidence': confidence,
                        'image': images[i].cpu(),
                        'metadata': metadata[i] if metadata else None
                    }
                    
                    # Categorize failure
                    if true_label != predicted_label:
                        if confidence < confidence_threshold:
                            failure_cases['low_confidence'].append(failure_case)
                        else:
                            failure_cases['high_confidence_errors'].append(failure_case)
                        
                        # Specific false positive/negative categorization
                        if true_label == 0 and predicted_label == 1:  # Real classified as spoof
                            failure_cases['false_positives'].append(failure_case)
                        elif true_label == 1 and predicted_label == 0:  # Spoof classified as real
                            failure_cases['false_negatives'].append(failure_case)
                    elif confidence < confidence_threshold:
                        # Correct prediction but low confidence
                        failure_cases['low_confidence'].append(failure_case)
        
        # Log failure statistics
        total_failures = sum(len(cases) for cases in failure_cases.values())
        logger.info(f"Identified {total_failures} failure cases:")
        for category, cases in failure_cases.items():
            logger.info(f"  {category}: {len(cases)} cases")
        
        return failure_cases
    
    def analyze_failure_patterns(self, failure_cases: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Analyze patterns in failure cases.
        
        Args:
            failure_cases: Dictionary of failure cases
            
        Returns:
            Dictionary with failure pattern analysis
        """
        analysis = {}
        
        for category, cases in failure_cases.items():
            if not cases:
                continue
            
            # Extract features for analysis
            confidences = [case['confidence'] for case in cases]
            true_labels = [case['true_label'] for case in cases]
            predicted_labels = [case['predicted_label'] for case in cases]
            
            # Compute statistics
            stats = {
                'count': len(cases),
                'confidence_mean': np.mean(confidences),
                'confidence_std': np.std(confidences),
                'confidence_min': np.min(confidences),
                'confidence_max': np.max(confidences),
                'true_label_distribution': np.bincount(true_labels),
                'predicted_label_distribution': np.bincount(predicted_labels)
            }
            
            # Analyze metadata if available
            if cases[0]['metadata']:
                metadata_analysis = self._analyze_metadata_patterns(cases)
                stats['metadata_patterns'] = metadata_analysis
            
            analysis[category] = stats
        
        return analysis
    
    def _analyze_metadata_patterns(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze metadata patterns in failure cases."""
        metadata_patterns = {}
        
        # Collect all metadata keys
        all_keys = set()
        for case in cases:
            if case['metadata']:
                all_keys.update(case['metadata'].keys())
        
        # Analyze each metadata field
        for key in all_keys:
            values = []
            for case in cases:
                if case['metadata'] and key in case['metadata']:
                    values.append(case['metadata'][key])
            
            if values:
                # Compute value distribution
                unique_values, counts = np.unique(values, return_counts=True)
                metadata_patterns[key] = {
                    'unique_values': unique_values.tolist(),
                    'counts': counts.tolist(),
                    'most_common': unique_values[np.argmax(counts)]
                }
        
        return metadata_patterns
    
    def create_failure_visualization(self, failure_cases: Dict[str, List[Dict[str, Any]]],
                                   max_cases_per_category: int = 10) -> plt.Figure:
        """
        Create visualization of failure cases.
        
        Args:
            failure_cases: Dictionary of failure cases
            max_cases_per_category: Maximum cases to show per category
            
        Returns:
            Matplotlib figure
        """
        # Count total categories with cases
        categories_with_cases = [cat for cat, cases in failure_cases.items() if cases]
        n_categories = len(categories_with_cases)
        
        if n_categories == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No failure cases found', ha='center', va='center', fontsize=16)
            ax.set_title('Failure Case Analysis')
            return fig
        
        # Create subplots
        fig, axes = plt.subplots(n_categories, 1, figsize=(15, 5 * n_categories))
        if n_categories == 1:
            axes = [axes]
        
        for i, category in enumerate(categories_with_cases):
            cases = failure_cases[category][:max_cases_per_category]
            ax = axes[i]
            
            # Create grid of failure cases
            n_cases = len(cases)
            n_cols = min(5, n_cases)
            n_rows = (n_cases + n_cols - 1) // n_cols
            
            # Create subplot grid
            subfig = ax.figure.add_subplot(111)
            subfig.axis('off')
            
            for j, case in enumerate(cases):
                row = j // n_cols
                col = j % n_cols
                
                # Get image
                image = case['image']
                if image.dim() == 3:
                    image_np = image.permute(1, 2, 0).numpy()
                else:
                    image_np = image.numpy()
                
                # Normalize image
                image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
                
                # Create subplot
                sub_ax = plt.subplot2grid((n_rows, n_cols), (row, col))
                sub_ax.imshow(image_np)
                sub_ax.set_title(f'True: {case["true_label"]}, Pred: {case["predicted_label"]}\nConf: {case["confidence"]:.3f}')
                sub_ax.axis('off')
            
            # Hide unused subplots
            for j in range(n_cases, n_rows * n_cols):
                row = j // n_cols
                col = j % n_cols
                sub_ax = plt.subplot2grid((n_rows, n_cols), (row, col))
                sub_ax.axis('off')
            
            ax.set_title(f'{category.replace("_", " ").title()} ({len(failure_cases[category])} cases)')
        
        plt.tight_layout()
        return fig
    
    def analyze_confidence_distribution(self, dataloader) -> Dict[str, Any]:
        """
        Analyze confidence distribution across the dataset.
        
        Args:
            dataloader: Data loader for analysis
            
        Returns:
            Dictionary with confidence analysis
        """
        all_confidences = []
        correct_confidences = []
        incorrect_confidences = []
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    images, labels, metadata = batch
                else:
                    images, labels = batch
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Get predictions
                outputs = self.model(images)
                predictions = outputs.argmax(dim=1)
                confidences = F.softmax(outputs, dim=1).max(dim=1)[0]
                
                # Categorize confidences
                for i in range(images.size(0)):
                    confidence = confidences[i].item()
                    is_correct = predictions[i].item() == labels[i].item()
                    
                    all_confidences.append(confidence)
                    
                    if is_correct:
                        correct_confidences.append(confidence)
                    else:
                        incorrect_confidences.append(confidence)
        
        # Compute statistics
        analysis = {
            'overall': {
                'mean': np.mean(all_confidences),
                'std': np.std(all_confidences),
                'min': np.min(all_confidences),
                'max': np.max(all_confidences),
                'median': np.median(all_confidences)
            },
            'correct_predictions': {
                'mean': np.mean(correct_confidences),
                'std': np.std(correct_confidences),
                'min': np.min(correct_confidences),
                'max': np.max(correct_confidences),
                'median': np.median(correct_confidences)
            },
            'incorrect_predictions': {
                'mean': np.mean(incorrect_confidences),
                'std': np.std(incorrect_confidences),
                'min': np.min(incorrect_confidences),
                'max': np.max(incorrect_confidences),
                'median': np.median(incorrect_confidences)
            }
        }
        
        return analysis
    
    def create_confidence_visualization(self, confidence_analysis: Dict[str, Any]) -> plt.Figure:
        """
        Create visualization of confidence distribution.
        
        Args:
            confidence_analysis: Results from confidence analysis
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Overall confidence distribution
        ax = axes[0, 0]
        ax.hist(confidence_analysis['overall']['mean'], bins=50, alpha=0.7, color='blue')
        ax.set_title('Overall Confidence Distribution')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Correct vs Incorrect confidence comparison
        ax = axes[0, 1]
        ax.hist(confidence_analysis['correct_predictions']['mean'], bins=30, alpha=0.7, 
               color='green', label='Correct', density=True)
        ax.hist(confidence_analysis['incorrect_predictions']['mean'], bins=30, alpha=0.7, 
               color='red', label='Incorrect', density=True)
        ax.set_title('Confidence Distribution: Correct vs Incorrect')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Confidence statistics
        ax = axes[1, 0]
        categories = ['Overall', 'Correct', 'Incorrect']
        means = [
            confidence_analysis['overall']['mean'],
            confidence_analysis['correct_predictions']['mean'],
            confidence_analysis['incorrect_predictions']['mean']
        ]
        stds = [
            confidence_analysis['overall']['std'],
            confidence_analysis['correct_predictions']['std'],
            confidence_analysis['incorrect_predictions']['std']
        ]
        
        x = np.arange(len(categories))
        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_title('Confidence Statistics')
        ax.set_xlabel('Category')
        ax.set_ylabel('Mean Confidence')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.grid(True, alpha=0.3)
        
        # Confidence range
        ax = axes[1, 1]
        categories = ['Overall', 'Correct', 'Incorrect']
        mins = [
            confidence_analysis['overall']['min'],
            confidence_analysis['correct_predictions']['min'],
            confidence_analysis['incorrect_predictions']['min']
        ]
        maxs = [
            confidence_analysis['overall']['max'],
            confidence_analysis['correct_predictions']['max'],
            confidence_analysis['incorrect_predictions']['max']
        ]
        
        x = np.arange(len(categories))
        ax.bar(x, maxs, alpha=0.7, color='lightblue', label='Max')
        ax.bar(x, mins, alpha=0.7, color='darkblue', label='Min')
        ax.set_title('Confidence Range')
        ax.set_xlabel('Category')
        ax.set_ylabel('Confidence')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class FailureCaseReporter:
    """Reporter for failure case analysis results."""
    
    def __init__(self, output_dir: str):
        """
        Initialize failure case reporter.
        
        Args:
            output_dir: Output directory for reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized failure case reporter with output directory: {output_dir}")
    
    def generate_failure_report(self, failure_cases: Dict[str, List[Dict[str, Any]]],
                              failure_patterns: Dict[str, Any],
                              confidence_analysis: Dict[str, Any]) -> str:
        """
        Generate comprehensive failure case report.
        
        Args:
            failure_cases: Dictionary of failure cases
            failure_patterns: Results from failure pattern analysis
            confidence_analysis: Results from confidence analysis
            
        Returns:
            Report text
        """
        report = []
        report.append("# Failure Case Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary statistics
        total_failures = sum(len(cases) for cases in failure_cases.values())
        report.append("## Summary Statistics")
        report.append("")
        report.append(f"- **Total Failure Cases**: {total_failures}")
        report.append("")
        
        for category, cases in failure_cases.items():
            report.append(f"- **{category.replace('_', ' ').title()}**: {len(cases)} cases")
        report.append("")
        
        # Failure pattern analysis
        report.append("## Failure Pattern Analysis")
        report.append("")
        
        for category, patterns in failure_patterns.items():
            if not patterns:
                continue
            
            report.append(f"### {category.replace('_', ' ').title()}")
            report.append("")
            report.append(f"- **Count**: {patterns['count']}")
            report.append(f"- **Mean Confidence**: {patterns['confidence_mean']:.4f}")
            report.append(f"- **Confidence Std**: {patterns['confidence_std']:.4f}")
            report.append(f"- **Confidence Range**: {patterns['confidence_min']:.4f} - {patterns['confidence_max']:.4f}")
            report.append("")
        
        # Confidence analysis
        report.append("## Confidence Analysis")
        report.append("")
        report.append("### Overall Statistics")
        report.append(f"- **Mean Confidence**: {confidence_analysis['overall']['mean']:.4f}")
        report.append(f"- **Confidence Std**: {confidence_analysis['overall']['std']:.4f}")
        report.append(f"- **Confidence Range**: {confidence_analysis['overall']['min']:.4f} - {confidence_analysis['overall']['max']:.4f}")
        report.append("")
        
        report.append("### Correct Predictions")
        report.append(f"- **Mean Confidence**: {confidence_analysis['correct_predictions']['mean']:.4f}")
        report.append(f"- **Confidence Std**: {confidence_analysis['correct_predictions']['std']:.4f}")
        report.append("")
        
        report.append("### Incorrect Predictions")
        report.append(f"- **Mean Confidence**: {confidence_analysis['incorrect_predictions']['mean']:.4f}")
        report.append(f"- **Confidence Std**: {confidence_analysis['incorrect_predictions']['std']:.4f}")
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        if confidence_analysis['incorrect_predictions']['mean'] > confidence_analysis['correct_predictions']['mean']:
            report.append("- **High Confidence Errors**: Model is overconfident in incorrect predictions")
            report.append("- **Recommendation**: Consider confidence calibration or threshold adjustment")
        else:
            report.append("- **Low Confidence Errors**: Model is appropriately uncertain about incorrect predictions")
            report.append("- **Recommendation**: Model confidence is well-calibrated")
        
        if failure_cases['false_positives']:
            report.append(f"- **False Positives**: {len(failure_cases['false_positives'])} cases of real faces classified as spoof")
            report.append("- **Recommendation**: Review real face samples and consider data augmentation")
        
        if failure_cases['false_negatives']:
            report.append(f"- **False Negatives**: {len(failure_cases['false_negatives'])} cases of spoof faces classified as real")
            report.append("- **Recommendation**: Review spoof samples and consider additional training data")
        
        return "\n".join(report)
    
    def save_failure_report(self, failure_cases: Dict[str, List[Dict[str, Any]]],
                          failure_patterns: Dict[str, Any],
                          confidence_analysis: Dict[str, Any],
                          filename: str = "failure_analysis_report.md") -> None:
        """
        Save failure case report to file.
        
        Args:
            failure_cases: Dictionary of failure cases
            failure_patterns: Results from failure pattern analysis
            confidence_analysis: Results from confidence analysis
            filename: Output filename
        """
        report = self.generate_failure_report(failure_cases, failure_patterns, confidence_analysis)
        
        report_path = self.output_dir / filename
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Saved failure case report to {report_path}")
    
    def save_failure_cases(self, failure_cases: Dict[str, List[Dict[str, Any]]],
                          filename: str = "failure_cases.json") -> None:
        """
        Save failure cases to JSON file.
        
        Args:
            failure_cases: Dictionary of failure cases
            filename: Output filename
        """
        # Convert failure cases to serializable format
        serializable_cases = {}
        for category, cases in failure_cases.items():
            serializable_cases[category] = []
            for case in cases:
                serializable_case = {
                    'batch_idx': case['batch_idx'],
                    'sample_idx': case['sample_idx'],
                    'true_label': case['true_label'],
                    'predicted_label': case['predicted_label'],
                    'confidence': case['confidence'],
                    'metadata': case['metadata']
                }
                serializable_cases[category].append(serializable_case)
        
        cases_path = self.output_dir / filename
        with open(cases_path, 'w') as f:
            json.dump(serializable_cases, f, indent=2)
        
        logger.info(f"Saved failure cases to {cases_path}")


def analyze_failure_cases(model: nn.Module, dataloader, confidence_threshold: float = 0.5,
                         device: str = "cuda") -> Dict[str, Any]:
    """
    Analyze failure cases in a model.
    
    Args:
        model: Model to analyze
        dataloader: Data loader for analysis
        confidence_threshold: Confidence threshold for failure detection
        device: Device to run analysis on
        
    Returns:
        Dictionary with failure case analysis results
    """
    analyzer = FailureCaseAnalyzer(model, device)
    
    # Identify failure cases
    failure_cases = analyzer.identify_failure_cases(dataloader, confidence_threshold)
    
    # Analyze failure patterns
    failure_patterns = analyzer.analyze_failure_patterns(failure_cases)
    
    # Analyze confidence distribution
    confidence_analysis = analyzer.analyze_confidence_distribution(dataloader)
    
    return {
        'failure_cases': failure_cases,
        'failure_patterns': failure_patterns,
        'confidence_analysis': confidence_analysis
    }


def create_failure_visualization(model: nn.Module, dataloader, confidence_threshold: float = 0.5,
                               device: str = "cuda") -> plt.Figure:
    """
    Create failure case visualization.
    
    Args:
        model: Model to analyze
        dataloader: Data loader for analysis
        confidence_threshold: Confidence threshold for failure detection
        device: Device to run analysis on
        
    Returns:
        Matplotlib figure
    """
    analyzer = FailureCaseAnalyzer(model, device)
    
    # Identify failure cases
    failure_cases = analyzer.identify_failure_cases(dataloader, confidence_threshold)
    
    # Create visualization
    return analyzer.create_failure_visualization(failure_cases)
