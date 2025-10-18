"""
Cross-Dataset Evaluation for Face Anti-Spoofing

This module provides tools for evaluating models across different datasets
to assess generalization and robustness for face anti-spoofing tasks.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

from utils.logging_config import get_logger

logger = get_logger(__name__)


class CrossDatasetEvaluator:
    """Cross-dataset evaluation for face anti-spoofing models."""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Initialize cross-dataset evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        logger.info("Initialized cross-dataset evaluator")
    
    def evaluate_cross_dataset(self, dataloaders: Dict[str, DataLoader]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model across multiple datasets.
        
        Args:
            dataloaders: Dictionary of dataset name -> dataloader
            
        Returns:
            Dictionary with cross-dataset evaluation results
        """
        results = {}
        
        for dataset_name, dataloader in dataloaders.items():
            logger.info(f"Evaluating on {dataset_name} dataset...")
            
            dataset_results = self._evaluate_single_dataset(dataloader)
            results[dataset_name] = dataset_results
        
        # Compute cross-dataset statistics
        cross_dataset_stats = self._compute_cross_dataset_statistics(results)
        results['cross_dataset_statistics'] = cross_dataset_stats
        
        return results
    
    def _evaluate_single_dataset(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a single dataset."""
        total_samples = 0
        correct_predictions = 0
        real_correct = 0
        spoof_correct = 0
        real_total = 0
        spoof_total = 0
        
        # Confusion matrix
        confusion_matrix = np.zeros((2, 2))  # [real, spoof] x [predicted_real, predicted_spoof]
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    images, labels, metadata = batch
                else:
                    images, labels = batch
                    metadata = None
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                predictions = outputs.argmax(dim=1)
                
                # Update counters
                total_samples += labels.size(0)
                correct_predictions += (predictions == labels).sum().item()
                
                # Real samples
                real_mask = labels == 0
                if real_mask.any():
                    real_total += real_mask.sum().item()
                    real_correct += (predictions[real_mask] == labels[real_mask]).sum().item()
                
                # Spoof samples
                spoof_mask = labels == 1
                if spoof_mask.any():
                    spoof_total += spoof_mask.sum().item()
                    spoof_correct += (predictions[spoof_mask] == labels[spoof_mask]).sum().item()
                
                # Update confusion matrix
                for i in range(labels.size(0)):
                    true_label = labels[i].item()
                    pred_label = predictions[i].item()
                    confusion_matrix[true_label, pred_label] += 1
        
        # Compute metrics
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        real_accuracy = real_correct / real_total if real_total > 0 else 0
        spoof_accuracy = spoof_correct / spoof_total if spoof_total > 0 else 0
        
        # Compute additional metrics
        precision_real = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0]) if (confusion_matrix[0, 0] + confusion_matrix[1, 0]) > 0 else 0
        precision_spoof = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1]) if (confusion_matrix[1, 1] + confusion_matrix[0, 1]) > 0 else 0
        recall_real = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1]) if (confusion_matrix[0, 0] + confusion_matrix[0, 1]) > 0 else 0
        recall_spoof = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0]) if (confusion_matrix[1, 1] + confusion_matrix[1, 0]) > 0 else 0
        
        f1_real = 2 * (precision_real * recall_real) / (precision_real + recall_real) if (precision_real + recall_real) > 0 else 0
        f1_spoof = 2 * (precision_spoof * recall_spoof) / (precision_spoof + recall_spoof) if (precision_spoof + recall_spoof) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'real_accuracy': real_accuracy,
            'spoof_accuracy': spoof_accuracy,
            'precision_real': precision_real,
            'precision_spoof': precision_spoof,
            'recall_real': recall_real,
            'recall_spoof': recall_spoof,
            'f1_real': f1_real,
            'f1_spoof': f1_spoof,
            'total_samples': total_samples,
            'real_samples': real_total,
            'spoof_samples': spoof_total,
            'confusion_matrix': confusion_matrix.tolist()
        }
    
    def _compute_cross_dataset_statistics(self, results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Compute cross-dataset statistics."""
        accuracies = [result['accuracy'] for result in results.values()]
        real_accuracies = [result['real_accuracy'] for result in results.values()]
        spoof_accuracies = [result['spoof_accuracy'] for result in results.values()]
        
        return {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'mean_real_accuracy': np.mean(real_accuracies),
            'std_real_accuracy': np.std(real_accuracies),
            'mean_spoof_accuracy': np.mean(spoof_accuracies),
            'std_spoof_accuracy': np.std(spoof_accuracies),
            'accuracy_range': np.max(accuracies) - np.min(accuracies),
            'num_datasets': len(results)
        }


class DomainAdaptationEvaluator:
    """Evaluator for domain adaptation scenarios."""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Initialize domain adaptation evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        logger.info("Initialized domain adaptation evaluator")
    
    def evaluate_domain_adaptation(self, source_dataloader: DataLoader, 
                                 target_dataloader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate domain adaptation performance.
        
        Args:
            source_dataloader: Source domain dataloader
            target_dataloader: Target domain dataloader
            
        Returns:
            Dictionary with domain adaptation results
        """
        # Evaluate on source domain
        source_results = self._evaluate_single_domain(source_dataloader, "source")
        
        # Evaluate on target domain
        target_results = self._evaluate_single_domain(target_dataloader, "target")
        
        # Compute domain gap
        domain_gap = self._compute_domain_gap(source_results, target_results)
        
        return {
            'source_domain': source_results,
            'target_domain': target_results,
            'domain_gap': domain_gap
        }
    
    def _evaluate_single_domain(self, dataloader: DataLoader, domain_name: str) -> Dict[str, float]:
        """Evaluate on a single domain."""
        total_samples = 0
        correct_predictions = 0
        real_correct = 0
        spoof_correct = 0
        real_total = 0
        spoof_total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    images, labels, metadata = batch
                else:
                    images, labels = batch
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                predictions = outputs.argmax(dim=1)
                
                # Update counters
                total_samples += labels.size(0)
                correct_predictions += (predictions == labels).sum().item()
                
                # Real samples
                real_mask = labels == 0
                if real_mask.any():
                    real_total += real_mask.sum().item()
                    real_correct += (predictions[real_mask] == labels[real_mask]).sum().item()
                
                # Spoof samples
                spoof_mask = labels == 1
                if spoof_mask.any():
                    spoof_total += spoof_mask.sum().item()
                    spoof_correct += (predictions[spoof_mask] == labels[spoof_mask]).sum().item()
        
        # Compute metrics
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        real_accuracy = real_correct / real_total if real_total > 0 else 0
        spoof_accuracy = spoof_correct / spoof_total if spoof_total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'real_accuracy': real_accuracy,
            'spoof_accuracy': spoof_accuracy,
            'total_samples': total_samples,
            'real_samples': real_total,
            'spoof_samples': spoof_total
        }
    
    def _compute_domain_gap(self, source_results: Dict[str, float], 
                          target_results: Dict[str, float]) -> Dict[str, float]:
        """Compute domain gap between source and target."""
        return {
            'accuracy_gap': source_results['accuracy'] - target_results['accuracy'],
            'real_accuracy_gap': source_results['real_accuracy'] - target_results['real_accuracy'],
            'spoof_accuracy_gap': source_results['spoof_accuracy'] - target_results['spoof_accuracy'],
            'relative_accuracy_drop': (source_results['accuracy'] - target_results['accuracy']) / source_results['accuracy'] if source_results['accuracy'] > 0 else 0
        }


class RobustnessEvaluator:
    """Evaluator for model robustness across different conditions."""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Initialize robustness evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        logger.info("Initialized robustness evaluator")
    
    def evaluate_robustness(self, dataloaders: Dict[str, DataLoader]) -> Dict[str, Any]:
        """
        Evaluate model robustness across different conditions.
        
        Args:
            dataloaders: Dictionary of condition name -> dataloader
            
        Returns:
            Dictionary with robustness evaluation results
        """
        results = {}
        
        for condition_name, dataloader in dataloaders.items():
            logger.info(f"Evaluating robustness on {condition_name} condition...")
            
            condition_results = self._evaluate_condition(dataloader)
            results[condition_name] = condition_results
        
        # Compute robustness statistics
        robustness_stats = self._compute_robustness_statistics(results)
        results['robustness_statistics'] = robustness_stats
        
        return results
    
    def _evaluate_condition(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a specific condition."""
        total_samples = 0
        correct_predictions = 0
        real_correct = 0
        spoof_correct = 0
        real_total = 0
        spoof_total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    images, labels, metadata = batch
                else:
                    images, labels = batch
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                predictions = outputs.argmax(dim=1)
                
                # Update counters
                total_samples += labels.size(0)
                correct_predictions += (predictions == labels).sum().item()
                
                # Real samples
                real_mask = labels == 0
                if real_mask.any():
                    real_total += real_mask.sum().item()
                    real_correct += (predictions[real_mask] == labels[real_mask]).sum().item()
                
                # Spoof samples
                spoof_mask = labels == 1
                if spoof_mask.any():
                    spoof_total += spoof_mask.sum().item()
                    spoof_correct += (predictions[spoof_mask] == labels[spoof_mask]).sum().item()
        
        # Compute metrics
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        real_accuracy = real_correct / real_total if real_total > 0 else 0
        spoof_accuracy = spoof_correct / spoof_total if spoof_total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'real_accuracy': real_accuracy,
            'spoof_accuracy': spoof_accuracy,
            'total_samples': total_samples,
            'real_samples': real_total,
            'spoof_samples': spoof_total
        }
    
    def _compute_robustness_statistics(self, results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Compute robustness statistics."""
        accuracies = [result['accuracy'] for result in results.values()]
        real_accuracies = [result['real_accuracy'] for result in results.values()]
        spoof_accuracies = [result['spoof_accuracy'] for result in results.values()]
        
        return {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'accuracy_consistency': 1.0 - np.std(accuracies),  # Higher is better
            'mean_real_accuracy': np.mean(real_accuracies),
            'std_real_accuracy': np.std(real_accuracies),
            'mean_spoof_accuracy': np.mean(spoof_accuracies),
            'std_spoof_accuracy': np.std(spoof_accuracies),
            'num_conditions': len(results)
        }


class CrossDatasetAnalysis:
    """Analysis tools for cross-dataset evaluation results."""
    
    def __init__(self, results: Dict[str, Any]):
        """
        Initialize cross-dataset analysis.
        
        Args:
            results: Cross-dataset evaluation results
        """
        self.results = results
    
    def generate_report(self) -> str:
        """Generate comprehensive cross-dataset evaluation report."""
        report = []
        report.append("# Cross-Dataset Evaluation Report")
        report.append("=" * 50)
        report.append("")
        
        # Dataset performance summary
        report.append("## Dataset Performance Summary")
        report.append("")
        report.append("| Dataset | Accuracy | Real Acc | Spoof Acc | Samples |")
        report.append("|---------|----------|----------|-----------|---------|")
        
        for dataset_name, metrics in self.results.items():
            if dataset_name == 'cross_dataset_statistics':
                continue
            
            report.append(f"| {dataset_name} | {metrics['accuracy']:.4f} | "
                         f"{metrics['real_accuracy']:.4f} | {metrics['spoof_accuracy']:.4f} | "
                         f"{metrics['total_samples']} |")
        
        report.append("")
        
        # Cross-dataset statistics
        if 'cross_dataset_statistics' in self.results:
            stats = self.results['cross_dataset_statistics']
            report.append("## Cross-Dataset Statistics")
            report.append("")
            report.append(f"- **Mean Accuracy**: {stats['mean_accuracy']:.4f}")
            report.append(f"- **Accuracy Std**: {stats['std_accuracy']:.4f}")
            report.append(f"- **Accuracy Range**: {stats['accuracy_range']:.4f}")
            report.append(f"- **Min Accuracy**: {stats['min_accuracy']:.4f}")
            report.append(f"- **Max Accuracy**: {stats['max_accuracy']:.4f}")
            report.append(f"- **Number of Datasets**: {stats['num_datasets']}")
            report.append("")
        
        # Performance analysis
        report.append("## Performance Analysis")
        report.append("")
        
        # Find best and worst performing datasets
        dataset_accuracies = {name: metrics['accuracy'] for name, metrics in self.results.items() 
                             if name != 'cross_dataset_statistics'}
        
        if dataset_accuracies:
            best_dataset = max(dataset_accuracies, key=dataset_accuracies.get)
            worst_dataset = min(dataset_accuracies, key=dataset_accuracies.get)
            
            report.append(f"- **Best Performing Dataset**: {best_dataset} ({dataset_accuracies[best_dataset]:.4f})")
            report.append(f"- **Worst Performing Dataset**: {worst_dataset} ({dataset_accuracies[worst_dataset]:.4f})")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        if 'cross_dataset_statistics' in self.results:
            stats = self.results['cross_dataset_statistics']
            
            if stats['std_accuracy'] > 0.1:
                report.append("- **High Variance**: Model performance varies significantly across datasets")
                report.append("- **Recommendation**: Consider domain adaptation or data augmentation")
            else:
                report.append("- **Low Variance**: Model shows consistent performance across datasets")
                report.append("- **Recommendation**: Model generalizes well across different domains")
            
            if stats['mean_accuracy'] < 0.8:
                report.append("- **Low Overall Performance**: Consider model improvement or more training data")
            else:
                report.append("- **Good Overall Performance**: Model performs well across datasets")
        
        return "\n".join(report)
    
    def save_results(self, output_path: str) -> None:
        """Save results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Cross-dataset evaluation results saved to {output_path}")


def evaluate_cross_dataset(model: nn.Module, dataloaders: Dict[str, DataLoader], 
                          device: str = "cuda") -> Dict[str, Any]:
    """
    Evaluate model across multiple datasets.
    
    Args:
        model: Model to evaluate
        dataloaders: Dictionary of dataset name -> dataloader
        device: Device to run evaluation on
        
    Returns:
        Cross-dataset evaluation results
    """
    evaluator = CrossDatasetEvaluator(model, device)
    return evaluator.evaluate_cross_dataset(dataloaders)


def evaluate_domain_adaptation(model: nn.Module, source_dataloader: DataLoader,
                              target_dataloader: DataLoader, device: str = "cuda") -> Dict[str, Any]:
    """
    Evaluate domain adaptation performance.
    
    Args:
        model: Model to evaluate
        source_dataloader: Source domain dataloader
        target_dataloader: Target domain dataloader
        device: Device to run evaluation on
        
    Returns:
        Domain adaptation evaluation results
    """
    evaluator = DomainAdaptationEvaluator(model, device)
    return evaluator.evaluate_domain_adaptation(source_dataloader, target_dataloader)


def evaluate_robustness(model: nn.Module, dataloaders: Dict[str, DataLoader],
                       device: str = "cuda") -> Dict[str, Any]:
    """
    Evaluate model robustness across different conditions.
    
    Args:
        model: Model to evaluate
        dataloaders: Dictionary of condition name -> dataloader
        device: Device to run evaluation on
        
    Returns:
        Robustness evaluation results
    """
    evaluator = RobustnessEvaluator(model, device)
    return evaluator.evaluate_robustness(dataloaders)
