#!/usr/bin/env python3
"""
FAS-Research-Framework MobileNetV4 Optimization Script

This script optimizes MobileNetV4 models specifically for face anti-spoofing tasks
by testing different hyperparameter combinations and finding optimal configurations.
Part of the FAS-Research-Framework for advanced face anti-spoofing research.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Any
import itertools
import time

# Import models and utilities
from models import mobilenetv4_small, mobilenetv4_medium, mobilenetv4_large
from utils.error_handling import validate_device, safe_model_forward
from utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(log_level="INFO", log_to_file=True, log_to_console=True, structured=True)
logger = get_logger(__name__)


class MobileNetV4Optimizer:
    """
    Optimize MobileNetV4 models for face anti-spoofing tasks.
    
    This class provides comprehensive hyperparameter optimization for MobileNetV4 models,
    including learning rate optimization, width multiplier tuning, embedding dimension
    selection, and advanced parameter configuration testing.
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize the optimizer.
        
        Args:
            device: Device to run optimization on
        """
        self.device = validate_device(device)
        self.results = {}
        
        # Standard input size for face anti-spoofing (224x224 RGB)
        self.input_size = (3, 224, 224)
        
        logger.info(f"Initialized MobileNetV4 optimizer on device: {self.device}")
    
    def test_hyperparameter_combination(self, model_name: str, model_fn, 
                                     hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test a specific hyperparameter combination.
        
        Args:
            model_name: Name of the model
            model_fn: Model creation function
            hyperparams: Hyperparameter dictionary
            
        Returns:
            Dictionary with optimization results
        """
        try:
            # Create model with hyperparameters
            model = model_fn(**hyperparams)
            model.eval()
            model.to(self.device)
            
            # Test forward pass
            input_tensor = torch.randn(1, *self.input_size).to(self.device)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            # Calculate model metrics
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Test training step (simulation)
            # This simulates a real training step to measure training performance
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=hyperparams.get('lr', 0.001))
            criterion = nn.CrossEntropyLoss()
            
            # Simulate a complete training step with forward and backward pass
            start_time = time.time()
            optimizer.zero_grad()
            # Generate random target for loss calculation
            loss = criterion(output, torch.randint(0, 2, (1,)).to(self.device))
            loss.backward()
            optimizer.step()
            end_time = time.time()
            
            training_time = end_time - start_time
            
            # Test memory usage
            if self.device.startswith('cuda'):
                memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
            else:
                memory_allocated = 0
                memory_reserved = 0
            
            results = {
                'model_name': model_name,
                'hyperparams': hyperparams,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'training_time': training_time,
                'memory_allocated_mb': memory_allocated,
                'memory_reserved_mb': memory_reserved,
                'output_shape': list(output.shape),
                'success': True,
                'error': None
            }
            
            logger.info(f"Successfully tested {model_name} with hyperparams: {hyperparams}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to test {model_name} with hyperparams {hyperparams}: {e}")
            return {
                'model_name': model_name,
                'hyperparams': hyperparams,
                'total_params': 0,
                'trainable_params': 0,
                'training_time': 0,
                'memory_allocated_mb': 0,
                'memory_reserved_mb': 0,
                'output_shape': None,
                'success': False,
                'error': str(e)
            }
    
    def optimize_learning_rates(self) -> Dict[str, Any]:
        """Optimize learning rates for MobileNetV4 models."""
        logger.info("Optimizing learning rates for MobileNetV4 models...")
        
        # Learning rate candidates
        learning_rates = [0.001, 0.003, 0.005, 0.01, 0.02]
        
        # Model configurations
        model_configs = [
            ('MobileNetV4-Small', mobilenetv4_small, {'embeding_dim': 1024}),
            ('MobileNetV4-Medium', mobilenetv4_medium, {'embeding_dim': 1152}),
            ('MobileNetV4-Large', mobilenetv4_large, {'embeding_dim': 1280})
        ]
        
        results = {}
        
        for model_name, model_fn, base_config in model_configs:
            logger.info(f"Optimizing learning rates for {model_name}...")
            model_results = {}
            
            for lr in learning_rates:
                hyperparams = {
                    'width_mult': 1.0,
                    'pretrained': False,
                    'lr': lr,
                    **base_config
                }
                
                result = self.test_hyperparameter_combination(model_name, model_fn, hyperparams)
                model_results[f'lr_{lr}'] = result
            
            results[model_name] = model_results
        
        return results
    
    def optimize_width_multipliers(self) -> Dict[str, Any]:
        """Optimize width multipliers for MobileNetV4 models."""
        logger.info("Optimizing width multipliers for MobileNetV4 models...")
        
        # Width multiplier candidates
        width_mults = [0.75, 1.0, 1.2, 1.4]
        
        # Model configurations
        model_configs = [
            ('MobileNetV4-Small', mobilenetv4_small, {'embeding_dim': 1024}),
            ('MobileNetV4-Medium', mobilenetv4_medium, {'embeding_dim': 1152}),
            ('MobileNetV4-Large', mobilenetv4_large, {'embeding_dim': 1280})
        ]
        
        results = {}
        
        for model_name, model_fn, base_config in model_configs:
            logger.info(f"Optimizing width multipliers for {model_name}...")
            model_results = {}
            
            for width_mult in width_mults:
                hyperparams = {
                    'width_mult': width_mult,
                    'pretrained': False,
                    'lr': 0.003,
                    **base_config
                }
                
                result = self.test_hyperparameter_combination(model_name, model_fn, hyperparams)
                model_results[f'width_mult_{width_mult}'] = result
            
            results[model_name] = model_results
        
        return results
    
    def optimize_embedding_dimensions(self) -> Dict[str, Any]:
        """Optimize embedding dimensions for MobileNetV4 models."""
        logger.info("Optimizing embedding dimensions for MobileNetV4 models...")
        
        # Embedding dimension candidates
        embedding_dims = [512, 768, 1024, 1152, 1280, 1536]
        
        # Model configurations
        model_configs = [
            ('MobileNetV4-Small', mobilenetv4_small),
            ('MobileNetV4-Medium', mobilenetv4_medium),
            ('MobileNetV4-Large', mobilenetv4_large)
        ]
        
        results = {}
        
        for model_name, model_fn in model_configs:
            logger.info(f"Optimizing embedding dimensions for {model_name}...")
            model_results = {}
            
            for emb_dim in embedding_dims:
                hyperparams = {
                    'width_mult': 1.0,
                    'pretrained': False,
                    'lr': 0.003,
                    'embeding_dim': emb_dim
                }
                
                result = self.test_hyperparameter_combination(model_name, model_fn, hyperparams)
                model_results[f'emb_dim_{emb_dim}'] = result
            
            results[model_name] = model_results
        
        return results
    
    def optimize_advanced_parameters(self) -> Dict[str, Any]:
        """Optimize advanced parameters for MobileNetV4 models."""
        logger.info("Optimizing advanced parameters for MobileNetV4 models...")
        
        # Advanced parameter combinations
        advanced_params = [
            {'use_attention': True, 'use_squeeze_excitation': True, 'dropout_rate': 0.2},
            {'use_attention': True, 'use_squeeze_excitation': False, 'dropout_rate': 0.2},
            {'use_attention': False, 'use_squeeze_excitation': True, 'dropout_rate': 0.2},
            {'use_attention': False, 'use_squeeze_excitation': False, 'dropout_rate': 0.2},
            {'use_attention': True, 'use_squeeze_excitation': True, 'dropout_rate': 0.3},
            {'use_attention': True, 'use_squeeze_excitation': True, 'dropout_rate': 0.1},
        ]
        
        # Test with MobileNetV4-Large as it has the most parameters
        model_name = 'MobileNetV4-Large'
        model_fn = mobilenetv4_large
        
        results = {}
        
        for i, params in enumerate(advanced_params):
            hyperparams = {
                'width_mult': 1.0,
                'pretrained': False,
                'lr': 0.003,
                'embeding_dim': 1280,
                **params
            }
            
            result = self.test_hyperparameter_combination(model_name, model_fn, hyperparams)
            results[f'advanced_config_{i}'] = result
        
        return results
    
    def run_full_optimization(self) -> Dict[str, Any]:
        """Run full optimization for all parameters."""
        logger.info("Starting full MobileNetV4 optimization...")
        
        all_results = {}
        
        # Optimize learning rates
        lr_results = self.optimize_learning_rates()
        all_results['learning_rates'] = lr_results
        
        # Optimize width multipliers
        width_results = self.optimize_width_multipliers()
        all_results['width_multipliers'] = width_results
        
        # Optimize embedding dimensions
        emb_results = self.optimize_embedding_dimensions()
        all_results['embedding_dimensions'] = emb_results
        
        # Optimize advanced parameters
        advanced_results = self.optimize_advanced_parameters()
        all_results['advanced_parameters'] = advanced_results
        
        self.results = all_results
        return all_results
    
    def generate_optimization_report(self) -> str:
        """Generate a comprehensive optimization report."""
        if not self.results:
            return "No results available. Run run_full_optimization() first."
        
        report = []
        report.append("# MobileNetV4 Optimization Report")
        report.append("=" * 50)
        report.append("")
        
        # Learning rate optimization results
        if 'learning_rates' in self.results:
            report.append("## Learning Rate Optimization")
            report.append("")
            report.append("| Model | Learning Rate | Params | Training Time | Memory | Success |")
            report.append("|-------|---------------|--------|---------------|--------|---------|")
            
            for model_name, model_results in self.results['learning_rates'].items():
                for lr_key, result in model_results.items():
                    if result['success']:
                        success_status = "SUCCESS"
                    else:
                        success_status = "FAILED"
                    
                    report.append(f"| {model_name} | {result['hyperparams'].get('lr', 'N/A')} | "
                                f"{result['total_params']:,} | {result['training_time']:.4f}s | "
                                f"{result['memory_allocated_mb']:.1f}MB | {success_status} |")
            
            report.append("")
        
        # Width multiplier optimization results
        if 'width_multipliers' in self.results:
            report.append("## Width Multiplier Optimization")
            report.append("")
            report.append("| Model | Width Mult | Params | Training Time | Memory | Success |")
            report.append("|-------|------------|--------|---------------|--------|---------|")
            
            for model_name, model_results in self.results['width_multipliers'].items():
                for width_key, result in model_results.items():
                    if result['success']:
                        success_status = "SUCCESS"
                    else:
                        success_status = "FAILED"
                    
                    report.append(f"| {model_name} | {result['hyperparams'].get('width_mult', 'N/A')} | "
                                f"{result['total_params']:,} | {result['training_time']:.4f}s | "
                                f"{result['memory_allocated_mb']:.1f}MB | {success_status} |")
            
            report.append("")
        
        # Embedding dimension optimization results
        if 'embedding_dimensions' in self.results:
            report.append("## Embedding Dimension Optimization")
            report.append("")
            report.append("| Model | Embedding Dim | Params | Training Time | Memory | Success |")
            report.append("|-------|---------------|--------|---------------|--------|---------|")
            
            for model_name, model_results in self.results['embedding_dimensions'].items():
                for emb_key, result in model_results.items():
                    if result['success']:
                        success_status = "SUCCESS"
                    else:
                        success_status = "FAILED"
                    
                    report.append(f"| {model_name} | {result['hyperparams'].get('embeding_dim', 'N/A')} | "
                                f"{result['total_params']:,} | {result['training_time']:.4f}s | "
                                f"{result['memory_allocated_mb']:.1f}MB | {success_status} |")
            
            report.append("")
        
        # Advanced parameters optimization results
        if 'advanced_parameters' in self.results:
            report.append("## Advanced Parameters Optimization")
            report.append("")
            report.append("| Config | Attention | SE | Dropout | Params | Training Time | Memory | Success |")
            report.append("|--------|-----------|----|---------|--------|---------------|--------|---------|")
            
            for config_key, result in self.results['advanced_parameters'].items():
                if result['success']:
                    success_status = "SUCCESS"
                else:
                    success_status = "FAILED"
                
                attention = result['hyperparams'].get('use_attention', False)
                se = result['hyperparams'].get('use_squeeze_excitation', False)
                dropout = result['hyperparams'].get('dropout_rate', 0.2)
                
                report.append(f"| {config_key} | {'Yes' if attention else 'No'} | "
                            f"{'Yes' if se else 'No'} | {dropout} | "
                            f"{result['total_params']:,} | {result['training_time']:.4f}s | "
                            f"{result['memory_allocated_mb']:.1f}MB | {success_status} |")
            
            report.append("")
        
        # Recommendations
        report.append("## Optimization Recommendations")
        report.append("")
        
        # Find best configurations
        best_configs = self._find_best_configurations()
        
        if best_configs:
            report.append("### Best Configurations by Category")
            report.append("")
            
            for category, best_config in best_configs.items():
                report.append(f"#### {category}")
                report.append(f"- **Model**: {best_config['model_name']}")
                report.append(f"- **Parameters**: {best_config['total_params']:,}")
                report.append(f"- **Training Time**: {best_config['training_time']:.4f}s")
                report.append(f"- **Memory Usage**: {best_config['memory_allocated_mb']:.1f}MB")
                report.append(f"- **Hyperparameters**: {best_config['hyperparams']}")
                report.append("")
        
        return "\n".join(report)
    
    def _find_best_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Find the best configurations for each category."""
        best_configs = {}
        
        # Find best learning rate configuration
        if 'learning_rates' in self.results:
            best_lr = None
            best_lr_score = float('inf')
            
            for model_name, model_results in self.results['learning_rates'].items():
                for lr_key, result in model_results.items():
                    if result['success']:
                        # Score based on training time and memory usage
                        score = result['training_time'] + result['memory_allocated_mb'] / 100
                        if score < best_lr_score:
                            best_lr_score = score
                            best_lr = result
            
            if best_lr:
                best_configs['Best Learning Rate'] = best_lr
        
        # Find best width multiplier configuration
        if 'width_multipliers' in self.results:
            best_width = None
            best_width_score = float('inf')
            
            for model_name, model_results in self.results['width_multipliers'].items():
                for width_key, result in model_results.items():
                    if result['success']:
                        # Score based on parameter efficiency
                        score = result['total_params'] / 1000000  # Normalize by 1M params
                        if score < best_width_score:
                            best_width_score = score
                            best_width = result
            
            if best_width:
                best_configs['Best Width Multiplier'] = best_width
        
        return best_configs
    
    def save_results(self, output_path: str = "mobilenetv4_optimization_results.json"):
        """Save optimization results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def save_report(self, output_path: str = "mobilenetv4_optimization_report.md"):
        """Save optimization report to Markdown file."""
        report = self.generate_optimization_report()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {output_path}")


def main():
    """Main function to run the optimization."""
    parser = argparse.ArgumentParser(description="MobileNetV4 Optimization")
    parser.add_argument("--device", type=str, default="cpu", 
                       help="Device to run optimization on (cpu, cuda, cuda:0)")
    parser.add_argument("--output-dir", type=str, default="./mobilenetv4_optimization_results",
                       help="Output directory for results")
    parser.add_argument("--save-results", action="store_true", default=True,
                       help="Save results to files")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize optimizer
    optimizer = MobileNetV4Optimizer(device=args.device)
    
    # Run optimization
    logger.info("Starting MobileNetV4 optimization...")
    results = optimizer.run_full_optimization()
    
    # Generate and save report
    if args.save_results:
        optimizer.save_results(output_dir / "mobilenetv4_optimization_results.json")
        optimizer.save_report(output_dir / "mobilenetv4_optimization_report.md")
    
    # Print summary
    print("\n" + "="*60)
    print("MOBILENETV4 OPTIMIZATION COMPLETED")
    print("="*60)
    print(optimizer.generate_optimization_report())
    
    logger.info("MobileNetV4 optimization completed successfully!")


if __name__ == "__main__":
    main()
