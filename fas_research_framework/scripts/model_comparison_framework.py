#!/usr/bin/env python3
"""
FAS-Research-Framework Model Comparison Framework

This framework provides comprehensive comparison capabilities for different model architectures
in face anti-spoofing tasks, including performance metrics, efficiency analysis, and visualization.
Part of the FAS-Research-Framework for advanced face anti-spoofing research.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd

# Import models
from models import (
    mobilenetv2, mobilenetv3_large, mobilenetv3_small,
    mobilenetv4_large, mobilenetv4_medium, mobilenetv4_small,
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    vit_tiny, vit_small, vit_base, vit_large,
    resnet18, resnet34, resnet50, resnet101
)

# Import utilities
from utils.error_handling import validate_device, safe_model_forward
from utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(log_level="INFO", log_to_file=True, log_to_console=True, structured=True)
logger = get_logger(__name__)


class ModelComparisonFramework:
    """
    Comprehensive model comparison framework for face anti-spoofing.
    
    This class provides functionality to benchmark and compare different model architectures
    including MobileNetV2, MobileNetV3, MobileNetV4, EfficientNet, Vision Transformer, and ResNet
    variants for face anti-spoofing tasks.
    """
    
    def __init__(self, device: str = "cpu", input_size: Tuple[int, int, int] = (3, 224, 224)):
        """
        Initialize the model comparison framework.
        
        Args:
            device: Device to run comparisons on
            input_size: Input tensor size (C, H, W)
        """
        self.device = validate_device(device)
        self.input_size = input_size
        self.results = {}
        
        # Define model configurations for comparison
        # Each model configuration includes the model function and parameters
        self.model_configs = {
            # MobileNetV2
            'MobileNetV2': {
                'model_fn': mobilenetv2,
                'params': {'width_mult': 1.0, 'pretrained': False, 'embeding_dim': 1280}
            },
            
            # MobileNetV3
            'MobileNetV3-Small': {
                'model_fn': mobilenetv3_small,
                'params': {'width_mult': 1.0, 'pretrained': False, 'embeding_dim': 1024}
            },
            'MobileNetV3-Large': {
                'model_fn': mobilenetv3_large,
                'params': {'width_mult': 1.0, 'pretrained': False, 'embeding_dim': 1280}
            },
            
            # MobileNetV4
            'MobileNetV4-Small': {
                'model_fn': mobilenetv4_small,
                'params': {'width_mult': 1.0, 'pretrained': False, 'embeding_dim': 1024}
            },
            'MobileNetV4-Medium': {
                'model_fn': mobilenetv4_medium,
                'params': {'width_mult': 1.0, 'pretrained': False, 'embeding_dim': 1152}
            },
            'MobileNetV4-Large': {
                'model_fn': mobilenetv4_large,
                'params': {'width_mult': 1.0, 'pretrained': False, 'embeding_dim': 1280}
            },
            
            # EfficientNet
            'EfficientNet-B0': {
                'model_fn': efficientnet_b0,
                'params': {'pretrained': False, 'embeding_dim': 1280}
            },
            'EfficientNet-B1': {
                'model_fn': efficientnet_b1,
                'params': {'pretrained': False, 'embeding_dim': 1280}
            },
            'EfficientNet-B2': {
                'model_fn': efficientnet_b2,
                'params': {'pretrained': False, 'embeding_dim': 1280}
            },
            'EfficientNet-B3': {
                'model_fn': efficientnet_b3,
                'params': {'pretrained': False, 'embeding_dim': 1280}
            },
            
            # Vision Transformer
            'ViT-Tiny': {
                'model_fn': vit_tiny,
                'params': {'pretrained': False, 'embeding_dim': 768}
            },
            'ViT-Small': {
                'model_fn': vit_small,
                'params': {'pretrained': False, 'embeding_dim': 768}
            },
            'ViT-Base': {
                'model_fn': vit_base,
                'params': {'pretrained': False, 'embeding_dim': 768}
            },
            'ViT-Large': {
                'model_fn': vit_large,
                'params': {'pretrained': False, 'embeding_dim': 1024}
            },
            
            # ResNet
            'ResNet-18': {
                'model_fn': resnet18,
                'params': {'pretrained': False, 'embeding_dim': 512}
            },
            'ResNet-34': {
                'model_fn': resnet34,
                'params': {'pretrained': False, 'embeding_dim': 512}
            },
            'ResNet-50': {
                'model_fn': resnet50,
                'params': {'pretrained': False, 'embeding_dim': 2048}
            },
            'ResNet-101': {
                'model_fn': resnet101,
                'params': {'pretrained': False, 'embeding_dim': 2048}
            },
        }
        
        logger.info(f"Initialized model comparison framework on device: {self.device}")
        logger.info(f"Input size: {self.input_size}")
        logger.info(f"Available models: {len(self.model_configs)}")
    
    def benchmark_model(self, model_name: str, model_fn, params: Dict[str, Any], 
                       batch_sizes: List[int] = [1, 8, 16, 32], num_iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark a single model with comprehensive performance metrics.
        
        This method tests model performance across different batch sizes, measuring:
        - Inference time and throughput
        - Memory usage (GPU only)
        - Model parameter count
        - Forward pass success rate
        
        Args:
            model_name: Name of the model for identification
            model_fn: Model creation function
            params: Model parameters for initialization
            batch_sizes: List of batch sizes to test (default: [1, 8, 16, 32])
            num_iterations: Number of iterations for timing accuracy (default: 100)
            
        Returns:
            Dictionary containing comprehensive benchmark results including:
            - Model metrics (parameters, success status)
            - Performance metrics (throughput, memory usage)
            - Batch-specific results for each tested batch size
        """
        try:
            # Create model
            model = model_fn(**params)
            model.eval()
            model.to(self.device)
            
            # Calculate model size metrics
            # Total parameters include all model parameters
            total_params = sum(p.numel() for p in model.parameters())
            # Trainable parameters exclude frozen layers
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Test different batch sizes
            batch_results = {}
            
            for batch_size in batch_sizes:
                try:
                    # Create test input
                    input_tensor = torch.randn(batch_size, *self.input_size).to(self.device)
                    
                    # Warmup phase to ensure consistent timing
                    # Run model a few times to initialize CUDA kernels and optimize memory
                    with torch.no_grad():
                        for _ in range(10):
                            _ = model(input_tensor)
                    
                    # Benchmark inference time with precise timing
                    # Synchronize CUDA operations for accurate GPU timing
                    torch.cuda.synchronize() if self.device.startswith('cuda') else None
                    start_time = time.time()
                    
                    # Run multiple iterations for statistical accuracy
                    with torch.no_grad():
                        for _ in range(num_iterations):
                            output = model(input_tensor)
                    
                    # Ensure all CUDA operations complete before timing
                    torch.cuda.synchronize() if self.device.startswith('cuda') else None
                    end_time = time.time()
                    
                    # Calculate metrics
                    total_time = end_time - start_time
                    avg_time_per_batch = total_time / num_iterations
                    throughput = batch_size / avg_time_per_batch
                    
                    # Calculate memory usage (GPU only)
                    # Memory metrics are only available on CUDA devices
                    if self.device.startswith('cuda'):
                        # Allocated memory: actually used by tensors
                        memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                        # Reserved memory: total memory reserved by PyTorch
                        memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
                    else:
                        # CPU memory tracking is not implemented
                        memory_allocated = 0
                        memory_reserved = 0
                    
                    batch_results[f'batch_{batch_size}'] = {
                        'batch_size': batch_size,
                        'avg_time_per_batch': avg_time_per_batch,
                        'throughput': throughput,
                        'memory_allocated_mb': memory_allocated,
                        'memory_reserved_mb': memory_reserved,
                        'num_iterations': num_iterations,
                        'success': True
                    }
                    
                    logger.info(f"Benchmarked {model_name} (batch_size={batch_size}): "
                               f"{avg_time_per_batch:.4f}s/batch, {throughput:.2f} samples/s")
                    
                except Exception as e:
                    logger.error(f"Failed to benchmark {model_name} with batch_size={batch_size}: {e}")
                    batch_results[f'batch_{batch_size}'] = {
                        'batch_size': batch_size,
                        'success': False,
                        'error': str(e)
                    }
            
            # Clean up
            del model
            torch.cuda.empty_cache() if self.device.startswith('cuda') else None
            
            return {
                'model_name': model_name,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'batch_results': batch_results,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Failed to create model {model_name}: {e}")
            return {
                'model_name': model_name,
                'total_params': 0,
                'trainable_params': 0,
                'batch_results': {},
                'success': False,
                'error': str(e)
            }
    
    def compare_all_models(self, batch_sizes: List[int] = [1, 8, 16, 32]) -> Dict[str, Any]:
        """
        Compare all available models.
        
        Args:
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary with comparison results
        """
        logger.info("Starting comprehensive model comparison...")
        
        all_results = {}
        
        for model_name, config in self.model_configs.items():
            logger.info(f"Benchmarking {model_name}...")
            
            model_fn = config['model_fn']
            params = config['params']
            
            results = self.benchmark_model(model_name, model_fn, params, batch_sizes)
            all_results[model_name] = results
        
        self.results = all_results
        return all_results
    
    def generate_comparison_report(self) -> str:
        """Generate a comprehensive comparison report."""
        if not self.results:
            return "No results available. Run compare_all_models() first."
        
        report = []
        report.append("# Model Comparison Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary table
        report.append("## Model Summary")
        report.append("")
        report.append("| Model | Parameters | Success | Best Throughput | Best Memory |")
        report.append("|-------|------------|---------|-----------------|-------------|")
        
        for model_name, results in self.results.items():
            if not results['success']:
                report.append(f"| {model_name} | - | FAILED | - | - |")
                continue
            
            # Find best throughput and memory usage
            best_throughput = 0
            best_memory = float('inf')
            
            for batch_key, batch_result in results['batch_results'].items():
                if batch_result['success']:
                    best_throughput = max(best_throughput, batch_result['throughput'])
                    best_memory = min(best_memory, batch_result['memory_allocated_mb'])
            
            report.append(f"| {model_name} | {results['total_params']:,} | SUCCESS | "
                        f"{best_throughput:.2f} | {best_memory:.1f}MB |")
        
        report.append("")
        
        # Detailed results by batch size
        report.append("## Detailed Results by Batch Size")
        report.append("")
        
        for batch_size in [1, 8, 16, 32]:
            report.append(f"### Batch Size {batch_size}")
            report.append("")
            report.append("| Model | Time (s) | Throughput | Memory (MB) |")
            report.append("|-------|----------|------------|-------------|")
            
            for model_name, results in self.results.items():
                if not results['success']:
                    continue
                
                batch_key = f'batch_{batch_size}'
                if batch_key in results['batch_results'] and results['batch_results'][batch_key]['success']:
                    batch_result = results['batch_results'][batch_key]
                    report.append(f"| {model_name} | {batch_result['avg_time_per_batch']:.4f} | "
                                f"{batch_result['throughput']:.2f} | {batch_result['memory_allocated_mb']:.1f} |")
            
            report.append("")
        
        # Performance rankings
        report.append("## Performance Rankings")
        report.append("")
        
        # Rank by throughput
        throughput_rankings = []
        for model_name, results in self.results.items():
            if not results['success']:
                continue
            
            max_throughput = 0
            for batch_result in results['batch_results'].values():
                if batch_result['success']:
                    max_throughput = max(max_throughput, batch_result['throughput'])
            
            if max_throughput > 0:
                throughput_rankings.append((model_name, max_throughput))
        
        throughput_rankings.sort(key=lambda x: x[1], reverse=True)
        
        report.append("### Top Models by Throughput")
        for i, (model_name, throughput) in enumerate(throughput_rankings[:10]):
            report.append(f"{i+1}. {model_name}: {throughput:.2f} samples/s")
        
        report.append("")
        
        # Rank by memory efficiency
        memory_rankings = []
        for model_name, results in self.results.items():
            if not results['success']:
                continue
            
            min_memory = float('inf')
            for batch_result in results['batch_results'].values():
                if batch_result['success']:
                    min_memory = min(min_memory, batch_result['memory_allocated_mb'])
            
            if min_memory < float('inf'):
                memory_rankings.append((model_name, min_memory))
        
        memory_rankings.sort(key=lambda x: x[1])
        
        report.append("### Most Memory Efficient Models")
        for i, (model_name, memory) in enumerate(memory_rankings[:10]):
            report.append(f"{i+1}. {model_name}: {memory:.1f}MB")
        
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        if throughput_rankings:
            best_throughput_model = throughput_rankings[0]
            report.append(f"**Best Overall Performance**: {best_throughput_model[0]} "
                         f"({best_throughput_model[1]:.2f} samples/s)")
        
        if memory_rankings:
            most_efficient_model = memory_rankings[0]
            report.append(f"**Most Memory Efficient**: {most_efficient_model[0]} "
                         f"({most_efficient_model[1]:.1f}MB)")
        
        report.append("")
        report.append("### Use Case Recommendations")
        report.append("")
        report.append("- **High Performance**: Choose models with highest throughput")
        report.append("- **Memory Constrained**: Choose models with lowest memory usage")
        report.append("- **Balanced**: Choose models with good throughput/memory ratio")
        report.append("- **Production**: Consider inference speed and model size")
        
        return "\n".join(report)
    
    def create_visualizations(self, output_dir: str = "./model_comparison_plots"):
        """Create visualization plots for model comparison."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for plotting
        plot_data = []
        
        for model_name, results in self.results.items():
            if not results['success']:
                continue
            
            for batch_key, batch_result in results['batch_results'].items():
                if batch_result['success']:
                    plot_data.append({
                        'Model': model_name,
                        'Batch Size': batch_result['batch_size'],
                        'Throughput': batch_result['throughput'],
                        'Memory (MB)': batch_result['memory_allocated_mb'],
                        'Time (s)': batch_result['avg_time_per_batch'],
                        'Parameters': results['total_params']
                    })
        
        if not plot_data:
            logger.warning("No data available for visualization")
            return
        
        df = pd.DataFrame(plot_data)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Throughput comparison
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        sns.barplot(data=df, x='Model', y='Throughput', hue='Batch Size')
        plt.title('Model Throughput Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Throughput (samples/s)')
        
        # 2. Memory usage comparison
        plt.subplot(2, 2, 2)
        sns.barplot(data=df, x='Model', y='Memory (MB)', hue='Batch Size')
        plt.title('Model Memory Usage Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Memory (MB)')
        
        # 3. Parameter count vs throughput
        plt.subplot(2, 2, 3)
        sns.scatterplot(data=df, x='Parameters', y='Throughput', hue='Model', size='Batch Size')
        plt.title('Parameters vs Throughput')
        plt.xlabel('Number of Parameters')
        plt.ylabel('Throughput (samples/s)')
        
        # 4. Memory vs throughput
        plt.subplot(2, 2, 4)
        sns.scatterplot(data=df, x='Memory (MB)', y='Throughput', hue='Model', size='Batch Size')
        plt.title('Memory vs Throughput')
        plt.xlabel('Memory (MB)')
        plt.ylabel('Throughput (samples/s)')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Detailed throughput heatmap
        plt.figure(figsize=(12, 8))
        pivot_table = df.pivot_table(values='Throughput', index='Model', columns='Batch Size', aggfunc='mean')
        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title('Throughput Heatmap (samples/s)')
        plt.tight_layout()
        plt.savefig(output_dir / 'throughput_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Memory usage heatmap
        plt.figure(figsize=(12, 8))
        pivot_table = df.pivot_table(values='Memory (MB)', index='Model', columns='Batch Size', aggfunc='mean')
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='Blues')
        plt.title('Memory Usage Heatmap (MB)')
        plt.tight_layout()
        plt.savefig(output_dir / 'memory_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def save_results(self, output_path: str = "model_comparison_results.json"):
        """Save comparison results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def save_report(self, output_path: str = "model_comparison_report.md"):
        """Save comparison report to Markdown file."""
        report = self.generate_comparison_report()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {output_path}")


def main():
    """Main function to run model comparison."""
    parser = argparse.ArgumentParser(description="Model Comparison Framework")
    parser.add_argument("--device", type=str, default="cpu", 
                       help="Device to run comparison on (cpu, cuda, cuda:0)")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 16, 32],
                       help="Batch sizes to test")
    parser.add_argument("--output-dir", type=str, default="./model_comparison_results",
                       help="Output directory for results")
    parser.add_argument("--save-results", action="store_true", default=True,
                       help="Save results to files")
    parser.add_argument("--create-plots", action="store_true", default=True,
                       help="Create visualization plots")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize framework
    framework = ModelComparisonFramework(device=args.device)
    
    # Run comparison
    logger.info("Starting model comparison...")
    results = framework.compare_all_models(batch_sizes=args.batch_sizes)
    
    # Generate and save results
    if args.save_results:
        framework.save_results(output_dir / "model_comparison_results.json")
        framework.save_report(output_dir / "model_comparison_report.md")
    
    # Create visualizations
    if args.create_plots:
        framework.create_visualizations(output_dir / "plots")
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL COMPARISON COMPLETED")
    print("="*60)
    print(framework.generate_comparison_report())
    
    logger.info("Model comparison completed successfully!")


if __name__ == "__main__":
    main()
