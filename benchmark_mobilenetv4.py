#!/usr/bin/env python3
"""
MobileNetV4 vs MobileNetV3 Performance Benchmarking Script

This script benchmarks MobileNetV4 performance against MobileNetV3 for face anti-spoofing tasks.
It tests all variants (small, medium, large) and provides comprehensive performance comparisons.
"""

import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Any

# Import models
from models import (
    mobilenetv3_small, mobilenetv3_large,
    mobilenetv4_small, mobilenetv4_medium, mobilenetv4_large
)

# Import utilities
from utils.error_handling import validate_device, safe_model_forward
from utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(log_level="INFO", log_to_file=True, log_to_console=True, structured=True)
logger = get_logger(__name__)


class ModelBenchmark:
    """Comprehensive model benchmarking for MobileNetV3 vs MobileNetV4."""
    
    def __init__(self, device: str = "cpu", batch_sizes: List[int] = [1, 8, 16, 32]):
        """
        Initialize the benchmark.
        
        Args:
            device: Device to run benchmarks on
            batch_sizes: List of batch sizes to test
        """
        self.device = validate_device(device)
        self.batch_sizes = batch_sizes
        self.results = {}
        
        # Test input size (224x224 RGB)
        self.input_size = (3, 224, 224)
        
        logger.info(f"Initialized benchmark on device: {self.device}")
        logger.info(f"Testing batch sizes: {self.batch_sizes}")
    
    def benchmark_model(self, model: nn.Module, model_name: str, batch_size: int, 
                       num_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark a single model configuration.
        
        Args:
            model: PyTorch model to benchmark
            model_name: Name of the model
            batch_size: Batch size for testing
            num_iterations: Number of iterations to run
            
        Returns:
            Dictionary with benchmark results
        """
        model.eval()
        model.to(self.device)
        
        # Create test input
        input_tensor = torch.randn(batch_size, *self.input_size).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # Benchmark inference time
        torch.cuda.synchronize() if self.device.startswith('cuda') else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                output = model(input_tensor)
        
        torch.cuda.synchronize() if self.device.startswith('cuda') else None
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time_per_batch = total_time / num_iterations
        throughput = batch_size / avg_time_per_batch
        
        # Calculate model size
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate memory usage
        if self.device.startswith('cuda'):
            memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
        else:
            memory_allocated = 0
            memory_reserved = 0
        
        results = {
            'model_name': model_name,
            'batch_size': batch_size,
            'avg_time_per_batch': avg_time_per_batch,
            'throughput': throughput,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'memory_allocated_mb': memory_allocated,
            'memory_reserved_mb': memory_reserved,
            'num_iterations': num_iterations
        }
        
        logger.info(f"Benchmarked {model_name} (batch_size={batch_size}): "
                   f"{avg_time_per_batch:.4f}s/batch, {throughput:.2f} samples/s")
        
        return results
    
    def benchmark_all_models(self) -> Dict[str, Any]:
        """Benchmark all model variants."""
        logger.info("Starting comprehensive model benchmarking...")
        
        # Define model configurations
        model_configs = [
            # MobileNetV3 models
            {
                'name': 'MobileNetV3-Small',
                'model_fn': lambda: mobilenetv3_small(width_mult=1.0, pretrained=False, embeding_dim=1024),
                'embedding_dim': 1024
            },
            {
                'name': 'MobileNetV3-Large',
                'model_fn': lambda: mobilenetv3_large(width_mult=1.0, pretrained=False, embeding_dim=1280),
                'embedding_dim': 1280
            },
            # MobileNetV4 models
            {
                'name': 'MobileNetV4-Small',
                'model_fn': lambda: mobilenetv4_small(width_mult=1.0, pretrained=False, embeding_dim=1024),
                'embedding_dim': 1024
            },
            {
                'name': 'MobileNetV4-Medium',
                'model_fn': lambda: mobilenetv4_medium(width_mult=1.0, pretrained=False, embeding_dim=1152),
                'embedding_dim': 1152
            },
            {
                'name': 'MobileNetV4-Large',
                'model_fn': lambda: mobilenetv4_large(width_mult=1.0, pretrained=False, embeding_dim=1280),
                'embedding_dim': 1280
            }
        ]
        
        all_results = {}
        
        for config in model_configs:
            model_name = config['name']
            logger.info(f"Benchmarking {model_name}...")
            
            try:
                # Create model
                model = config['model_fn']()
                
                # Benchmark for each batch size
                model_results = {}
                for batch_size in self.batch_sizes:
                    try:
                        results = self.benchmark_model(model, model_name, batch_size)
                        model_results[f'batch_{batch_size}'] = results
                    except Exception as e:
                        logger.error(f"Failed to benchmark {model_name} with batch_size={batch_size}: {e}")
                        model_results[f'batch_{batch_size}'] = {'error': str(e)}
                
                all_results[model_name] = model_results
                
                # Clean up
                del model
                torch.cuda.empty_cache() if self.device.startswith('cuda') else None
                
            except Exception as e:
                logger.error(f"Failed to create model {model_name}: {e}")
                all_results[model_name] = {'error': str(e)}
        
        self.results = all_results
        return all_results
    
    def generate_comparison_report(self) -> str:
        """Generate a comprehensive comparison report."""
        if not self.results:
            return "No results available. Run benchmark_all_models() first."
        
        report = []
        report.append("# MobileNetV4 vs MobileNetV3 Performance Comparison")
        report.append("=" * 60)
        report.append("")
        
        # Summary table
        report.append("## Performance Summary")
        report.append("")
        report.append("| Model | Batch Size | Avg Time (s) | Throughput (samples/s) | Params | Memory (MB) |")
        report.append("|-------|------------|-------------|------------------------|--------|------------|")
        
        for model_name, model_results in self.results.items():
            if 'error' in model_results:
                report.append(f"| {model_name} | - | ERROR | - | - | - |")
                continue
                
            for batch_key, results in model_results.items():
                if 'error' in results:
                    report.append(f"| {model_name} | {results.get('batch_size', 'N/A')} | ERROR | - | - | - |")
                    continue
                
                report.append(f"| {model_name} | {results['batch_size']} | "
                            f"{results['avg_time_per_batch']:.4f} | "
                            f"{results['throughput']:.2f} | "
                            f"{results['total_params']:,} | "
                            f"{results['memory_allocated_mb']:.1f} |")
        
        report.append("")
        
        # Detailed analysis
        report.append("## Detailed Analysis")
        report.append("")
        
        # Find best performing models
        best_throughput = {}
        best_memory = {}
        
        for model_name, model_results in self.results.items():
            if 'error' in model_results:
                continue
                
            for batch_key, results in model_results.items():
                if 'error' in results:
                    continue
                
                batch_size = results['batch_size']
                if batch_size not in best_throughput or results['throughput'] > best_throughput[batch_size]['throughput']:
                    best_throughput[batch_size] = {'model': model_name, 'throughput': results['throughput']}
                
                if batch_size not in best_memory or results['memory_allocated_mb'] < best_memory[batch_size]['memory']:
                    best_memory[batch_size] = {'model': model_name, 'memory': results['memory_allocated_mb']}
        
        # Best throughput analysis
        report.append("### Best Throughput by Batch Size")
        for batch_size in sorted(best_throughput.keys()):
            best = best_throughput[batch_size]
            report.append(f"- Batch Size {batch_size}: {best['model']} ({best['throughput']:.2f} samples/s)")
        
        report.append("")
        
        # Best memory analysis
        report.append("### Most Memory Efficient by Batch Size")
        for batch_size in sorted(best_memory.keys()):
            best = best_memory[batch_size]
            report.append(f"- Batch Size {batch_size}: {best['model']} ({best['memory']:.1f} MB)")
        
        report.append("")
        
        # MobileNetV4 vs MobileNetV3 comparison
        report.append("### MobileNetV4 vs MobileNetV3 Comparison")
        report.append("")
        
        # Compare small models
        if 'MobileNetV3-Small' in self.results and 'MobileNetV4-Small' in self.results:
            report.append("#### Small Models Comparison")
            self._compare_models(report, 'MobileNetV3-Small', 'MobileNetV4-Small')
        
        # Compare large models
        if 'MobileNetV3-Large' in self.results and 'MobileNetV4-Large' in self.results:
            report.append("#### Large Models Comparison")
            self._compare_models(report, 'MobileNetV3-Large', 'MobileNetV4-Large')
        
        return "\n".join(report)
    
    def _compare_models(self, report: List[str], model1: str, model2: str):
        """Compare two models and add to report."""
        if model1 not in self.results or model2 not in self.results:
            return
        
        report.append(f"| Metric | {model1} | {model2} | Improvement |")
        report.append("|--------|-----------|-----------|-------------|")
        
        # Compare for each batch size
        for batch_size in self.batch_sizes:
            batch_key = f'batch_{batch_size}'
            
            if (batch_key in self.results[model1] and batch_key in self.results[model2] and
                'error' not in self.results[model1][batch_key] and 'error' not in self.results[model2][batch_key]):
                
                r1 = self.results[model1][batch_key]
                r2 = self.results[model2][batch_key]
                
                # Throughput comparison
                throughput_improvement = ((r2['throughput'] - r1['throughput']) / r1['throughput']) * 100
                report.append(f"| Throughput (batch={batch_size}) | {r1['throughput']:.2f} | {r2['throughput']:.2f} | {throughput_improvement:+.1f}% |")
                
                # Memory comparison
                memory_improvement = ((r1['memory_allocated_mb'] - r2['memory_allocated_mb']) / r1['memory_allocated_mb']) * 100
                report.append(f"| Memory (batch={batch_size}) | {r1['memory_allocated_mb']:.1f}MB | {r2['memory_allocated_mb']:.1f}MB | {memory_improvement:+.1f}% |")
        
        report.append("")
    
    def save_results(self, output_path: str = "benchmark_results.json"):
        """Save benchmark results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def save_report(self, output_path: str = "benchmark_report.md"):
        """Save benchmark report to Markdown file."""
        report = self.generate_comparison_report()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {output_path}")


def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="MobileNetV4 vs MobileNetV3 Benchmarking")
    parser.add_argument("--device", type=str, default="cpu", 
                       help="Device to run benchmarks on (cpu, cuda, cuda:0)")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 16, 32],
                       help="Batch sizes to test")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--save-results", action="store_true", default=True,
                       help="Save results to files")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize benchmark
    benchmark = ModelBenchmark(device=args.device, batch_sizes=args.batch_sizes)
    
    # Run benchmark
    logger.info("Starting MobileNetV4 vs MobileNetV3 benchmarking...")
    results = benchmark.benchmark_all_models()
    
    # Generate and save report
    if args.save_results:
        benchmark.save_results(output_dir / "benchmark_results.json")
        benchmark.save_report(output_dir / "benchmark_report.md")
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK COMPLETED")
    print("="*60)
    print(benchmark.generate_comparison_report())
    
    logger.info("Benchmarking completed successfully!")


if __name__ == "__main__":
    main()
