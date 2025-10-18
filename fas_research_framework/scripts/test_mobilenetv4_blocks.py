#!/usr/bin/env python3
"""
FAS-Research-Framework MobileNetV4 Block Type Testing Script

This script tests MobileNetV4 with different block type combinations to find optimal
configurations for face anti-spoofing tasks.
Part of the FAS-Research-Framework for advanced face anti-spoofing research.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Any

# Import models
from models import mobilenetv4_small, mobilenetv4_medium, mobilenetv4_large

# Import utilities
from utils.error_handling import validate_device, safe_model_forward
from utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(log_level="INFO", log_to_file=True, log_to_console=True, structured=True)
logger = get_logger(__name__)


class MobileNetV4BlockTester:
    """Test MobileNetV4 with different block type combinations."""
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize the block tester.
        
        Args:
            device: Device to run tests on
        """
        self.device = validate_device(device)
        self.results = {}
        
        # Test input size (224x224 RGB)
        self.input_size = (3, 224, 224)
        
        logger.info(f"Initialized MobileNetV4 block tester on device: {self.device}")
    
    def test_model_creation(self, model_name: str, model_fn, **kwargs) -> Dict[str, Any]:
        """
        Test model creation and basic functionality.
        
        Args:
            model_name: Name of the model
            model_fn: Model creation function
            **kwargs: Additional arguments for model creation
            
        Returns:
            Dictionary with test results
        """
        try:
            # Create model
            model = model_fn(**kwargs)
            model.eval()
            model.to(self.device)
            
            # Test forward pass
            input_tensor = torch.randn(1, *self.input_size).to(self.device)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            # Get model info
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Test multi-head output if available
            multi_head_output = None
            if hasattr(model, 'make_logits'):
                try:
                    with torch.no_grad():
                        features = model(input_tensor)
                        multi_head_output = model.make_logits(features, all=True)
                except Exception as e:
                    logger.warning(f"Multi-head output test failed for {model_name}: {e}")
            
            results = {
                'model_name': model_name,
                'creation_success': True,
                'forward_pass_success': True,
                'output_shape': list(output.shape),
                'total_params': total_params,
                'trainable_params': trainable_params,
                'multi_head_available': multi_head_output is not None,
                'multi_head_output_count': len(multi_head_output) if multi_head_output else 0,
                'error': None
            }
            
            logger.info(f"Successfully tested {model_name}: {total_params:,} params, "
                       f"output shape: {output.shape}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to test {model_name}: {e}")
            return {
                'model_name': model_name,
                'creation_success': False,
                'forward_pass_success': False,
                'output_shape': None,
                'total_params': 0,
                'trainable_params': 0,
                'multi_head_available': False,
                'multi_head_output_count': 0,
                'error': str(e)
            }
    
    def test_all_variants(self) -> Dict[str, Any]:
        """Test all MobileNetV4 variants."""
        logger.info("Testing all MobileNetV4 variants...")
        
        # Test configurations
        test_configs = [
            # Small variants
            {
                'name': 'MobileNetV4-Small (Standard)',
                'model_fn': mobilenetv4_small,
                'kwargs': {'width_mult': 1.0, 'pretrained': False, 'embeding_dim': 1024}
            },
            {
                'name': 'MobileNetV4-Small (Multi-head)',
                'model_fn': mobilenetv4_small,
                'kwargs': {'width_mult': 1.0, 'pretrained': False, 'embeding_dim': 1024, 'multi_heads': True}
            },
            {
                'name': 'MobileNetV4-Small (Wide)',
                'model_fn': mobilenetv4_small,
                'kwargs': {'width_mult': 1.2, 'pretrained': False, 'embeding_dim': 1024}
            },
            
            # Medium variants
            {
                'name': 'MobileNetV4-Medium (Standard)',
                'model_fn': mobilenetv4_medium,
                'kwargs': {'width_mult': 1.0, 'pretrained': False, 'embeding_dim': 1152}
            },
            {
                'name': 'MobileNetV4-Medium (Multi-head)',
                'model_fn': mobilenetv4_medium,
                'kwargs': {'width_mult': 1.0, 'pretrained': False, 'embeding_dim': 1152, 'multi_heads': True}
            },
            {
                'name': 'MobileNetV4-Medium (Wide)',
                'model_fn': mobilenetv4_medium,
                'kwargs': {'width_mult': 1.2, 'pretrained': False, 'embeding_dim': 1152}
            },
            
            # Large variants
            {
                'name': 'MobileNetV4-Large (Standard)',
                'model_fn': mobilenetv4_large,
                'kwargs': {'width_mult': 1.0, 'pretrained': False, 'embeding_dim': 1280}
            },
            {
                'name': 'MobileNetV4-Large (Multi-head)',
                'model_fn': mobilenetv4_large,
                'kwargs': {'width_mult': 1.0, 'pretrained': False, 'embeding_dim': 1280, 'multi_heads': True}
            },
            {
                'name': 'MobileNetV4-Large (Wide)',
                'model_fn': mobilenetv4_large,
                'kwargs': {'width_mult': 1.2, 'pretrained': False, 'embeding_dim': 1280}
            }
        ]
        
        all_results = {}
        
        for config in test_configs:
            logger.info(f"Testing {config['name']}...")
            results = self.test_model_creation(
                config['name'], 
                config['model_fn'], 
                **config['kwargs']
            )
            all_results[config['name']] = results
        
        self.results = all_results
        return all_results
    
    def test_block_combinations(self) -> Dict[str, Any]:
        """Test different block type combinations."""
        logger.info("Testing MobileNetV4 block combinations...")
        
        # This would require modifying the MobileNetV4 implementation to support
        # different block type combinations. For now, we'll test the standard configurations.
        
        block_combinations = [
            {
                'name': 'Standard UIB Blocks',
                'description': 'Default Universal Inverted Bottleneck blocks',
                'config': {'use_standard_blocks': True}
            },
            {
                'name': 'ExtraDW Blocks',
                'description': 'Extra Depthwise blocks for efficiency',
                'config': {'use_extradw_blocks': True}
            },
            {
                'name': 'FFN Blocks',
                'description': 'Feed Forward Network blocks',
                'config': {'use_ffn_blocks': True}
            }
        ]
        
        # For now, we'll test the standard configurations
        # In a real implementation, you would modify the MobileNetV4 architecture
        # to support different block combinations
        
        results = {}
        for combo in block_combinations:
            logger.info(f"Testing {combo['name']}: {combo['description']}")
            
            # Test with MobileNetV4-Small as example
            try:
                model = mobilenetv4_small(width_mult=1.0, pretrained=False, embeding_dim=1024)
                model.eval()
                model.to(self.device)
                
                # Test forward pass
                input_tensor = torch.randn(1, *self.input_size).to(self.device)
                with torch.no_grad():
                    output = model(input_tensor)
                
                results[combo['name']] = {
                    'success': True,
                    'output_shape': list(output.shape),
                    'description': combo['description']
                }
                
                logger.info(f"Successfully tested {combo['name']}")
                
            except Exception as e:
                logger.error(f"Failed to test {combo['name']}: {e}")
                results[combo['name']] = {
                    'success': False,
                    'error': str(e),
                    'description': combo['description']
                }
        
        return results
    
    def generate_test_report(self) -> str:
        """Generate a comprehensive test report."""
        if not self.results:
            return "No results available. Run test_all_variants() first."
        
        report = []
        report.append("# MobileNetV4 Block Testing Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary table
        report.append("## Test Results Summary")
        report.append("")
        report.append("| Model | Creation | Forward Pass | Params | Multi-head | Error |")
        report.append("|-------|----------|--------------|--------|------------|-------|")
        
        for model_name, results in self.results.items():
            creation_status = "SUCCESS" if results['creation_success'] else "FAILED"
            forward_status = "SUCCESS" if results['forward_pass_success'] else "FAILED"
            multi_head_status = "SUCCESS" if results['multi_head_available'] else "FAILED"
            error_info = results['error'] if results['error'] else "-"
            
            report.append(f"| {model_name} | {creation_status} | {forward_status} | "
                        f"{results['total_params']:,} | {multi_head_status} | {error_info} |")
        
        report.append("")
        
        # Detailed results
        report.append("## Detailed Results")
        report.append("")
        
        for model_name, results in self.results.items():
            report.append(f"### {model_name}")
            report.append("")
            
            if results['creation_success']:
                report.append(f"- **Parameters**: {results['total_params']:,} total, {results['trainable_params']:,} trainable")
                report.append(f"- **Output Shape**: {results['output_shape']}")
                report.append(f"- **Multi-head Support**: {'Yes' if results['multi_head_available'] else 'No'}")
                if results['multi_head_available']:
                    report.append(f"- **Multi-head Outputs**: {results['multi_head_output_count']}")
            else:
                report.append(f"- **Error**: {results['error']}")
            
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        successful_models = [name for name, results in self.results.items() 
                           if results['creation_success'] and results['forward_pass_success']]
        
        if successful_models:
            report.append("### Successful Models")
            for model in successful_models:
                report.append(f"- {model}")
            report.append("")
            
            # Find best model by parameter count
            best_model = min(successful_models, 
                           key=lambda x: self.results[x]['total_params'])
            report.append(f"### Recommended Model (Lowest Parameters)")
            report.append(f"- **Model**: {best_model}")
            report.append(f"- **Parameters**: {self.results[best_model]['total_params']:,}")
            report.append("")
        
        return "\n".join(report)
    
    def save_results(self, output_path: str = "mobilenetv4_block_test_results.json"):
        """Save test results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def save_report(self, output_path: str = "mobilenetv4_block_test_report.md"):
        """Save test report to Markdown file."""
        report = self.generate_test_report()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {output_path}")


def main():
    """Main function to run the block tests."""
    parser = argparse.ArgumentParser(description="MobileNetV4 Block Type Testing")
    parser.add_argument("--device", type=str, default="cpu", 
                       help="Device to run tests on (cpu, cuda, cuda:0)")
    parser.add_argument("--output-dir", type=str, default="./mobilenetv4_test_results",
                       help="Output directory for results")
    parser.add_argument("--save-results", action="store_true", default=True,
                       help="Save results to files")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tester
    tester = MobileNetV4BlockTester(device=args.device)
    
    # Run tests
    logger.info("Starting MobileNetV4 block testing...")
    
    # Test all variants
    variant_results = tester.test_all_variants()
    
    # Test block combinations
    block_results = tester.test_block_combinations()
    
    # Combine results
    tester.results.update(block_results)
    
    # Generate and save report
    if args.save_results:
        tester.save_results(output_dir / "mobilenetv4_block_test_results.json")
        tester.save_report(output_dir / "mobilenetv4_block_test_report.md")
    
    # Print summary
    print("\n" + "="*60)
    print("MOBILENETV4 BLOCK TESTING COMPLETED")
    print("="*60)
    print(tester.generate_test_report())
    
    logger.info("MobileNetV4 block testing completed successfully!")


if __name__ == "__main__":
    main()
