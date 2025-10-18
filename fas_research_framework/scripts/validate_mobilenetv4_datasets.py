#!/usr/bin/env python3
"""
FAS-Research-Framework MobileNetV4 Dataset Validation Script

This script validates MobileNetV4 models on both CelebA-Spoof and LCC FASD datasets
to ensure proper integration and performance.
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
import time

# Import models and datasets
from models import mobilenetv4_small, mobilenetv4_medium, mobilenetv4_large
from datasets import get_datasets
from utils import Transform, make_dataset, make_loader, build_model, read_py_config

# Import utilities
from utils.error_handling import validate_device, safe_model_forward
from utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(log_level="INFO", log_to_file=True, log_to_console=True, structured=True)
logger = get_logger(__name__)


class MobileNetV4DatasetValidator:
    """
    Validate MobileNetV4 models on face anti-spoofing datasets.
    
    This class provides comprehensive validation of MobileNetV4 models on both
    CelebA-Spoof and LCC FASD datasets, measuring accuracy, throughput, and
    memory usage across different model variants.
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize the dataset validator.
        
        Args:
            device: Device to run validation on
        """
        self.device = validate_device(device)
        self.results = {}
        
        logger.info(f"Initialized MobileNetV4 dataset validator on device: {self.device}")
    
    def validate_model_on_dataset(self, model_name: str, model: nn.Module, 
                                 dataset_name: str, dataset, batch_size: int = 32) -> Dict[str, Any]:
        """
        Validate a model on a specific dataset.
        
        Args:
            model_name: Name of the model
            model: PyTorch model
            dataset_name: Name of the dataset
            dataset: Dataset object
            batch_size: Batch size for validation
            
        Returns:
            Dictionary with validation results
        """
        try:
            model.eval()
            model.to(self.device)
            
            # Create data loader
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=2
            )
            
            # Validation metrics
            total_samples = 0
            correct_predictions = 0
            total_loss = 0.0
            inference_times = []
            
            # Loss function
            criterion = nn.CrossEntropyLoss()
            
            logger.info(f"Validating {model_name} on {dataset_name}...")
            
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(dataloader):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Measure inference time
                    start_time = time.time()
                    outputs = model(inputs)
                    end_time = time.time()
                    
                    inference_times.append(end_time - start_time)
                    
                    # Calculate loss based on task type
                    if len(targets.shape) > 1 and targets.shape[1] > 1:
                        # Multi-task learning case: use spoof detection label (first column)
                        loss = criterion(outputs, targets[:, 0])  # Use spoof label
                        predictions = outputs.argmax(dim=1)
                        correct = (predictions == targets[:, 0]).sum().item()
                    else:
                        # Single task case: standard binary classification
                        loss = criterion(outputs, targets)
                        predictions = outputs.argmax(dim=1)
                        correct = (predictions == targets).sum().item()
                    
                    total_loss += loss.item()
                    correct_predictions += correct
                    total_samples += inputs.size(0)
                    
                    # Log progress
                    if batch_idx % 10 == 0:
                        logger.info(f"Batch {batch_idx}/{len(dataloader)}, "
                                  f"Accuracy: {correct_predictions/total_samples:.4f}")
            
            # Calculate final metrics
            accuracy = correct_predictions / total_samples
            avg_loss = total_loss / len(dataloader)
            avg_inference_time = np.mean(inference_times)
            throughput = batch_size / avg_inference_time
            
            results = {
                'model_name': model_name,
                'dataset_name': dataset_name,
                'total_samples': total_samples,
                'correct_predictions': correct_predictions,
                'accuracy': accuracy,
                'avg_loss': avg_loss,
                'avg_inference_time': avg_inference_time,
                'throughput': throughput,
                'batch_size': batch_size,
                'success': True,
                'error': None
            }
            
            logger.info(f"Validation completed for {model_name} on {dataset_name}: "
                       f"Accuracy: {accuracy:.4f}, Throughput: {throughput:.2f} samples/s")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to validate {model_name} on {dataset_name}: {e}")
            return {
                'model_name': model_name,
                'dataset_name': dataset_name,
                'success': False,
                'error': str(e)
            }
    
    def validate_on_celeba_spoof(self) -> Dict[str, Any]:
        """Validate MobileNetV4 models on CelebA-Spoof dataset."""
        logger.info("Validating MobileNetV4 models on CelebA-Spoof dataset...")
        
        # Create a simple config for CelebA-Spoof
        config = {
            'dataset': 'celeba_spoof',
            'multi_task_learning': True,
            'datasets': {
                'Celeba_root': '../CelebA-Spoof-zips/CelebA_Spoof',
                'Casia_root': './CASIA',
                'LCCFASD_root': '../LCC_FASD'
            },
            'img_norm_cfg': {
                'mean': [0.5931, 0.4690, 0.4229],
                'std': [0.2471, 0.2214, 0.2157]
            },
            'resize': {'height': 224, 'width': 224}
        }
        
        # Create transform
        from albumentations import A
        normalize = A.Normalize(**config['img_norm_cfg'])
        val_transform = A.Compose([
            A.Resize(**config['resize'], interpolation=1),  # INTER_CUBIC
            normalize
        ])
        transform = Transform(val=val_transform)
        
        try:
            # Create dataset
            dataset = make_dataset(config, val_transform=transform, mode='eval')
            
            # Test all MobileNetV4 variants
            models_to_test = [
                ('MobileNetV4-Small', mobilenetv4_small),
                ('MobileNetV4-Medium', mobilenetv4_medium),
                ('MobileNetV4-Large', mobilenetv4_large)
            ]
            
            results = {}
            for model_name, model_fn in models_to_test:
                try:
                    # Create model
                    if 'Small' in model_name:
                        model = model_fn(width_mult=1.0, pretrained=False, embeding_dim=1024)
                    elif 'Medium' in model_name:
                        model = model_fn(width_mult=1.0, pretrained=False, embeding_dim=1152)
                    else:  # Large
                        model = model_fn(width_mult=1.0, pretrained=False, embeding_dim=1280)
                    
                    # Validate model
                    validation_results = self.validate_model_on_dataset(
                        model_name, model, 'CelebA-Spoof', dataset
                    )
                    results[model_name] = validation_results
                    
                except Exception as e:
                    logger.error(f"Failed to test {model_name} on CelebA-Spoof: {e}")
                    results[model_name] = {
                        'model_name': model_name,
                        'dataset_name': 'CelebA-Spoof',
                        'success': False,
                        'error': str(e)
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to create CelebA-Spoof dataset: {e}")
            return {'error': str(e)}
    
    def validate_on_lcc_fasd(self) -> Dict[str, Any]:
        """Validate MobileNetV4 models on LCC FASD dataset."""
        logger.info("Validating MobileNetV4 models on LCC FASD dataset...")
        
        # Create a simple config for LCC FASD
        config = {
            'dataset': 'lcc_fasd',
            'multi_task_learning': False,
            'datasets': {
                'LCCFASD_root': '../LCC_FASD'
            },
            'img_norm_cfg': {
                'mean': [0.5931, 0.4690, 0.4229],
                'std': [0.2471, 0.2214, 0.2157]
            },
            'resize': {'height': 224, 'width': 224}
        }
        
        # Create transform
        from albumentations import A
        normalize = A.Normalize(**config['img_norm_cfg'])
        val_transform = A.Compose([
            A.Resize(**config['resize'], interpolation=1),  # INTER_CUBIC
            normalize
        ])
        transform = Transform(val=val_transform)
        
        try:
            # Create dataset
            dataset = make_dataset(config, val_transform=transform, mode='eval')
            
            # Test all MobileNetV4 variants
            models_to_test = [
                ('MobileNetV4-Small', mobilenetv4_small),
                ('MobileNetV4-Medium', mobilenetv4_medium),
                ('MobileNetV4-Large', mobilenetv4_large)
            ]
            
            results = {}
            for model_name, model_fn in models_to_test:
                try:
                    # Create model
                    if 'Small' in model_name:
                        model = model_fn(width_mult=1.0, pretrained=False, embeding_dim=1024)
                    elif 'Medium' in model_name:
                        model = model_fn(width_mult=1.0, pretrained=False, embeding_dim=1152)
                    else:  # Large
                        model = model_fn(width_mult=1.0, pretrained=False, embeding_dim=1280)
                    
                    # Validate model
                    validation_results = self.validate_model_on_dataset(
                        model_name, model, 'LCC-FASD', dataset
                    )
                    results[model_name] = validation_results
                    
                except Exception as e:
                    logger.error(f"Failed to test {model_name} on LCC-FASD: {e}")
                    results[model_name] = {
                        'model_name': model_name,
                        'dataset_name': 'LCC-FASD',
                        'success': False,
                        'error': str(e)
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to create LCC-FASD dataset: {e}")
            return {'error': str(e)}
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run full validation on both datasets."""
        logger.info("Starting full MobileNetV4 dataset validation...")
        
        all_results = {}
        
        # Validate on CelebA-Spoof
        celeba_results = self.validate_on_celeba_spoof()
        all_results['CelebA-Spoof'] = celeba_results
        
        # Validate on LCC-FASD
        lcc_results = self.validate_on_lcc_fasd()
        all_results['LCC-FASD'] = lcc_results
        
        self.results = all_results
        return all_results
    
    def generate_validation_report(self) -> str:
        """Generate a comprehensive validation report."""
        if not self.results:
            return "No results available. Run run_full_validation() first."
        
        report = []
        report.append("# MobileNetV4 Dataset Validation Report")
        report.append("=" * 60)
        report.append("")
        
        # Summary table
        report.append("## Validation Results Summary")
        report.append("")
        report.append("| Model | Dataset | Success | Accuracy | Throughput | Error |")
        report.append("|-------|---------|---------|----------|------------|-------|")
        
        for dataset_name, dataset_results in self.results.items():
            if 'error' in dataset_results:
                report.append(f"| - | {dataset_name} | FAILED | - | - | {dataset_results['error']} |")
                continue
                
            for model_name, results in dataset_results.items():
                if 'error' in results:
                    report.append(f"| {model_name} | {dataset_name} | FAILED | - | - | {results['error']} |")
                    continue
                
                success_status = "SUCCESS" if results['success'] else "FAILED"
                accuracy = f"{results['accuracy']:.4f}" if results['success'] else "-"
                throughput = f"{results['throughput']:.2f}" if results['success'] else "-"
                error_info = results['error'] if results['error'] else "-"
                
                report.append(f"| {model_name} | {dataset_name} | {success_status} | "
                            f"{accuracy} | {throughput} | {error_info} |")
        
        report.append("")
        
        # Detailed results
        report.append("## Detailed Results")
        report.append("")
        
        for dataset_name, dataset_results in self.results.items():
            if 'error' in dataset_results:
                report.append(f"### {dataset_name} - Dataset Error")
                report.append(f"Error: {dataset_results['error']}")
                report.append("")
                continue
            
            report.append(f"### {dataset_name}")
            report.append("")
            
            for model_name, results in dataset_results.items():
                report.append(f"#### {model_name}")
                report.append("")
                
                if results['success']:
                    report.append(f"- **Total Samples**: {results['total_samples']:,}")
                    report.append(f"- **Correct Predictions**: {results['correct_predictions']:,}")
                    report.append(f"- **Accuracy**: {results['accuracy']:.4f}")
                    report.append(f"- **Average Loss**: {results['avg_loss']:.4f}")
                    report.append(f"- **Average Inference Time**: {results['avg_inference_time']:.4f}s")
                    report.append(f"- **Throughput**: {results['throughput']:.2f} samples/s")
                    report.append(f"- **Batch Size**: {results['batch_size']}")
                else:
                    report.append(f"- **Error**: {results['error']}")
                
                report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        successful_validations = []
        for dataset_name, dataset_results in self.results.items():
            if 'error' in dataset_results:
                continue
                
            for model_name, results in dataset_results.items():
                if results['success']:
                    successful_validations.append((model_name, dataset_name, results))
        
        if successful_validations:
            report.append("### Successful Validations")
            for model_name, dataset_name, results in successful_validations:
                report.append(f"- **{model_name}** on **{dataset_name}**: "
                            f"Accuracy: {results['accuracy']:.4f}, "
                            f"Throughput: {results['throughput']:.2f} samples/s")
            report.append("")
            
            # Find best model by accuracy
            best_model = max(successful_validations, key=lambda x: x[2]['accuracy'])
            report.append(f"### Best Performing Model")
            report.append(f"- **Model**: {best_model[0]}")
            report.append(f"- **Dataset**: {best_model[1]}")
            report.append(f"- **Accuracy**: {best_model[2]['accuracy']:.4f}")
            report.append(f"- **Throughput**: {best_model[2]['throughput']:.2f} samples/s")
            report.append("")
        
        return "\n".join(report)
    
    def save_results(self, output_path: str = "mobilenetv4_validation_results.json"):
        """Save validation results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def save_report(self, output_path: str = "mobilenetv4_validation_report.md"):
        """Save validation report to Markdown file."""
        report = self.generate_validation_report()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {output_path}")


def main():
    """Main function to run the dataset validation."""
    parser = argparse.ArgumentParser(description="MobileNetV4 Dataset Validation")
    parser.add_argument("--device", type=str, default="cpu", 
                       help="Device to run validation on (cpu, cuda, cuda:0)")
    parser.add_argument("--output-dir", type=str, default="./mobilenetv4_validation_results",
                       help="Output directory for results")
    parser.add_argument("--save-results", action="store_true", default=True,
                       help="Save results to files")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize validator
    validator = MobileNetV4DatasetValidator(device=args.device)
    
    # Run validation
    logger.info("Starting MobileNetV4 dataset validation...")
    results = validator.run_full_validation()
    
    # Generate and save report
    if args.save_results:
        validator.save_results(output_dir / "mobilenetv4_validation_results.json")
        validator.save_report(output_dir / "mobilenetv4_validation_report.md")
    
    # Print summary
    print("\n" + "="*60)
    print("MOBILENETV4 DATASET VALIDATION COMPLETED")
    print("="*60)
    print(validator.generate_validation_report())
    
    logger.info("MobileNetV4 dataset validation completed successfully!")


if __name__ == "__main__":
    main()
