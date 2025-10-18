#!/usr/bin/env python3
"""
FAS-Research-Framework MobileNetV4 Complete Optimization Runner

This script runs all MobileNetV4 optimization tasks including:
- Performance benchmarking against MobileNetV3
- Block type testing
- Dataset validation
- Hyperparameter optimization
- Configuration generation

Part of the FAS-Research-Framework for advanced face anti-spoofing research.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import logging
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MobileNetV4OptimizationRunner:
    """Run all MobileNetV4 optimization tasks."""
    
    def __init__(self, device: str = "cpu", output_dir: str = "./mobilenetv4_optimization_results"):
        """
        Initialize the optimization runner.
        
        Args:
            device: Device to run optimization on
            output_dir: Output directory for all results
        """
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Scripts to run
        self.scripts = [
            {
                'name': 'Performance Benchmarking',
                'script': 'benchmark_mobilenetv4.py',
                'args': ['--device', device, '--output-dir', str(self.output_dir / 'benchmark')]
            },
            {
                'name': 'Block Type Testing',
                'script': 'test_mobilenetv4_blocks.py',
                'args': ['--device', device, '--output-dir', str(self.output_dir / 'block_test')]
            },
            {
                'name': 'Dataset Validation',
                'script': 'validate_mobilenetv4_datasets.py',
                'args': ['--device', device, '--output-dir', str(self.output_dir / 'validation')]
            },
            {
                'name': 'Hyperparameter Optimization',
                'script': 'optimize_mobilenetv4.py',
                'args': ['--device', device, '--output-dir', str(self.output_dir / 'optimization')]
            }
        ]
        
        logger.info(f"Initialized MobileNetV4 optimization runner on device: {device}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def run_script(self, script_info: Dict[str, Any]) -> bool:
        """
        Run a single optimization script.
        
        Args:
            script_info: Dictionary containing script information
            
        Returns:
            True if script ran successfully, False otherwise
        """
        script_name = script_info['name']
        script_path = script_info['script']
        script_args = script_info['args']
        
        logger.info(f"Running {script_name}...")
        
        try:
            # Run the script
            cmd = [sys.executable, script_path] + script_args
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                logger.info(f"SUCCESS: {script_name} completed successfully")
                if result.stdout:
                    logger.info(f"Output: {result.stdout}")
                return True
            else:
                logger.error(f"FAILED: {script_name} failed with return code {result.returncode}")
                if result.stderr:
                    logger.error(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"TIMEOUT: {script_name} timed out after 1 hour")
            return False
        except Exception as e:
            logger.error(f"ERROR: {script_name} failed with exception: {e}")
            return False
    
    def run_all_optimizations(self) -> Dict[str, bool]:
        """
        Run all optimization scripts.
        
        Returns:
            Dictionary with script names and their success status
        """
        logger.info("Starting MobileNetV4 complete optimization...")
        
        results = {}
        
        for script_info in self.scripts:
            script_name = script_info['name']
            success = self.run_script(script_info)
            results[script_name] = success
        
        return results
    
    def generate_summary_report(self, results: Dict[str, bool]) -> str:
        """
        Generate a summary report of all optimization results.
        
        Args:
            results: Dictionary with script names and success status
            
        Returns:
            Summary report string
        """
        report = []
        report.append("# MobileNetV4 Complete Optimization Summary")
        report.append("=" * 60)
        report.append("")
        
        # Summary table
        report.append("## Optimization Results")
        report.append("")
        report.append("| Task | Status | Description |")
        report.append("|------|--------|-------------|")
        
        for script_name, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            description = self._get_task_description(script_name)
            report.append(f"| {script_name} | {status} | {description} |")
        
        report.append("")
        
        # Success rate
        successful_tasks = sum(results.values())
        total_tasks = len(results)
        success_rate = (successful_tasks / total_tasks) * 100
        
        report.append(f"## Summary Statistics")
        report.append("")
        report.append(f"- **Total Tasks**: {total_tasks}")
        report.append(f"- **Successful Tasks**: {successful_tasks}")
        report.append(f"- **Failed Tasks**: {total_tasks - successful_tasks}")
        report.append(f"- **Success Rate**: {success_rate:.1f}%")
        report.append("")
        
        # Output directory information
        report.append("## Output Files")
        report.append("")
        report.append("All optimization results are saved in the following directory structure:")
        report.append("")
        report.append("```")
        report.append(f"{self.output_dir}/")
        report.append("├── benchmark/")
        report.append("│   ├── benchmark_results.json")
        report.append("│   └── benchmark_report.md")
        report.append("├── block_test/")
        report.append("│   ├── mobilenetv4_block_test_results.json")
        report.append("│   └── mobilenetv4_block_test_report.md")
        report.append("├── validation/")
        report.append("│   ├── mobilenetv4_validation_results.json")
        report.append("│   └── mobilenetv4_validation_report.md")
        report.append("└── optimization/")
        report.append("    ├── mobilenetv4_optimization_results.json")
        report.append("    └── mobilenetv4_optimization_report.md")
        report.append("```")
        report.append("")
        
        # Next steps
        report.append("## Next Steps")
        report.append("")
        
        if success_rate == 100:
            report.append("**All optimization tasks completed successfully!**")
            report.append("")
            report.append("You can now:")
            report.append("1. Review the optimization reports in each subdirectory")
            report.append("2. Use the optimized configurations for training")
            report.append("3. Compare MobileNetV4 performance with MobileNetV3")
            report.append("4. Deploy the optimized models for production")
        else:
            report.append("**WARNING: Some optimization tasks failed.**")
            report.append("")
            report.append("Please check the logs above for error details and:")
            report.append("1. Fix any issues with the failed tasks")
            report.append("2. Re-run the optimization runner")
            report.append("3. Contact support if issues persist")
        
        report.append("")
        
        return "\n".join(report)
    
    def _get_task_description(self, task_name: str) -> str:
        """Get description for a task."""
        descriptions = {
            'Performance Benchmarking': 'Compare MobileNetV4 vs MobileNetV3 performance',
            'Block Type Testing': 'Test MobileNetV4 with different block combinations',
            'Dataset Validation': 'Validate MobileNetV4 on CelebA-Spoof and LCC-FASD',
            'Hyperparameter Optimization': 'Find optimal hyperparameters for MobileNetV4'
        }
        return descriptions.get(task_name, 'Unknown task')
    
    def save_summary_report(self, results: Dict[str, bool], output_path: str = None):
        """Save the summary report to a file."""
        if output_path is None:
            output_path = self.output_dir / "mobilenetv4_optimization_summary.md"
        
        report = self.generate_summary_report(results)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Summary report saved to {output_path}")


def main():
    """Main function to run all MobileNetV4 optimizations."""
    parser = argparse.ArgumentParser(description="MobileNetV4 Complete Optimization Runner")
    parser.add_argument("--device", type=str, default="cpu", 
                       help="Device to run optimization on (cpu, cuda, cuda:0)")
    parser.add_argument("--output-dir", type=str, default="./mobilenetv4_optimization_results",
                       help="Output directory for all results")
    parser.add_argument("--skip-failed", action="store_true", default=False,
                       help="Continue running even if some tasks fail")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = MobileNetV4OptimizationRunner(device=args.device, output_dir=args.output_dir)
    
    # Run all optimizations
    logger.info("Starting MobileNetV4 complete optimization...")
    results = runner.run_all_optimizations()
    
    # Generate and save summary report
    runner.save_summary_report(results)
    
    # Print summary
    print("\n" + "="*60)
    print("MOBILENETV4 COMPLETE OPTIMIZATION SUMMARY")
    print("="*60)
    print(runner.generate_summary_report(results))
    
    # Check if all tasks succeeded
    failed_tasks = [name for name, success in results.items() if not success]
    
    if failed_tasks:
        logger.warning(f"Some tasks failed: {failed_tasks}")
        if not args.skip_failed:
            sys.exit(1)
    else:
        logger.info("All MobileNetV4 optimization tasks completed successfully!")
    
    logger.info("MobileNetV4 complete optimization finished!")


if __name__ == "__main__":
    main()
