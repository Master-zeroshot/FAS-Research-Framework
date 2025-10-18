#!/usr/bin/env python3
"""
Architecture Search Tools

This module provides tools for neural architecture search (NAS) and automated
architecture optimization for face anti-spoofing tasks.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import itertools
from dataclasses import dataclass
import time

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


@dataclass
class ArchitectureCandidate:
    """Represents a candidate architecture for search."""
    name: str
    model_type: str
    model_size: str
    embedding_dim: int
    parameters: Dict[str, Any]
    performance_score: float = 0.0
    efficiency_score: float = 0.0
    total_score: float = 0.0


class ArchitectureSearchEngine:
    """Neural Architecture Search engine for face anti-spoofing."""
    
    def __init__(self, device: str = "cpu", input_size: Tuple[int, int, int] = (3, 224, 224)):
        """
        Initialize the architecture search engine.
        
        Args:
            device: Device to run search on
            input_size: Input tensor size (C, H, W)
        """
        self.device = validate_device(device)
        self.input_size = input_size
        self.search_results = {}
        
        # Define search space
        self.search_space = {
            'MobileNetV2': {
                'width_mult': [0.75, 1.0, 1.2],
                'embedding_dim': [512, 1024, 1280]
            },
            'MobileNetV3': {
                'model_size': ['small', 'large'],
                'width_mult': [0.75, 1.0, 1.2],
                'embedding_dim': [1024, 1280]
            },
            'MobileNetV4': {
                'model_size': ['small', 'medium', 'large'],
                'width_mult': [0.75, 1.0, 1.2],
                'embedding_dim': [1024, 1152, 1280]
            },
            'EfficientNet': {
                'model_size': ['b0', 'b1', 'b2', 'b3'],
                'embedding_dim': [1280, 1536, 2048]
            },
            'ViT': {
                'model_size': ['tiny', 'small', 'base', 'large'],
                'embedding_dim': [768, 1024, 1280]
            },
            'ResNet': {
                'model_size': ['18', '34', '50', '101'],
                'embedding_dim': [512, 1024, 2048]
            }
        }
        
        # Model creation functions
        self.model_functions = {
            'MobileNetV2': mobilenetv2,
            'MobileNetV3': {
                'small': mobilenetv3_small,
                'large': mobilenetv3_large
            },
            'MobileNetV4': {
                'small': mobilenetv4_small,
                'medium': mobilenetv4_medium,
                'large': mobilenetv4_large
            },
            'EfficientNet': {
                'b0': efficientnet_b0,
                'b1': efficientnet_b1,
                'b2': efficientnet_b2,
                'b3': efficientnet_b3
            },
            'ViT': {
                'tiny': vit_tiny,
                'small': vit_small,
                'base': vit_base,
                'large': vit_large
            },
            'ResNet': {
                '18': resnet18,
                '34': resnet34,
                '50': resnet50,
                '101': resnet101
            }
        }
        
        logger.info(f"Initialized architecture search engine on device: {self.device}")
        logger.info(f"Search space: {len(self.search_space)} model types")
    
    def generate_random_candidate(self) -> ArchitectureCandidate:
        """Generate a random architecture candidate."""
        model_type = random.choice(list(self.search_space.keys()))
        model_config = self.search_space[model_type]
        
        # Generate random parameters
        params = {}
        for param_name, param_values in model_config.items():
            params[param_name] = random.choice(param_values)
        
        # Create candidate
        candidate = ArchitectureCandidate(
            name=f"{model_type}-{params.get('model_size', 'default')}-{random.randint(1000, 9999)}",
            model_type=model_type,
            model_size=params.get('model_size', 'default'),
            embedding_dim=params.get('embedding_dim', 1024),
            parameters=params
        )
        
        return candidate
    
    def evaluate_candidate(self, candidate: ArchitectureCandidate, 
                          batch_size: int = 8, num_iterations: int = 50) -> Dict[str, float]:
        """
        Evaluate a candidate architecture.
        
        Args:
            candidate: Architecture candidate to evaluate
            batch_size: Batch size for evaluation
            num_iterations: Number of iterations for timing
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Get model function
            if candidate.model_type == 'MobileNetV2':
                model_fn = self.model_functions['MobileNetV2']
                model_params = {
                    'width_mult': candidate.parameters.get('width_mult', 1.0),
                    'pretrained': False,
                    'embeding_dim': candidate.embedding_dim
                }
            else:
                model_fn = self.model_functions[candidate.model_type][candidate.model_size]
                model_params = {
                    'pretrained': False,
                    'embeding_dim': candidate.embedding_dim
                }
                if 'width_mult' in candidate.parameters:
                    model_params['width_mult'] = candidate.parameters['width_mult']
            
            # Create model
            model = model_fn(**model_params)
            model.eval()
            model.to(self.device)
            
            # Calculate model size
            total_params = sum(p.numel() for p in model.parameters())
            
            # Test forward pass
            input_tensor = torch.randn(batch_size, *self.input_size).to(self.device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
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
            
            # Calculate memory usage
            if self.device.startswith('cuda'):
                memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            else:
                memory_allocated = 0
            
            # Calculate scores
            performance_score = throughput  # Higher throughput = better performance
            efficiency_score = throughput / (total_params / 1000000)  # Throughput per million parameters
            
            # Clean up
            del model
            torch.cuda.empty_cache() if self.device.startswith('cuda') else None
            
            return {
                'total_params': total_params,
                'throughput': throughput,
                'memory_allocated_mb': memory_allocated,
                'avg_time_per_batch': avg_time_per_batch,
                'performance_score': performance_score,
                'efficiency_score': efficiency_score,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Failed to evaluate candidate {candidate.name}: {e}")
            return {
                'total_params': 0,
                'throughput': 0,
                'memory_allocated_mb': 0,
                'avg_time_per_batch': 0,
                'performance_score': 0,
                'efficiency_score': 0,
                'success': False,
                'error': str(e)
            }
    
    def random_search(self, num_candidates: int = 50) -> List[ArchitectureCandidate]:
        """
        Perform random search over the architecture space.
        
        Args:
            num_candidates: Number of candidates to evaluate
            
        Returns:
            List of evaluated candidates
        """
        logger.info(f"Starting random search with {num_candidates} candidates...")
        
        candidates = []
        
        for i in range(num_candidates):
            logger.info(f"Evaluating candidate {i+1}/{num_candidates}...")
            
            # Generate random candidate
            candidate = self.generate_random_candidate()
            
            # Evaluate candidate
            evaluation = self.evaluate_candidate(candidate)
            
            if evaluation['success']:
                candidate.performance_score = evaluation['performance_score']
                candidate.efficiency_score = evaluation['efficiency_score']
                candidate.total_score = (candidate.performance_score + candidate.efficiency_score) / 2
            
            candidates.append(candidate)
            
            # Log progress
            if evaluation['success']:
                logger.info(f"Candidate {candidate.name}: "
                           f"Performance={candidate.performance_score:.2f}, "
                           f"Efficiency={candidate.efficiency_score:.2f}")
            else:
                logger.warning(f"Candidate {candidate.name}: Failed evaluation")
        
        # Sort by total score
        candidates.sort(key=lambda x: x.total_score, reverse=True)
        
        self.search_results['random_search'] = candidates
        return candidates
    
    def grid_search(self, max_combinations: int = 100) -> List[ArchitectureCandidate]:
        """
        Perform grid search over the architecture space.
        
        Args:
            max_combinations: Maximum number of combinations to evaluate
            
        Returns:
            List of evaluated candidates
        """
        logger.info(f"Starting grid search with max {max_combinations} combinations...")
        
        candidates = []
        combination_count = 0
        
        for model_type, model_config in self.search_space.items():
            if combination_count >= max_combinations:
                break
            
            # Generate all combinations for this model type
            param_names = list(model_config.keys())
            param_values = list(model_config.values())
            
            for combination in itertools.product(*param_values):
                if combination_count >= max_combinations:
                    break
                
                # Create candidate
                params = dict(zip(param_names, combination))
                candidate = ArchitectureCandidate(
                    name=f"{model_type}-{params.get('model_size', 'default')}-{combination_count}",
                    model_type=model_type,
                    model_size=params.get('model_size', 'default'),
                    embedding_dim=params.get('embedding_dim', 1024),
                    parameters=params
                )
                
                # Evaluate candidate
                evaluation = self.evaluate_candidate(candidate)
                
                if evaluation['success']:
                    candidate.performance_score = evaluation['performance_score']
                    candidate.efficiency_score = evaluation['efficiency_score']
                    candidate.total_score = (candidate.performance_score + candidate.efficiency_score) / 2
                
                candidates.append(candidate)
                combination_count += 1
                
                # Log progress
                if evaluation['success']:
                    logger.info(f"Grid candidate {candidate.name}: "
                               f"Performance={candidate.performance_score:.2f}, "
                               f"Efficiency={candidate.efficiency_score:.2f}")
                else:
                    logger.warning(f"Grid candidate {candidate.name}: Failed evaluation")
        
        # Sort by total score
        candidates.sort(key=lambda x: x.total_score, reverse=True)
        
        self.search_results['grid_search'] = candidates
        return candidates
    
    def evolutionary_search(self, population_size: int = 20, generations: int = 10, 
                          mutation_rate: float = 0.1) -> List[ArchitectureCandidate]:
        """
        Perform evolutionary search over the architecture space.
        
        Args:
            population_size: Size of the population
            generations: Number of generations
            mutation_rate: Probability of mutation
            
        Returns:
            List of evaluated candidates
        """
        logger.info(f"Starting evolutionary search: {population_size} individuals, "
                   f"{generations} generations, mutation rate {mutation_rate}")
        
        # Initialize population
        population = []
        for _ in range(population_size):
            candidate = self.generate_random_candidate()
            evaluation = self.evaluate_candidate(candidate)
            
            if evaluation['success']:
                candidate.performance_score = evaluation['performance_score']
                candidate.efficiency_score = evaluation['efficiency_score']
                candidate.total_score = (candidate.performance_score + candidate.efficiency_score) / 2
            
            population.append(candidate)
        
        # Sort initial population
        population.sort(key=lambda x: x.total_score, reverse=True)
        
        for generation in range(generations):
            logger.info(f"Generation {generation + 1}/{generations}")
            
            # Select top performers for reproduction
            elite_size = population_size // 4
            elite = population[:elite_size]
            
            # Create new generation
            new_population = elite.copy()  # Keep elite
            
            while len(new_population) < population_size:
                # Select parents (tournament selection)
                parent1 = self._tournament_selection(population, tournament_size=3)
                parent2 = self._tournament_selection(population, tournament_size=3)
                
                # Create offspring
                offspring = self._crossover(parent1, parent2)
                
                # Mutate offspring
                if random.random() < mutation_rate:
                    offspring = self._mutate(offspring)
                
                # Evaluate offspring
                evaluation = self.evaluate_candidate(offspring)
                
                if evaluation['success']:
                    offspring.performance_score = evaluation['performance_score']
                    offspring.efficiency_score = evaluation['efficiency_score']
                    offspring.total_score = (offspring.performance_score + offspring.efficiency_score) / 2
                
                new_population.append(offspring)
            
            # Replace population
            population = new_population
            population.sort(key=lambda x: x.total_score, reverse=True)
            
            # Log best candidate
            best_candidate = population[0]
            logger.info(f"Best candidate: {best_candidate.name}, "
                       f"Score: {best_candidate.total_score:.2f}")
        
        self.search_results['evolutionary_search'] = population
        return population
    
    def _tournament_selection(self, population: List[ArchitectureCandidate], 
                            tournament_size: int = 3) -> ArchitectureCandidate:
        """Tournament selection for evolutionary search."""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.total_score)
    
    def _crossover(self, parent1: ArchitectureCandidate, 
                  parent2: ArchitectureCandidate) -> ArchitectureCandidate:
        """Crossover operation for evolutionary search."""
        # Create offspring by combining parameters
        offspring_params = {}
        
        for param_name in parent1.parameters:
            if random.random() < 0.5:
                offspring_params[param_name] = parent1.parameters[param_name]
            else:
                offspring_params[param_name] = parent2.parameters[param_name]
        
        # Create offspring
        offspring = ArchitectureCandidate(
            name=f"offspring-{random.randint(1000, 9999)}",
            model_type=random.choice([parent1.model_type, parent2.model_type]),
            model_size=offspring_params.get('model_size', 'default'),
            embedding_dim=offspring_params.get('embedding_dim', 1024),
            parameters=offspring_params
        )
        
        return offspring
    
    def _mutate(self, candidate: ArchitectureCandidate) -> ArchitectureCandidate:
        """Mutation operation for evolutionary search."""
        # Randomly change one parameter
        param_name = random.choice(list(candidate.parameters.keys()))
        
        if param_name in self.search_space[candidate.model_type]:
            new_value = random.choice(self.search_space[candidate.model_type][param_name])
            candidate.parameters[param_name] = new_value
        
        return candidate
    
    def generate_search_report(self, search_type: str = "random_search") -> str:
        """Generate a report for the search results."""
        if search_type not in self.search_results:
            return f"No results available for {search_type}."
        
        candidates = self.search_results[search_type]
        
        report = []
        report.append(f"# Architecture Search Report - {search_type.title()}")
        report.append("=" * 60)
        report.append("")
        
        # Summary statistics
        successful_candidates = [c for c in candidates if c.total_score > 0]
        report.append(f"## Summary Statistics")
        report.append(f"- **Total Candidates**: {len(candidates)}")
        report.append(f"- **Successful Candidates**: {len(successful_candidates)}")
        report.append(f"- **Success Rate**: {len(successful_candidates)/len(candidates)*100:.1f}%")
        report.append("")
        
        if successful_candidates:
            # Top candidates
            report.append("## Top 10 Candidates")
            report.append("")
            report.append("| Rank | Model | Performance | Efficiency | Total Score |")
            report.append("|------|-------|-------------|------------|-------------|")
            
            for i, candidate in enumerate(successful_candidates[:10]):
                report.append(f"| {i+1} | {candidate.name} | "
                            f"{candidate.performance_score:.2f} | "
                            f"{candidate.efficiency_score:.2f} | "
                            f"{candidate.total_score:.2f} |")
            
            report.append("")
            
            # Best candidate details
            best_candidate = successful_candidates[0]
            report.append("## Best Candidate")
            report.append("")
            report.append(f"- **Name**: {best_candidate.name}")
            report.append(f"- **Model Type**: {best_candidate.model_type}")
            report.append(f"- **Model Size**: {best_candidate.model_size}")
            report.append(f"- **Embedding Dimension**: {best_candidate.embedding_dim}")
            report.append(f"- **Parameters**: {best_candidate.parameters}")
            report.append(f"- **Performance Score**: {best_candidate.performance_score:.2f}")
            report.append(f"- **Efficiency Score**: {best_candidate.efficiency_score:.2f}")
            report.append(f"- **Total Score**: {best_candidate.total_score:.2f}")
            report.append("")
            
            # Model type distribution
            model_type_counts = {}
            for candidate in successful_candidates:
                model_type_counts[candidate.model_type] = model_type_counts.get(candidate.model_type, 0) + 1
            
            report.append("## Model Type Distribution")
            report.append("")
            for model_type, count in sorted(model_type_counts.items(), key=lambda x: x[1], reverse=True):
                report.append(f"- **{model_type}**: {count} candidates")
            report.append("")
        
        return "\n".join(report)
    
    def save_results(self, output_path: str = "architecture_search_results.json"):
        """Save search results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert candidates to serializable format
        serializable_results = {}
        for search_type, candidates in self.search_results.items():
            serializable_results[search_type] = []
            for candidate in candidates:
                serializable_results[search_type].append({
                    'name': candidate.name,
                    'model_type': candidate.model_type,
                    'model_size': candidate.model_size,
                    'embedding_dim': candidate.embedding_dim,
                    'parameters': candidate.parameters,
                    'performance_score': candidate.performance_score,
                    'efficiency_score': candidate.efficiency_score,
                    'total_score': candidate.total_score
                })
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def save_report(self, search_type: str = "random_search", 
                   output_path: str = "architecture_search_report.md"):
        """Save search report to Markdown file."""
        report = self.generate_search_report(search_type)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {output_path}")


def main():
    """Main function to run architecture search."""
    parser = argparse.ArgumentParser(description="Architecture Search Tools")
    parser.add_argument("--device", type=str, default="cpu", 
                       help="Device to run search on (cpu, cuda, cuda:0)")
    parser.add_argument("--search-type", type=str, default="random", 
                       choices=["random", "grid", "evolutionary"],
                       help="Type of search to perform")
    parser.add_argument("--num-candidates", type=int, default=50,
                       help="Number of candidates to evaluate")
    parser.add_argument("--output-dir", type=str, default="./architecture_search_results",
                       help="Output directory for results")
    parser.add_argument("--save-results", action="store_true", default=True,
                       help="Save results to files")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize search engine
    search_engine = ArchitectureSearchEngine(device=args.device)
    
    # Run search
    logger.info(f"Starting {args.search_type} search...")
    
    if args.search_type == "random":
        candidates = search_engine.random_search(args.num_candidates)
    elif args.search_type == "grid":
        candidates = search_engine.grid_search(args.num_candidates)
    elif args.search_type == "evolutionary":
        candidates = search_engine.evolutionary_search(
            population_size=args.num_candidates // 2,
            generations=10
        )
    
    # Generate and save results
    if args.save_results:
        search_engine.save_results(output_dir / "architecture_search_results.json")
        search_engine.save_report(args.search_type, output_dir / f"{args.search_type}_search_report.md")
    
    # Print summary
    print("\n" + "="*60)
    print("ARCHITECTURE SEARCH COMPLETED")
    print("="*60)
    print(search_engine.generate_search_report(args.search_type + "_search"))
    
    logger.info("Architecture search completed successfully!")


if __name__ == "__main__":
    main()
