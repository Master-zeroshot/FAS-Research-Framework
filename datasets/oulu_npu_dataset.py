"""
OULU-NPU Dataset Implementation

This module implements the OULU-NPU dataset for face anti-spoofing tasks.
OULU-NPU is a high-quality dataset with controlled lighting conditions.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

from utils.logging_config import get_logger

logger = get_logger(__name__)


class OULUNPUDataset(Dataset):
    """OULU-NPU dataset for face anti-spoofing."""
    
    def __init__(self, root_dir: str, split: str = "train", transform=None, 
                 target_transform=None, load_metadata: bool = True):
        """
        Initialize OULU-NPU dataset.
        
        Args:
            root_dir: Root directory of OULU-NPU dataset
            split: Dataset split ('train', 'test', 'val')
            transform: Image transformations
            target_transform: Target transformations
            load_metadata: Whether to load metadata
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.load_metadata = load_metadata
        
        # Validate dataset structure
        if not self.root_dir.exists():
            raise FileNotFoundError(f"OULU-NPU dataset not found at {root_dir}")
        
        # Load dataset metadata
        self.metadata = self._load_metadata()
        self.samples = self._load_samples()
        
        logger.info(f"Loaded OULU-NPU {split} dataset with {len(self.samples)} samples")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata."""
        metadata_file = self.root_dir / f"{self.split}_metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            # Generate metadata if not exists
            metadata = self._generate_metadata()
            self._save_metadata(metadata)
        
        return metadata
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate dataset metadata."""
        metadata = {
            'dataset_name': 'OULU-NPU',
            'split': self.split,
            'total_samples': 0,
            'real_samples': 0,
            'spoof_samples': 0,
            'subjects': [],
            'sessions': [],
            'lighting_conditions': [],
            'devices': []
        }
        
        # Scan dataset directory
        for subject_dir in self.root_dir.iterdir():
            if not subject_dir.is_dir():
                continue
            
            subject_id = subject_dir.name
            metadata['subjects'].append(subject_id)
            
            for session_dir in subject_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                
                session_id = session_dir.name
                metadata['sessions'].append(f"{subject_id}_{session_id}")
                
                # Count samples in session
                for file_path in session_dir.rglob("*.jpg"):
                    if file_path.is_file():
                        metadata['total_samples'] += 1
                        
                        # Determine if real or spoof
                        if "real" in file_path.name.lower():
                            metadata['real_samples'] += 1
                        else:
                            metadata['spoof_samples'] += 1
                            
                            # Extract lighting condition and device
                            lighting = self._extract_lighting_condition(file_path)
                            device = self._extract_device(file_path)
                            
                            if lighting not in metadata['lighting_conditions']:
                                metadata['lighting_conditions'].append(lighting)
                            if device not in metadata['devices']:
                                metadata['devices'].append(device)
        
        return metadata
    
    def _extract_lighting_condition(self, file_path: Path) -> str:
        """Extract lighting condition from file path."""
        filename = file_path.name.lower()
        
        if "normal" in filename:
            return "normal_lighting"
        elif "strong" in filename:
            return "strong_lighting"
        elif "weak" in filename:
            return "weak_lighting"
        else:
            return "unknown_lighting"
    
    def _extract_device(self, file_path: Path) -> str:
        """Extract device information from file path."""
        filename = file_path.name.lower()
        
        if "phone" in filename:
            return "phone"
        elif "tablet" in filename:
            return "tablet"
        elif "laptop" in filename:
            return "laptop"
        else:
            return "unknown_device"
    
    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save metadata to file."""
        metadata_file = self.root_dir / f"{self.split}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load dataset samples."""
        samples = []
        
        for subject_dir in self.root_dir.iterdir():
            if not subject_dir.is_dir():
                continue
            
            subject_id = subject_dir.name
            
            for session_dir in subject_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                
                session_id = session_dir.name
                
                # Load real samples
                real_dir = session_dir / "real"
                if real_dir.exists():
                    for file_path in real_dir.glob("*.jpg"):
                        samples.append({
                            'image_path': str(file_path),
                            'label': 0,  # Real
                            'subject_id': subject_id,
                            'session_id': session_id,
                            'lighting_condition': 'normal_lighting',
                            'device': 'unknown_device',
                            'metadata': self._extract_image_metadata(file_path)
                        })
                
                # Load spoof samples
                spoof_dir = session_dir / "spoof"
                if spoof_dir.exists():
                    for file_path in spoof_dir.glob("*.jpg"):
                        lighting_condition = self._extract_lighting_condition(file_path)
                        device = self._extract_device(file_path)
                        samples.append({
                            'image_path': str(file_path),
                            'label': 1,  # Spoof
                            'subject_id': subject_id,
                            'session_id': session_id,
                            'lighting_condition': lighting_condition,
                            'device': device,
                            'metadata': self._extract_image_metadata(file_path)
                        })
        
        return samples
    
    def _extract_image_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from image file."""
        try:
            # Load image to get dimensions
            image = Image.open(file_path)
            width, height = image.size
            
            metadata = {
                'width': width,
                'height': height,
                'channels': 3,
                'file_size': file_path.stat().st_size,
                'filename': file_path.name
            }
            
            return metadata
        except Exception as e:
            logger.warning(f"Failed to extract metadata from {file_path}: {e}")
            return {}
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Get dataset item.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image, label, metadata)
        """
        sample = self.samples[idx]
        
        # Load image
        image_path = sample['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            # Return dummy image if loading fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Get label
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        # Prepare metadata
        metadata = {
            'subject_id': sample['subject_id'],
            'session_id': sample['session_id'],
            'lighting_condition': sample['lighting_condition'],
            'device': sample['device'],
            'image_path': image_path
        }
        
        if self.load_metadata:
            metadata.update(sample['metadata'])
        
        return image, label, metadata
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution."""
        distribution = {'real': 0, 'spoof': 0}
        
        for sample in self.samples:
            if sample['label'] == 0:
                distribution['real'] += 1
            else:
                distribution['spoof'] += 1
        
        return distribution
    
    def get_lighting_distribution(self) -> Dict[str, int]:
        """Get lighting condition distribution."""
        distribution = {}
        
        for sample in self.samples:
            lighting = sample['lighting_condition']
            distribution[lighting] = distribution.get(lighting, 0) + 1
        
        return distribution
    
    def get_device_distribution(self) -> Dict[str, int]:
        """Get device distribution."""
        distribution = {}
        
        for sample in self.samples:
            device = sample['device']
            distribution[device] = distribution.get(device, 0) + 1
        
        return distribution
    
    def get_subject_distribution(self) -> Dict[str, int]:
        """Get subject distribution."""
        distribution = {}
        
        for sample in self.samples:
            subject_id = sample['subject_id']
            distribution[subject_id] = distribution.get(subject_id, 0) + 1
        
        return distribution


class OULUNPUDataLoader:
    """Data loader for OULU-NPU dataset."""
    
    def __init__(self, root_dir: str, batch_size: int = 32, num_workers: int = 4,
                 transform=None, target_transform=None):
        """
        Initialize OULU-NPU data loader.
        
        Args:
            root_dir: Root directory of OULU-NPU dataset
            batch_size: Batch size
            num_workers: Number of worker processes
            transform: Image transformations
            target_transform: Target transformations
        """
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.target_transform = target_transform
        
        logger.info(f"Initialized OULU-NPU data loader with batch_size={batch_size}")
    
    def get_dataloader(self, split: str = "train", shuffle: bool = True) -> DataLoader:
        """
        Get data loader for specified split.
        
        Args:
            split: Dataset split
            shuffle: Whether to shuffle data
            
        Returns:
            Data loader
        """
        dataset = OULUNPUDataset(
            root_dir=self.root_dir,
            split=split,
            transform=self.transform,
            target_transform=self.target_transform
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        return dataloader
    
    def get_train_dataloader(self) -> DataLoader:
        """Get training data loader."""
        return self.get_dataloader("train", shuffle=True)
    
    def get_val_dataloader(self) -> DataLoader:
        """Get validation data loader."""
        return self.get_dataloader("val", shuffle=False)
    
    def get_test_dataloader(self) -> DataLoader:
        """Get test data loader."""
        return self.get_dataloader("test", shuffle=False)


class OULUNPUEvaluator:
    """Evaluator for OULU-NPU dataset."""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Initialize OULU-NPU evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        logger.info("Initialized OULU-NPU evaluator")
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on OULU-NPU dataset.
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        total_samples = 0
        correct_predictions = 0
        real_correct = 0
        spoof_correct = 0
        real_total = 0
        spoof_total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images, labels, metadata = batch
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
    
    def evaluate_by_lighting(self, dataloader: DataLoader) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model by lighting condition.
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            Dictionary with lighting condition specific metrics
        """
        lighting_metrics = {}
        
        with torch.no_grad():
            for batch in dataloader:
                images, labels, metadata = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                predictions = outputs.argmax(dim=1)
                
                # Group by lighting condition
                for i, lighting in enumerate(metadata['lighting_condition']):
                    if lighting not in lighting_metrics:
                        lighting_metrics[lighting] = {
                            'total': 0,
                            'correct': 0,
                            'real_total': 0,
                            'real_correct': 0,
                            'spoof_total': 0,
                            'spoof_correct': 0
                        }
                    
                    lighting_metrics[lighting]['total'] += 1
                    if predictions[i] == labels[i]:
                        lighting_metrics[lighting]['correct'] += 1
                    
                    if labels[i] == 0:  # Real
                        lighting_metrics[lighting]['real_total'] += 1
                        if predictions[i] == labels[i]:
                            lighting_metrics[lighting]['real_correct'] += 1
                    else:  # Spoof
                        lighting_metrics[lighting]['spoof_total'] += 1
                        if predictions[i] == labels[i]:
                            lighting_metrics[lighting]['spoof_correct'] += 1
        
        # Compute final metrics
        for lighting in lighting_metrics:
            metrics = lighting_metrics[lighting]
            metrics['accuracy'] = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0
            metrics['real_accuracy'] = metrics['real_correct'] / metrics['real_total'] if metrics['real_total'] > 0 else 0
            metrics['spoof_accuracy'] = metrics['spoof_correct'] / metrics['spoof_total'] if metrics['spoof_total'] > 0 else 0
        
        return lighting_metrics
    
    def evaluate_by_device(self, dataloader: DataLoader) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model by device.
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            Dictionary with device specific metrics
        """
        device_metrics = {}
        
        with torch.no_grad():
            for batch in dataloader:
                images, labels, metadata = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                predictions = outputs.argmax(dim=1)
                
                # Group by device
                for i, device in enumerate(metadata['device']):
                    if device not in device_metrics:
                        device_metrics[device] = {
                            'total': 0,
                            'correct': 0,
                            'real_total': 0,
                            'real_correct': 0,
                            'spoof_total': 0,
                            'spoof_correct': 0
                        }
                    
                    device_metrics[device]['total'] += 1
                    if predictions[i] == labels[i]:
                        device_metrics[device]['correct'] += 1
                    
                    if labels[i] == 0:  # Real
                        device_metrics[device]['real_total'] += 1
                        if predictions[i] == labels[i]:
                            device_metrics[device]['real_correct'] += 1
                    else:  # Spoof
                        device_metrics[device]['spoof_total'] += 1
                        if predictions[i] == labels[i]:
                            device_metrics[device]['spoof_correct'] += 1
        
        # Compute final metrics
        for device in device_metrics:
            metrics = device_metrics[device]
            metrics['accuracy'] = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0
            metrics['real_accuracy'] = metrics['real_correct'] / metrics['real_total'] if metrics['real_total'] > 0 else 0
            metrics['spoof_accuracy'] = metrics['spoof_correct'] / metrics['spoof_total'] if metrics['spoof_total'] > 0 else 0
        
        return device_metrics


def create_oulu_npu_dataloader(root_dir: str, split: str = "train", batch_size: int = 32,
                              transform=None, target_transform=None) -> DataLoader:
    """
    Create OULU-NPU data loader.
    
    Args:
        root_dir: Root directory of OULU-NPU dataset
        split: Dataset split
        batch_size: Batch size
        transform: Image transformations
        target_transform: Target transformations
        
    Returns:
        Data loader
    """
    dataset = OULUNPUDataset(
        root_dir=root_dir,
        split=split,
        transform=transform,
        target_transform=target_transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader
