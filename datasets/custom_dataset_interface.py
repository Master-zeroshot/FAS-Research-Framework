"""
Custom Dataset Interface for Face Anti-Spoofing

This module provides a flexible interface for creating custom datasets
for face anti-spoofing tasks with various data formats and structures.
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
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path
from abc import ABC, abstractmethod

from utils.logging_config import get_logger

logger = get_logger(__name__)


class CustomDatasetInterface(ABC):
    """Abstract base class for custom dataset interfaces."""
    
    @abstractmethod
    def load_sample(self, sample_path: str) -> Tuple[Image.Image, int, Dict[str, Any]]:
        """Load a single sample from the dataset."""
        pass
    
    @abstractmethod
    def get_sample_metadata(self, sample_path: str) -> Dict[str, Any]:
        """Get metadata for a sample."""
        pass
    
    @abstractmethod
    def validate_dataset(self) -> bool:
        """Validate dataset structure and integrity."""
        pass


class ImageFolderDataset(Dataset):
    """Custom dataset for image folder structure."""
    
    def __init__(self, root_dir: str, transform=None, target_transform=None,
                 label_mapping: Dict[str, int] = None, file_extensions: List[str] = None):
        """
        Initialize image folder dataset.
        
        Args:
            root_dir: Root directory containing subdirectories for each class
            transform: Image transformations
            target_transform: Target transformations
            label_mapping: Mapping from folder names to labels
            file_extensions: List of valid file extensions
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.label_mapping = label_mapping or {'real': 0, 'spoof': 1}
        self.file_extensions = file_extensions or ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Load samples
        self.samples = self._load_samples()
        
        logger.info(f"Loaded custom dataset with {len(self.samples)} samples")
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load dataset samples from folder structure."""
        samples = []
        
        for class_dir in self.root_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            if class_name not in self.label_mapping:
                logger.warning(f"Unknown class: {class_name}")
                continue
            
            label = self.label_mapping[class_name]
            
            # Find all valid image files
            for file_path in class_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in self.file_extensions:
                    samples.append({
                        'image_path': str(file_path),
                        'label': label,
                        'class_name': class_name,
                        'metadata': self._extract_metadata(file_path)
                    })
        
        return samples
    
    def _extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from file."""
        try:
            # Load image to get dimensions
            image = Image.open(file_path)
            width, height = image.size
            
            metadata = {
                'width': width,
                'height': height,
                'channels': 3,
                'file_size': file_path.stat().st_size,
                'filename': file_path.name,
                'relative_path': str(file_path.relative_to(self.root_dir))
            }
            
            return metadata
        except Exception as e:
            logger.warning(f"Failed to extract metadata from {file_path}: {e}")
            return {}
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Get dataset item."""
        sample = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {sample['image_path']}: {e}")
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
            'class_name': sample['class_name'],
            'image_path': sample['image_path'],
            'relative_path': sample['metadata'].get('relative_path', ''),
            'filename': sample['metadata'].get('filename', '')
        }
        
        return image, label, metadata


class JSONDataset(Dataset):
    """Custom dataset for JSON-based annotation files."""
    
    def __init__(self, root_dir: str, annotation_file: str, transform=None, 
                 target_transform=None, image_key: str = "image_path", 
                 label_key: str = "label"):
        """
        Initialize JSON dataset.
        
        Args:
            root_dir: Root directory of images
            annotation_file: Path to JSON annotation file
            transform: Image transformations
            target_transform: Target transformations
            image_key: Key for image path in JSON
            label_key: Key for label in JSON
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.image_key = image_key
        self.label_key = label_key
        
        # Load annotations
        self.annotations = self._load_annotations(annotation_file)
        
        logger.info(f"Loaded JSON dataset with {len(self.annotations)} samples")
    
    def _load_annotations(self, annotation_file: str) -> List[Dict[str, Any]]:
        """Load annotations from JSON file."""
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        # Validate annotations
        if not isinstance(annotations, list):
            raise ValueError("Annotations must be a list of dictionaries")
        
        # Add full paths
        for annotation in annotations:
            if self.image_key in annotation:
                annotation['full_image_path'] = str(self.root_dir / annotation[self.image_key])
        
        return annotations
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Get dataset item."""
        annotation = self.annotations[idx]
        
        # Load image
        image_path = annotation['full_image_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Get label
        label = torch.tensor(annotation[self.label_key], dtype=torch.long)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        # Prepare metadata
        metadata = {
            'image_path': image_path,
            'annotation': annotation
        }
        
        return image, label, metadata


class CSVDataset(Dataset):
    """Custom dataset for CSV-based annotation files."""
    
    def __init__(self, root_dir: str, csv_file: str, transform=None, 
                 target_transform=None, image_column: str = "image_path",
                 label_column: str = "label", delimiter: str = ","):
        """
        Initialize CSV dataset.
        
        Args:
            root_dir: Root directory of images
            csv_file: Path to CSV annotation file
            transform: Image transformations
            target_transform: Target transformations
            image_column: Name of image path column
            label_column: Name of label column
            delimiter: CSV delimiter
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.image_column = image_column
        self.label_column = label_column
        
        # Load CSV data
        self.data = self._load_csv(csv_file, delimiter)
        
        logger.info(f"Loaded CSV dataset with {len(self.data)} samples")
    
    def _load_csv(self, csv_file: str, delimiter: str) -> List[Dict[str, Any]]:
        """Load data from CSV file."""
        import pandas as pd
        
        df = pd.read_csv(csv_file, delimiter=delimiter)
        
        # Validate required columns
        if self.image_column not in df.columns:
            raise ValueError(f"Image column '{self.image_column}' not found in CSV")
        if self.label_column not in df.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in CSV")
        
        # Convert to list of dictionaries
        data = df.to_dict('records')
        
        # Add full paths
        for item in data:
            item['full_image_path'] = str(self.root_dir / item[self.image_column])
        
        return data
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Get dataset item."""
        item = self.data[idx]
        
        # Load image
        image_path = item['full_image_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Get label
        label = torch.tensor(int(item[self.label_column]), dtype=torch.long)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        # Prepare metadata
        metadata = {
            'image_path': image_path,
            'data': item
        }
        
        return image, label, metadata


class CustomDatasetFactory:
    """Factory for creating custom datasets."""
    
    @staticmethod
    def create_dataset(dataset_type: str, **kwargs) -> Dataset:
        """
        Create custom dataset based on type.
        
        Args:
            dataset_type: Type of dataset ('image_folder', 'json', 'csv')
            **kwargs: Additional arguments for dataset creation
            
        Returns:
            Dataset instance
        """
        if dataset_type == "image_folder":
            return ImageFolderDataset(**kwargs)
        elif dataset_type == "json":
            return JSONDataset(**kwargs)
        elif dataset_type == "csv":
            return CSVDataset(**kwargs)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    @staticmethod
    def create_dataloader(dataset: Dataset, batch_size: int = 32, 
                         shuffle: bool = True, num_workers: int = 4) -> DataLoader:
        """
        Create data loader for custom dataset.
        
        Args:
            dataset: Dataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            
        Returns:
            Data loader
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )


class DatasetValidator:
    """Validator for custom datasets."""
    
    def __init__(self, dataset: Dataset):
        """
        Initialize dataset validator.
        
        Args:
            dataset: Dataset to validate
        """
        self.dataset = dataset
    
    def validate_dataset(self) -> Dict[str, Any]:
        """
        Validate dataset integrity.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check dataset length
        if len(self.dataset) == 0:
            results['valid'] = False
            results['errors'].append("Dataset is empty")
            return results
        
        # Check sample loading
        try:
            sample = self.dataset[0]
            if len(sample) != 3:
                results['valid'] = False
                results['errors'].append("Sample should return (image, label, metadata)")
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Failed to load first sample: {e}")
        
        # Check image format
        try:
            image, label, metadata = self.dataset[0]
            if not isinstance(image, torch.Tensor):
                results['warnings'].append("Image should be a torch.Tensor")
            if not isinstance(label, torch.Tensor):
                results['warnings'].append("Label should be a torch.Tensor")
        except Exception as e:
            results['warnings'].append(f"Failed to check sample format: {e}")
        
        # Collect statistics
        results['statistics'] = self._collect_statistics()
        
        return results
    
    def _collect_statistics(self) -> Dict[str, Any]:
        """Collect dataset statistics."""
        statistics = {
            'total_samples': len(self.dataset),
            'class_distribution': {},
            'image_sizes': [],
            'metadata_keys': set()
        }
        
        # Sample a subset for statistics
        sample_size = min(100, len(self.dataset))
        indices = np.random.choice(len(self.dataset), sample_size, replace=False)
        
        for idx in indices:
            try:
                image, label, metadata = self.dataset[idx]
                
                # Class distribution
                label_val = label.item() if isinstance(label, torch.Tensor) else label
                statistics['class_distribution'][label_val] = statistics['class_distribution'].get(label_val, 0) + 1
                
                # Image sizes
                if isinstance(image, torch.Tensor):
                    statistics['image_sizes'].append(image.shape)
                
                # Metadata keys
                if isinstance(metadata, dict):
                    statistics['metadata_keys'].update(metadata.keys())
                    
            except Exception as e:
                logger.warning(f"Failed to process sample {idx}: {e}")
        
        # Convert set to list for JSON serialization
        statistics['metadata_keys'] = list(statistics['metadata_keys'])
        
        return statistics


def create_custom_dataset(dataset_type: str, root_dir: str, **kwargs) -> Dataset:
    """
    Create custom dataset.
    
    Args:
        dataset_type: Type of dataset
        root_dir: Root directory
        **kwargs: Additional arguments
        
    Returns:
        Dataset instance
    """
    return CustomDatasetFactory.create_dataset(
        dataset_type=dataset_type,
        root_dir=root_dir,
        **kwargs
    )


def validate_custom_dataset(dataset: Dataset) -> Dict[str, Any]:
    """
    Validate custom dataset.
    
    Args:
        dataset: Dataset to validate
        
    Returns:
        Validation results
    """
    validator = DatasetValidator(dataset)
    return validator.validate_dataset()
