"""
Dataset Conversion Tools for Face Anti-Spoofing

This module provides tools for converting between different dataset formats
and structures for face anti-spoofing tasks.
"""

import os
import json
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from PIL import Image
import cv2

from utils.logging_config import get_logger

logger = get_logger(__name__)


class DatasetConverter:
    """Base class for dataset conversion tools."""
    
    def __init__(self, source_dir: str, target_dir: str):
        """
        Initialize dataset converter.
        
        Args:
            source_dir: Source dataset directory
            target_dir: Target dataset directory
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        
        # Create target directory
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized dataset converter: {source_dir} -> {target_dir}")
    
    def convert(self) -> Dict[str, Any]:
        """Convert dataset format."""
        raise NotImplementedError("Subclasses must implement convert method")
    
    def validate_conversion(self) -> bool:
        """Validate conversion results."""
        raise NotImplementedError("Subclasses must implement validate_conversion method")


class ImageFolderToJSONConverter(DatasetConverter):
    """Convert image folder structure to JSON format."""
    
    def __init__(self, source_dir: str, target_dir: str, 
                 label_mapping: Dict[str, int] = None):
        """
        Initialize image folder to JSON converter.
        
        Args:
            source_dir: Source image folder directory
            target_dir: Target JSON directory
            label_mapping: Mapping from folder names to labels
        """
        super().__init__(source_dir, target_dir)
        self.label_mapping = label_mapping or {'real': 0, 'spoof': 1}
    
    def convert(self) -> Dict[str, Any]:
        """Convert image folder to JSON format."""
        annotations = []
        statistics = {
            'total_samples': 0,
            'class_distribution': {},
            'converted_files': 0,
            'failed_files': 0
        }
        
        # Process each class folder
        for class_dir in self.source_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            if class_name not in self.label_mapping:
                logger.warning(f"Unknown class: {class_name}")
                continue
            
            label = self.label_mapping[class_name]
            class_samples = 0
            
            # Process images in class folder
            for image_path in class_dir.rglob("*.jpg"):
                if not image_path.is_file():
                    continue
                
                try:
                    # Create relative path
                    relative_path = str(image_path.relative_to(self.source_dir))
                    
                    # Extract metadata
                    metadata = self._extract_image_metadata(image_path)
                    
                    # Create annotation
                    annotation = {
                        'image_path': relative_path,
                        'label': label,
                        'class_name': class_name,
                        'metadata': metadata
                    }
                    
                    annotations.append(annotation)
                    class_samples += 1
                    statistics['converted_files'] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process {image_path}: {e}")
                    statistics['failed_files'] += 1
            
            statistics['class_distribution'][class_name] = class_samples
            statistics['total_samples'] += class_samples
        
        # Save annotations
        self._save_annotations(annotations)
        
        # Save statistics
        self._save_statistics(statistics)
        
        logger.info(f"Conversion completed: {statistics['converted_files']} files converted")
        return statistics
    
    def _extract_image_metadata(self, image_path: Path) -> Dict[str, Any]:
        """Extract metadata from image."""
        try:
            image = Image.open(image_path)
            width, height = image.size
            
            metadata = {
                'width': width,
                'height': height,
                'channels': 3,
                'file_size': image_path.stat().st_size,
                'filename': image_path.name
            }
            
            return metadata
        except Exception as e:
            logger.warning(f"Failed to extract metadata from {image_path}: {e}")
            return {}
    
    def _save_annotations(self, annotations: List[Dict[str, Any]]) -> None:
        """Save annotations to JSON file."""
        annotations_file = self.target_dir / "annotations.json"
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        logger.info(f"Saved annotations to {annotations_file}")
    
    def _save_statistics(self, statistics: Dict[str, Any]) -> None:
        """Save conversion statistics."""
        stats_file = self.target_dir / "conversion_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(statistics, f, indent=2)
        
        logger.info(f"Saved statistics to {stats_file}")
    
    def validate_conversion(self) -> bool:
        """Validate conversion results."""
        annotations_file = self.target_dir / "annotations.json"
        
        if not annotations_file.exists():
            logger.error("Annotations file not found")
            return False
        
        try:
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
            
            if not isinstance(annotations, list):
                logger.error("Annotations must be a list")
                return False
            
            if len(annotations) == 0:
                logger.error("No annotations found")
                return False
            
            # Validate annotation structure
            for annotation in annotations:
                if not all(key in annotation for key in ['image_path', 'label', 'class_name']):
                    logger.error("Invalid annotation structure")
                    return False
            
            logger.info("Conversion validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False


class JSONToCSVConverter(DatasetConverter):
    """Convert JSON annotations to CSV format."""
    
    def __init__(self, source_dir: str, target_dir: str, 
                 json_file: str = "annotations.json"):
        """
        Initialize JSON to CSV converter.
        
        Args:
            source_dir: Source JSON directory
            target_dir: Target CSV directory
            json_file: JSON annotations file name
        """
        super().__init__(source_dir, target_dir)
        self.json_file = json_file
    
    def convert(self) -> Dict[str, Any]:
        """Convert JSON to CSV format."""
        json_path = self.source_dir / self.json_file
        
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        # Load JSON annotations
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(annotations)
        
        # Flatten metadata if present
        if 'metadata' in df.columns:
            metadata_df = pd.json_normalize(df['metadata'])
            df = pd.concat([df.drop('metadata', axis=1), metadata_df], axis=1)
        
        # Save CSV
        csv_file = self.target_dir / "annotations.csv"
        df.to_csv(csv_file, index=False)
        
        statistics = {
            'total_samples': len(df),
            'columns': list(df.columns),
            'class_distribution': df['label'].value_counts().to_dict() if 'label' in df.columns else {}
        }
        
        logger.info(f"Converted JSON to CSV: {len(df)} samples")
        return statistics
    
    def validate_conversion(self) -> bool:
        """Validate CSV conversion."""
        csv_file = self.target_dir / "annotations.csv"
        
        if not csv_file.exists():
            logger.error("CSV file not found")
            return False
        
        try:
            df = pd.read_csv(csv_file)
            
            if len(df) == 0:
                logger.error("CSV file is empty")
                return False
            
            required_columns = ['image_path', 'label']
            if not all(col in df.columns for col in required_columns):
                logger.error("Required columns missing")
                return False
            
            logger.info("CSV conversion validation passed")
            return True
            
        except Exception as e:
            logger.error(f"CSV validation failed: {e}")
            return False


class DatasetSplitter:
    """Split dataset into train/val/test sets."""
    
    def __init__(self, source_dir: str, target_dir: str, 
                 train_ratio: float = 0.7, val_ratio: float = 0.15, 
                 test_ratio: float = 0.15, random_seed: int = 42):
        """
        Initialize dataset splitter.
        
        Args:
            source_dir: Source dataset directory
            target_dir: Target directory for splits
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_seed: Random seed for reproducibility
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        # Create target directories
        for split in ['train', 'val', 'test']:
            (self.target_dir / split).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized dataset splitter with ratios: {train_ratio}/{val_ratio}/{test_ratio}")
    
    def split_dataset(self, dataset_type: str = "image_folder") -> Dict[str, Any]:
        """
        Split dataset into train/val/test sets.
        
        Args:
            dataset_type: Type of dataset to split
            
        Returns:
            Dictionary with split statistics
        """
        if dataset_type == "image_folder":
            return self._split_image_folder()
        elif dataset_type == "json":
            return self._split_json()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def _split_image_folder(self) -> Dict[str, Any]:
        """Split image folder dataset."""
        statistics = {
            'train': {'samples': 0, 'classes': {}},
            'val': {'samples': 0, 'classes': {}},
            'test': {'samples': 0, 'classes': {}}
        }
        
        # Process each class
        for class_dir in self.source_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            
            # Get all images in class
            image_files = list(class_dir.glob("*.jpg"))
            if len(image_files) == 0:
                continue
            
            # Shuffle images
            np.random.seed(self.random_seed)
            np.random.shuffle(image_files)
            
            # Calculate split indices
            n_images = len(image_files)
            train_end = int(n_images * self.train_ratio)
            val_end = train_end + int(n_images * self.val_ratio)
            
            # Split images
            train_files = image_files[:train_end]
            val_files = image_files[train_end:val_end]
            test_files = image_files[val_end:]
            
            # Copy files to split directories
            self._copy_files_to_split(train_files, class_name, 'train', statistics)
            self._copy_files_to_split(val_files, class_name, 'val', statistics)
            self._copy_files_to_split(test_files, class_name, 'test', statistics)
        
        logger.info(f"Dataset split completed: {statistics}")
        return statistics
    
    def _split_json(self) -> Dict[str, Any]:
        """Split JSON dataset."""
        json_file = self.source_dir / "annotations.json"
        
        if not json_file.exists():
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        
        # Load annotations
        with open(json_file, 'r') as f:
            annotations = json.load(f)
        
        # Group by class
        class_annotations = {}
        for annotation in annotations:
            class_name = annotation.get('class_name', 'unknown')
            if class_name not in class_annotations:
                class_annotations[class_name] = []
            class_annotations[class_name].append(annotation)
        
        statistics = {
            'train': {'samples': 0, 'classes': {}},
            'val': {'samples': 0, 'classes': {}},
            'test': {'samples': 0, 'classes': {}}
        }
        
        # Split each class
        for class_name, class_anns in class_annotations.items():
            # Shuffle annotations
            np.random.seed(self.random_seed)
            np.random.shuffle(class_anns)
            
            # Calculate split indices
            n_anns = len(class_anns)
            train_end = int(n_anns * self.train_ratio)
            val_end = train_end + int(n_anns * self.val_ratio)
            
            # Split annotations
            train_anns = class_anns[:train_end]
            val_anns = class_anns[train_end:val_end]
            test_anns = class_anns[val_end:]
            
            # Save split annotations
            self._save_split_annotations(train_anns, 'train', statistics)
            self._save_split_annotations(val_anns, 'val', statistics)
            self._save_split_annotations(test_anns, 'test', statistics)
        
        logger.info(f"JSON dataset split completed: {statistics}")
        return statistics
    
    def _copy_files_to_split(self, files: List[Path], class_name: str, 
                            split: str, statistics: Dict[str, Any]) -> None:
        """Copy files to split directory."""
        split_dir = self.target_dir / split / class_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for file_path in files:
            target_path = split_dir / file_path.name
            shutil.copy2(file_path, target_path)
        
        statistics[split]['samples'] += len(files)
        statistics[split]['classes'][class_name] = len(files)
    
    def _save_split_annotations(self, annotations: List[Dict[str, Any]], 
                               split: str, statistics: Dict[str, Any]) -> None:
        """Save split annotations."""
        split_file = self.target_dir / split / "annotations.json"
        
        with open(split_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        statistics[split]['samples'] += len(annotations)
        
        # Count classes
        for annotation in annotations:
            class_name = annotation.get('class_name', 'unknown')
            statistics[split]['classes'][class_name] = statistics[split]['classes'].get(class_name, 0) + 1


class DatasetMerger:
    """Merge multiple datasets into one."""
    
    def __init__(self, source_dirs: List[str], target_dir: str):
        """
        Initialize dataset merger.
        
        Args:
            source_dirs: List of source dataset directories
            target_dir: Target merged dataset directory
        """
        self.source_dirs = [Path(d) for d in source_dirs]
        self.target_dir = Path(target_dir)
        
        # Create target directory
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized dataset merger: {len(source_dirs)} datasets -> {target_dir}")
    
    def merge_datasets(self, dataset_type: str = "image_folder") -> Dict[str, Any]:
        """
        Merge datasets.
        
        Args:
            dataset_type: Type of datasets to merge
            
        Returns:
            Dictionary with merge statistics
        """
        if dataset_type == "image_folder":
            return self._merge_image_folders()
        elif dataset_type == "json":
            return self._merge_json_datasets()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def _merge_image_folders(self) -> Dict[str, Any]:
        """Merge image folder datasets."""
        statistics = {
            'total_samples': 0,
            'class_distribution': {},
            'source_datasets': len(self.source_dirs)
        }
        
        # Process each source dataset
        for i, source_dir in enumerate(self.source_dirs):
            if not source_dir.exists():
                logger.warning(f"Source directory not found: {source_dir}")
                continue
            
            # Copy all class folders
            for class_dir in source_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                
                class_name = class_dir.name
                target_class_dir = self.target_dir / class_name
                target_class_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy images
                for image_file in class_dir.glob("*.jpg"):
                    target_file = target_class_dir / f"dataset_{i}_{image_file.name}"
                    shutil.copy2(image_file, target_file)
                    
                    statistics['total_samples'] += 1
                    statistics['class_distribution'][class_name] = statistics['class_distribution'].get(class_name, 0) + 1
        
        logger.info(f"Image folder merge completed: {statistics['total_samples']} samples")
        return statistics
    
    def _merge_json_datasets(self) -> Dict[str, Any]:
        """Merge JSON datasets."""
        all_annotations = []
        statistics = {
            'total_samples': 0,
            'class_distribution': {},
            'source_datasets': len(self.source_dirs)
        }
        
        # Process each source dataset
        for i, source_dir in enumerate(self.source_dirs):
            json_file = source_dir / "annotations.json"
            
            if not json_file.exists():
                logger.warning(f"JSON file not found: {json_file}")
                continue
            
            # Load annotations
            with open(json_file, 'r') as f:
                annotations = json.load(f)
            
            # Add dataset identifier
            for annotation in annotations:
                annotation['source_dataset'] = i
                all_annotations.append(annotation)
                
                statistics['total_samples'] += 1
                class_name = annotation.get('class_name', 'unknown')
                statistics['class_distribution'][class_name] = statistics['class_distribution'].get(class_name, 0) + 1
        
        # Save merged annotations
        merged_file = self.target_dir / "annotations.json"
        with open(merged_file, 'w') as f:
            json.dump(all_annotations, f, indent=2)
        
        logger.info(f"JSON dataset merge completed: {statistics['total_samples']} samples")
        return statistics


def convert_dataset(source_dir: str, target_dir: str, 
                   conversion_type: str, **kwargs) -> Dict[str, Any]:
    """
    Convert dataset format.
    
    Args:
        source_dir: Source dataset directory
        target_dir: Target dataset directory
        conversion_type: Type of conversion
        **kwargs: Additional arguments
        
    Returns:
        Conversion statistics
    """
    if conversion_type == "image_folder_to_json":
        converter = ImageFolderToJSONConverter(source_dir, target_dir, **kwargs)
    elif conversion_type == "json_to_csv":
        converter = JSONToCSVConverter(source_dir, target_dir, **kwargs)
    else:
        raise ValueError(f"Unknown conversion type: {conversion_type}")
    
    return converter.convert()


def split_dataset(source_dir: str, target_dir: str, 
                 train_ratio: float = 0.7, val_ratio: float = 0.15, 
                 test_ratio: float = 0.15, dataset_type: str = "image_folder") -> Dict[str, Any]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        source_dir: Source dataset directory
        target_dir: Target directory for splits
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        dataset_type: Type of dataset to split
        
    Returns:
        Split statistics
    """
    splitter = DatasetSplitter(source_dir, target_dir, train_ratio, val_ratio, test_ratio)
    return splitter.split_dataset(dataset_type)


def merge_datasets(source_dirs: List[str], target_dir: str, 
                  dataset_type: str = "image_folder") -> Dict[str, Any]:
    """
    Merge multiple datasets.
    
    Args:
        source_dirs: List of source dataset directories
        target_dir: Target merged dataset directory
        dataset_type: Type of datasets to merge
        
    Returns:
        Merge statistics
    """
    merger = DatasetMerger(source_dirs, target_dir)
    return merger.merge_datasets(dataset_type)
