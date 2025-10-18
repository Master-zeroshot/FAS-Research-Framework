"""
Error handling utilities for the face anti-spoofing project.
Provides comprehensive error handling, validation, and retry mechanisms.
"""

import functools
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


class ModelError(Exception):
    """Raised when model-related operations fail."""
    pass


class DataError(Exception):
    """Raised when data-related operations fail."""
    pass


class ConfigError(Exception):
    """Raised when configuration validation fails."""
    pass


def validate_file_path(file_path: Union[str, Path], must_exist: bool = True) -> Path:
    """
    Validate a file path and return a Path object.
    
    Args:
        file_path: Path to validate
        must_exist: Whether the file must exist
        
    Returns:
        Path object if valid
        
    Raises:
        ValidationError: If path is invalid
    """
    try:
        path = Path(file_path)
        if not path.is_absolute():
            path = path.resolve()
            
        if must_exist and not path.exists():
            raise ValidationError(f"File does not exist: {path}")
            
        return path
    except Exception as e:
        raise ValidationError(f"Invalid file path '{file_path}': {e}")


def validate_model_config(config: Dict[str, Any]) -> None:
    """
    Validate model configuration parameters.
    
    Args:
        config: Model configuration dictionary
        
    Raises:
        ConfigError: If configuration is invalid
    """
    required_keys = ['model_type', 'model_size', 'embeding_dim']
    
    for key in required_keys:
        if key not in config:
            raise ConfigError(f"Missing required model config key: {key}")
    
    # Validate model type
    valid_model_types = ['Mobilenet2', 'Mobilenet3', 'Mobilenet4']
    if config['model_type'] not in valid_model_types:
        raise ConfigError(f"Invalid model_type: {config['model_type']}. Must be one of {valid_model_types}")
    
    # Validate model size
    valid_sizes = ['small', 'medium', 'large']
    if config['model_size'] not in valid_sizes:
        raise ConfigError(f"Invalid model_size: {config['model_size']}. Must be one of {valid_sizes}")
    
    # Validate embedding dimension
    if not isinstance(config['embeding_dim'], int) or config['embeding_dim'] <= 0:
        raise ConfigError(f"Invalid embeding_dim: {config['embeding_dim']}. Must be a positive integer")


def validate_training_config(config: Dict[str, Any]) -> None:
    """
    Validate training configuration parameters.
    
    Args:
        config: Training configuration dictionary
        
    Raises:
        ConfigError: If configuration is invalid
    """
    # Validate optimizer parameters
    if 'optimizer' in config:
        opt_config = config['optimizer']
        if 'lr' in opt_config and (not isinstance(opt_config['lr'], (int, float)) or opt_config['lr'] <= 0):
            raise ConfigError(f"Invalid learning rate: {opt_config['lr']}")
    
    # Validate data parameters
    if 'data' in config:
        data_config = config['data']
        if 'batch_size' in data_config and (not isinstance(data_config['batch_size'], int) or data_config['batch_size'] <= 0):
            raise ConfigError(f"Invalid batch_size: {data_config['batch_size']}")
    
    # Validate epochs
    if 'epochs' in config:
        epochs_config = config['epochs']
        if 'max_epoch' in epochs_config and (not isinstance(epochs_config['max_epoch'], int) or epochs_config['max_epoch'] <= 0):
            raise ConfigError(f"Invalid max_epoch: {epochs_config['max_epoch']}")


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry a function on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {current_delay}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}: {e}")
                        raise last_exception
            
            raise last_exception
        return wrapper
    return decorator


def validate_tensor(tensor: torch.Tensor, expected_shape: Optional[Tuple] = None, 
                   expected_dtype: Optional[torch.dtype] = None, name: str = "tensor") -> None:
    """
    Validate a PyTorch tensor.
    
    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape (None for any shape)
        expected_dtype: Expected dtype (None for any dtype)
        name: Name for error messages
        
    Raises:
        ValidationError: If tensor is invalid
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValidationError(f"{name} is not a torch.Tensor, got {type(tensor)}")
    
    if expected_shape is not None and tensor.shape != expected_shape:
        raise ValidationError(f"{name} has shape {tensor.shape}, expected {expected_shape}")
    
    if expected_dtype is not None and tensor.dtype != expected_dtype:
        raise ValidationError(f"{name} has dtype {tensor.dtype}, expected {expected_dtype}")
    
    if torch.isnan(tensor).any():
        raise ValidationError(f"{name} contains NaN values")
    
    if torch.isinf(tensor).any():
        raise ValidationError(f"{name} contains infinite values")


def safe_model_forward(model: nn.Module, input_tensor: torch.Tensor, 
                       device: str = "cpu") -> torch.Tensor:
    """
    Safely perform a forward pass through a model with error handling.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor
        device: Device to use
        
    Returns:
        Model output tensor
        
    Raises:
        ModelError: If forward pass fails
    """
    try:
        model.eval()
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        validate_tensor(output, name="model_output")
        return output
        
    except Exception as e:
        raise ModelError(f"Model forward pass failed: {e}")


def safe_data_loading(dataset, batch_size: int = 32, num_workers: int = 4, 
                     pin_memory: bool = True, shuffle: bool = True) -> torch.utils.data.DataLoader:
    """
    Safely create a DataLoader with error handling.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader instance
        
    Raises:
        DataError: If DataLoader creation fails
    """
    try:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
    except Exception as e:
        raise DataError(f"Failed to create DataLoader: {e}")


def validate_device(device: str) -> str:
    """
    Validate and normalize device string.
    
    Args:
        device: Device string (e.g., 'cuda:0', 'cpu')
        
    Returns:
        Normalized device string
        
    Raises:
        ValidationError: If device is invalid
    """
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise ValidationError("CUDA requested but not available")
        
        if ':' in device:
            gpu_id = int(device.split(':')[1])
            if gpu_id >= torch.cuda.device_count():
                raise ValidationError(f"GPU {gpu_id} not available. Only {torch.cuda.device_count()} GPUs found")
    
    elif device != 'cpu':
        raise ValidationError(f"Invalid device: {device}. Must be 'cpu' or 'cuda[:gpu_id]'")
    
    return device


def log_system_info():
    """Log system information for debugging."""
    logger.info("=== System Information ===")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    logger.info("=========================")


def handle_graceful_shutdown(signum, frame):
    """Handle graceful shutdown signals."""
    logger.info("Received shutdown signal. Cleaning up...")
    # Add cleanup code here
    exit(0)


# Register signal handlers for graceful shutdown
import signal
signal.signal(signal.SIGINT, handle_graceful_shutdown)
signal.signal(signal.SIGTERM, handle_graceful_shutdown)
