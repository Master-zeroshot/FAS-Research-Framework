"""
Structured logging configuration for the face anti-spoofing project.
Provides comprehensive logging with different levels and structured output.
"""

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from configs.paths import get_log_path, ensure_directories


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format the message
        formatted = super().format(record)
        
        # Add color to the level name
        formatted = formatted.replace(
            record.levelname,
            f"{color}{record.levelname}{reset}"
        )
        
        return formatted


class TrainingLogger:
    """Specialized logger for training metrics."""
    
    def __init__(self, experiment_path: str):
        """Initialize training logger."""
        self.experiment_path = Path(experiment_path)
        self.experiment_path.mkdir(parents=True, exist_ok=True)
        
        # Create training metrics logger
        self.metrics_logger = logging.getLogger('training_metrics')
        self.metrics_logger.setLevel(logging.INFO)
        
        # Create metrics file handler
        metrics_file = self.experiment_path / 'training_metrics.jsonl'
        metrics_handler = logging.FileHandler(metrics_file)
        metrics_handler.setFormatter(StructuredFormatter())
        self.metrics_logger.addHandler(metrics_handler)
        
        # Prevent propagation to avoid duplicate logs
        self.metrics_logger.propagate = False
    
    def log_epoch(self, epoch: int, train_loss: float, train_acc: float, 
                  val_loss: float, val_acc: float, lr: float):
        """Log epoch metrics."""
        self.metrics_logger.info(
            f"Epoch {epoch} completed",
            extra={
                'extra_fields': {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'learning_rate': lr,
                    'metric_type': 'epoch'
                }
            }
        )
    
    def log_batch(self, epoch: int, batch: int, loss: float, acc: float, lr: float):
        """Log batch metrics."""
        self.metrics_logger.info(
            f"Batch {batch} in epoch {epoch}",
            extra={
                'extra_fields': {
                    'epoch': epoch,
                    'batch': batch,
                    'loss': loss,
                    'accuracy': acc,
                    'learning_rate': lr,
                    'metric_type': 'batch'
                }
            }
        )
    
    def log_evaluation(self, epoch: int, auc: float, eer: float, acer: float):
        """Log evaluation metrics."""
        self.metrics_logger.info(
            f"Evaluation at epoch {epoch}",
            extra={
                'extra_fields': {
                    'epoch': epoch,
                    'auc': auc,
                    'eer': eer,
                    'acer': acer,
                    'metric_type': 'evaluation'
                }
            }
        )


def setup_logging(log_level: str = "INFO", log_to_file: bool = True, 
                  log_to_console: bool = True, structured: bool = True) -> logging.Logger:
    """
    Setup comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        structured: Whether to use structured logging
        
    Returns:
        Configured logger
    """
    # Ensure directories exist
    ensure_directories()
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    if structured:
        file_formatter = StructuredFormatter()
        console_formatter = ColoredFormatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = ColoredFormatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # File handler
    if log_to_file:
        log_file = get_log_path('app_log')
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        root_logger.addHandler(console_handler)
    
    # Set specific logger levels
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)


def log_model_info(model: Any, input_shape: tuple, device: str):
    """Log model information."""
    logger = get_logger('model_info')
    
    logger.info("=== Model Information ===")
    logger.info(f"Model type: {type(model).__name__}")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Device: {device}")
    
    if hasattr(model, 'parameters'):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    logger.info("========================")


def log_data_info(dataset_size: int, batch_size: int, num_batches: int, 
                  num_workers: int):
    """Log dataset information."""
    logger = get_logger('data_info')
    
    logger.info("=== Dataset Information ===")
    logger.info(f"Dataset size: {dataset_size:,}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Number of batches: {num_batches}")
    logger.info(f"Number of workers: {num_workers}")
    logger.info("==========================")


def log_training_start(config: Dict[str, Any], device: str):
    """Log training start information."""
    logger = get_logger('training')
    
    logger.info("=== Training Started ===")
    logger.info(f"Device: {device}")
    logger.info(f"Model: {config.get('model', {}).get('model_type', 'Unknown')}")
    logger.info(f"Dataset: {config.get('dataset', 'Unknown')}")
    logger.info(f"Batch size: {config.get('data', {}).get('batch_size', 'Unknown')}")
    logger.info(f"Max epochs: {config.get('epochs', {}).get('max_epoch', 'Unknown')}")
    logger.info("========================")


def log_training_end(total_time: float, best_epoch: int, best_metrics: Dict[str, float]):
    """Log training end information."""
    logger = get_logger('training')
    
    logger.info("=== Training Completed ===")
    logger.info(f"Total training time: {total_time:.2f} seconds")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best metrics: {best_metrics}")
    logger.info("=========================")


# Performance monitoring
class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self, logger_name: str = 'performance'):
        """Initialize performance monitor."""
        self.logger = get_logger(logger_name)
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = datetime.now()
        self.logger.debug(f"Started timing: {operation}")
    
    def end_timer(self, operation: str):
        """End timing an operation and log the duration."""
        if operation in self.start_times:
            duration = (datetime.now() - self.start_times[operation]).total_seconds()
            self.logger.info(f"Operation '{operation}' took {duration:.3f} seconds")
            del self.start_times[operation]
        else:
            self.logger.warning(f"No start time found for operation: {operation}")
    
    def log_memory_usage(self):
        """Log current memory usage."""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        self.logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")
    
    def log_gpu_usage(self):
        """Log GPU memory usage if available."""
        if hasattr(torch.cuda, 'is_available') and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
                cached = torch.cuda.memory_reserved(i) / 1024 / 1024
                self.logger.info(f"GPU {i} memory: {allocated:.1f} MB allocated, {cached:.1f} MB cached")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
