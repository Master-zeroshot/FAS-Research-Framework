"""
Configuration validation system for the face anti-spoofing project.
Provides comprehensive validation of configuration parameters.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from utils.error_handling import ConfigError, ValidationError


class ConfigValidator:
    """Comprehensive configuration validator."""
    
    def __init__(self):
        """Initialize the validator."""
        self.errors = []
        self.warnings = []
    
    def validate(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Validate a complete configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Dictionary with 'errors' and 'warnings' lists
        """
        self.errors = []
        self.warnings = []
        
        # Validate all sections
        self._validate_model_config(config.get('model', {}))
        self._validate_data_config(config.get('data', {}))
        self._validate_optimizer_config(config.get('optimizer', {}))
        self._validate_loss_config(config.get('loss', {}))
        self._validate_augmentation_config(config.get('aug', {}))
        self._validate_checkpoint_config(config.get('checkpoint', {}))
        self._validate_datasets_config(config.get('datasets', {}))
        self._validate_epochs_config(config.get('epochs', {}))
        
        return {
            'errors': self.errors,
            'warnings': self.warnings
        }
    
    def _validate_model_config(self, model_config: Dict[str, Any]):
        """Validate model configuration."""
        required_keys = ['model_type', 'model_size', 'embeding_dim']
        
        for key in required_keys:
            if key not in model_config:
                self.errors.append(f"Missing required model config: {key}")
        
        # Validate model type
        if 'model_type' in model_config:
            valid_types = ['Mobilenet2', 'Mobilenet3', 'Mobilenet4']
            if model_config['model_type'] not in valid_types:
                self.errors.append(f"Invalid model_type: {model_config['model_type']}. Must be one of {valid_types}")
        
        # Validate model size
        if 'model_size' in model_config:
            valid_sizes = ['small', 'medium', 'large']
            if model_config['model_size'] not in valid_sizes:
                self.errors.append(f"Invalid model_size: {model_config['model_size']}. Must be one of {valid_sizes}")
        
        # Validate embedding dimension
        if 'embeding_dim' in model_config:
            emb_dim = model_config['embeding_dim']
            if not isinstance(emb_dim, int) or emb_dim <= 0:
                self.errors.append(f"Invalid embeding_dim: {emb_dim}. Must be a positive integer")
            
            # Check if embedding dimension is reasonable
            if emb_dim > 4096:
                self.warnings.append(f"Very large embedding dimension: {emb_dim}")
        
        # Validate pretrained setting
        if 'pretrained' in model_config:
            if not isinstance(model_config['pretrained'], bool):
                self.errors.append(f"Invalid pretrained: {model_config['pretrained']}. Must be boolean")
        
        # Validate width multiplier
        if 'width_mult' in model_config:
            width_mult = model_config['width_mult']
            if not isinstance(width_mult, (int, float)) or width_mult <= 0:
                self.errors.append(f"Invalid width_mult: {width_mult}. Must be positive number")
    
    def _validate_data_config(self, data_config: Dict[str, Any]):
        """Validate data configuration."""
        # Validate batch size
        if 'batch_size' in data_config:
            batch_size = data_config['batch_size']
            if not isinstance(batch_size, int) or batch_size <= 0:
                self.errors.append(f"Invalid batch_size: {batch_size}. Must be positive integer")
            elif batch_size > 256:
                self.warnings.append(f"Large batch size: {batch_size}. May cause memory issues")
        
        # Validate number of workers
        if 'data_loader_workers' in data_config:
            num_workers = data_config['data_loader_workers']
            if not isinstance(num_workers, int) or num_workers < 0:
                self.errors.append(f"Invalid data_loader_workers: {num_workers}. Must be non-negative integer")
            elif num_workers > 16:
                self.warnings.append(f"High number of workers: {num_workers}. May cause system issues")
        
        # Validate sampler setting
        if 'sampler' in data_config:
            if not isinstance(data_config['sampler'], bool):
                self.errors.append(f"Invalid sampler: {data_config['sampler']}. Must be boolean")
        
        # Validate pin_memory setting
        if 'pin_memory' in data_config:
            if not isinstance(data_config['pin_memory'], bool):
                self.errors.append(f"Invalid pin_memory: {data_config['pin_memory']}. Must be boolean")
    
    def _validate_optimizer_config(self, optimizer_config: Dict[str, Any]):
        """Validate optimizer configuration."""
        # Validate learning rate
        if 'lr' in optimizer_config:
            lr = optimizer_config['lr']
            if not isinstance(lr, (int, float)) or lr <= 0:
                self.errors.append(f"Invalid learning rate: {lr}. Must be positive number")
            elif lr > 1.0:
                self.warnings.append(f"High learning rate: {lr}. May cause training instability")
            elif lr < 1e-6:
                self.warnings.append(f"Very low learning rate: {lr}. May cause slow convergence")
        
        # Validate momentum
        if 'momentum' in optimizer_config:
            momentum = optimizer_config['momentum']
            if not isinstance(momentum, (int, float)) or not (0 <= momentum < 1):
                self.errors.append(f"Invalid momentum: {momentum}. Must be in range [0, 1)")
        
        # Validate weight decay
        if 'weight_decay' in optimizer_config:
            weight_decay = optimizer_config['weight_decay']
            if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
                self.errors.append(f"Invalid weight_decay: {weight_decay}. Must be non-negative")
            elif weight_decay > 1.0:
                self.warnings.append(f"High weight decay: {weight_decay}. May cause underfitting")
    
    def _validate_loss_config(self, loss_config: Dict[str, Any]):
        """Validate loss configuration."""
        if 'loss_type' in loss_config:
            valid_types = ['amsoftmax', 'cross_entropy', 'bce']
            if loss_config['loss_type'] not in valid_types:
                self.errors.append(f"Invalid loss_type: {loss_config['loss_type']}. Must be one of {valid_types}")
        
        # Validate AM-Softmax parameters
        if 'amsoftmax' in loss_config:
            amsoftmax_config = loss_config['amsoftmax']
            
            if 'm' in amsoftmax_config:
                m = amsoftmax_config['m']
                if not isinstance(m, (int, float)) or m < 0:
                    self.errors.append(f"Invalid AM-Softmax margin: {m}. Must be non-negative")
            
            if 's' in amsoftmax_config:
                s = amsoftmax_config['s']
                if not isinstance(s, (int, float)) or s <= 0:
                    self.errors.append(f"Invalid AM-Softmax scale: {s}. Must be positive")
    
    def _validate_augmentation_config(self, aug_config: Dict[str, Any]):
        """Validate augmentation configuration."""
        if 'type_aug' in aug_config:
            valid_types = [None, 'mixup', 'cutmix']
            if aug_config['type_aug'] not in valid_types:
                self.errors.append(f"Invalid type_aug: {aug_config['type_aug']}. Must be one of {valid_types}")
        
        # Validate augmentation parameters
        for param in ['alpha', 'beta', 'aug_prob']:
            if param in aug_config:
                value = aug_config[param]
                if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                    self.errors.append(f"Invalid {param}: {value}. Must be in range [0, 1]")
    
    def _validate_checkpoint_config(self, checkpoint_config: Dict[str, Any]):
        """Validate checkpoint configuration."""
        if 'snapshot_name' in checkpoint_config:
            snapshot_name = checkpoint_config['snapshot_name']
            if not isinstance(snapshot_name, str) or not snapshot_name:
                self.errors.append(f"Invalid snapshot_name: {snapshot_name}. Must be non-empty string")
        
        if 'experiment_path' in checkpoint_config:
            experiment_path = checkpoint_config['experiment_path']
            if not isinstance(experiment_path, str) or not experiment_path:
                self.errors.append(f"Invalid experiment_path: {experiment_path}. Must be non-empty string")
    
    def _validate_datasets_config(self, datasets_config: Dict[str, Any]):
        """Validate datasets configuration."""
        for dataset_name, dataset_path in datasets_config.items():
            if not isinstance(dataset_path, str):
                self.errors.append(f"Invalid {dataset_name} path: {dataset_path}. Must be string")
            elif dataset_path and not os.path.exists(dataset_path):
                self.warnings.append(f"Dataset path does not exist: {dataset_name} = {dataset_path}")
    
    def _validate_epochs_config(self, epochs_config: Dict[str, Any]):
        """Validate epochs configuration."""
        if 'start_epoch' in epochs_config:
            start_epoch = epochs_config['start_epoch']
            if not isinstance(start_epoch, int) or start_epoch < 0:
                self.errors.append(f"Invalid start_epoch: {start_epoch}. Must be non-negative integer")
        
        if 'max_epoch' in epochs_config:
            max_epoch = epochs_config['max_epoch']
            if not isinstance(max_epoch, int) or max_epoch <= 0:
                self.errors.append(f"Invalid max_epoch: {max_epoch}. Must be positive integer")
            
            if 'start_epoch' in epochs_config:
                start_epoch = epochs_config['start_epoch']
                if start_epoch >= max_epoch:
                    self.errors.append(f"start_epoch ({start_epoch}) must be less than max_epoch ({max_epoch})")


def validate_config_file(config_path: Union[str, Path]) -> Dict[str, List[str]]:
    """
    Validate a configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with validation results
    """
    try:
        from utils import read_py_config
        config = read_py_config(str(config_path))
        
        validator = ConfigValidator()
        return validator.validate(config)
    
    except Exception as e:
        return {
            'errors': [f"Failed to load config file: {e}"],
            'warnings': []
        }


def create_config_template(model_type: str = "Mobilenet4", model_size: str = "large") -> Dict[str, Any]:
    """
    Create a configuration template with default values.
    
    Args:
        model_type: Type of model (Mobilenet2, Mobilenet3, Mobilenet4)
        model_size: Size of model (small, medium, large)
        
    Returns:
        Configuration template dictionary
    """
    # Default embedding dimensions
    embedding_dims = {
        'Mobilenet2': {'small': 1280, 'medium': 1280, 'large': 1280},
        'Mobilenet3': {'small': 1024, 'medium': 1024, 'large': 1280},
        'Mobilenet4': {'small': 1024, 'medium': 1152, 'large': 1280}
    }
    
    template = {
        'exp_num': 0,
        'dataset': 'celeba_spoof',
        'multi_task_learning': True,
        'evaluation': True,
        'test_steps': None,
        'random_seed': 42,
        
        'datasets': {
            'Celeba_root': '../CelebA-Spoof-zips/CelebA_Spoof',
            'Casia_root': './CASIA',
            'LCCFASD_root': '../LCC_FASD'
        },
        
        'img_norm_cfg': {
            'mean': [0.5931, 0.4690, 0.4229],
            'std': [0.2471, 0.2214, 0.2157]
        },
        
        'optimizer': {
            'lr': 0.005,
            'momentum': 0.9,
            'weight_decay': 5e-4
        },
        
        'scheduler': {
            'milestones': [20, 50],
            'gamma': 0.2
        },
        
        'data': {
            'batch_size': 32,
            'data_loader_workers': 8,
            'sampler': True,
            'pin_memory': True
        },
        
        'resize': {
            'height': 224,
            'width': 224
        },
        
        'checkpoint': {
            'snapshot_name': f"{model_type.lower()}_{model_size}_224_antispoof.pth.tar",
            'experiment_path': f'./logs_{model_type.lower()}_{model_size}'
        },
        
        'loss': {
            'loss_type': 'amsoftmax',
            'amsoftmax': {
                'm': 0.5,
                's': 1,
                'margin_type': 'cross_entropy',
                'label_smooth': False,
                'smoothing': 0.1,
                'ratio': [1, 1],
                'gamma': 0
            }
        },
        
        'epochs': {
            'start_epoch': 0,
            'max_epoch': 71
        },
        
        'model': {
            'model_type': model_type,
            'model_size': model_size,
            'width_mult': 1.0,
            'pretrained': False,
            'embeding_dim': embedding_dims.get(model_type, {}).get(model_size, 1280),
            'imagenet_weights': f'./pretrained/{model_type.lower()}_{model_size}_imagenet.pth.tar'
        },
        
        'aug': {
            'type_aug': None,
            'alpha': 0.5,
            'beta': 0.5,
            'aug_prob': 0.7
        },
        
        'curves': {
            'det_curve': f'det_curve_{model_type.lower()}_{model_size}.png',
            'roc_curve': f'roc_curve_{model_type.lower()}_{model_size}.png'
        },
        
        'dropout': {
            'prob_dropout': 0.2,
            'classifier': 0.3,
            'type': 'gaussian',
            'mu': 0.5,
            'sigma': 0.3
        },
        
        'RSC': {
            'use_rsc': False,
            'p': 0.333,
            'b': 0.333
        },
        
        'test_dataset': {
            'type': 'LCC_FASD'
        },
        
        'conv_cd': {
            'theta': 0
        },
        
        'test_file_name': f"test_{model_type.lower()}_{model_size}"
    }
    
    return template


def save_config_template(config: Dict[str, Any], output_path: Union[str, Path]):
    """Save configuration template to file."""
    output_path = Path(output_path)
    
    # Create Python config file
    with open(output_path, 'w') as f:
        f.write(f'"""Configuration template for {config["model"]["model_type"]} {config["model"]["model_size"]}"""\n\n')
        
        for key, value in config.items():
            if isinstance(value, dict):
                f.write(f'{key} = dict(\n')
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, str):
                        f.write(f'    {sub_key}="{sub_value}",\n')
                    else:
                        f.write(f'    {sub_key}={sub_value},\n')
                f.write(')\n\n')
            elif isinstance(value, str):
                f.write(f'{key} = "{value}"\n\n')
            else:
                f.write(f'{key} = {value}\n\n')


def validate_and_fix_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration and attempt to fix common issues.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Fixed configuration dictionary
    """
    validator = ConfigValidator()
    results = validator.validate(config)
    
    # If there are errors, we can't auto-fix
    if results['errors']:
        raise ConfigError(f"Configuration has errors that cannot be auto-fixed: {results['errors']}")
    
    # Apply fixes for warnings
    fixed_config = config.copy()
    
    # Fix common issues
    if 'model' in fixed_config:
        model_config = fixed_config['model']
        
        # Ensure embedding dimension is reasonable
        if 'embeding_dim' in model_config:
            emb_dim = model_config['embeding_dim']
            if emb_dim > 4096:
                model_config['embeding_dim'] = 1280
                print(f"Warning: Reduced embedding dimension from {emb_dim} to 1280")
    
    return fixed_config
