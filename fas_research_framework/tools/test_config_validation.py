"""
Unit tests for configuration validation.
"""

import pytest
import tempfile
from pathlib import Path

from utils.config_validation import (ConfigValidator, create_config_template,
                                     validate_config_file, validate_and_fix_config)


class TestConfigValidator:
    """Test configuration validator."""
    
    def test_valid_model_config(self):
        """Test validation of valid model configuration."""
        validator = ConfigValidator()
        config = {
            'model': {
                'model_type': 'Mobilenet4',
                'model_size': 'large',
                'embeding_dim': 1280,
                'pretrained': False,
                'width_mult': 1.0
            }
        }
        
        results = validator.validate(config)
        assert len(results['errors']) == 0
        assert len(results['warnings']) == 0
    
    def test_invalid_model_type(self):
        """Test validation with invalid model type."""
        validator = ConfigValidator()
        config = {
            'model': {
                'model_type': 'InvalidModel',
                'model_size': 'large',
                'embeding_dim': 1280
            }
        }
        
        results = validator.validate(config)
        assert len(results['errors']) > 0
        assert any('Invalid model_type' in error for error in results['errors'])
    
    def test_invalid_model_size(self):
        """Test validation with invalid model size."""
        validator = ConfigValidator()
        config = {
            'model': {
                'model_type': 'Mobilenet4',
                'model_size': 'invalid_size',
                'embeding_dim': 1280
            }
        }
        
        results = validator.validate(config)
        assert len(results['errors']) > 0
        assert any('Invalid model_size' in error for error in results['errors'])
    
    def test_invalid_embedding_dimension(self):
        """Test validation with invalid embedding dimension."""
        validator = ConfigValidator()
        config = {
            'model': {
                'model_type': 'Mobilenet4',
                'model_size': 'large',
                'embeding_dim': -1
            }
        }
        
        results = validator.validate(config)
        assert len(results['errors']) > 0
        assert any('Invalid embeding_dim' in error for error in results['errors'])
    
    def test_missing_required_fields(self):
        """Test validation with missing required fields."""
        validator = ConfigValidator()
        config = {
            'model': {
                'model_type': 'Mobilenet4'
                # Missing model_size and embeding_dim
            }
        }
        
        results = validator.validate(config)
        assert len(results['errors']) > 0
        assert any('Missing required model config' in error for error in results['errors'])
    
    def test_valid_data_config(self):
        """Test validation of valid data configuration."""
        validator = ConfigValidator()
        config = {
            'data': {
                'batch_size': 32,
                'data_loader_workers': 4,
                'sampler': True,
                'pin_memory': True
            }
        }
        
        results = validator.validate(config)
        assert len(results['errors']) == 0
    
    def test_invalid_batch_size(self):
        """Test validation with invalid batch size."""
        validator = ConfigValidator()
        config = {
            'data': {
                'batch_size': -1,
                'data_loader_workers': 4
            }
        }
        
        results = validator.validate(config)
        assert len(results['errors']) > 0
        assert any('Invalid batch_size' in error for error in results['errors'])
    
    def test_high_batch_size_warning(self):
        """Test warning for high batch size."""
        validator = ConfigValidator()
        config = {
            'data': {
                'batch_size': 512,
                'data_loader_workers': 4
            }
        }
        
        results = validator.validate(config)
        assert len(results['warnings']) > 0
        assert any('Large batch size' in warning for warning in results['warnings'])
    
    def test_valid_optimizer_config(self):
        """Test validation of valid optimizer configuration."""
        validator = ConfigValidator()
        config = {
            'optimizer': {
                'lr': 0.001,
                'momentum': 0.9,
                'weight_decay': 1e-4
            }
        }
        
        results = validator.validate(config)
        assert len(results['errors']) == 0
    
    def test_invalid_learning_rate(self):
        """Test validation with invalid learning rate."""
        validator = ConfigValidator()
        config = {
            'optimizer': {
                'lr': -0.001,
                'momentum': 0.9
            }
        }
        
        results = validator.validate(config)
        assert len(results['errors']) > 0
        assert any('Invalid learning rate' in error for error in results['errors'])
    
    def test_high_learning_rate_warning(self):
        """Test warning for high learning rate."""
        validator = ConfigValidator()
        config = {
            'optimizer': {
                'lr': 2.0,
                'momentum': 0.9
            }
        }
        
        results = validator.validate(config)
        assert len(results['warnings']) > 0
        assert any('High learning rate' in warning for warning in results['warnings'])


class TestConfigTemplate:
    """Test configuration template creation."""
    
    def test_create_mobilenetv4_large_template(self):
        """Test creation of MobileNetV4-Large template."""
        template = create_config_template("Mobilenet4", "large")
        
        assert template['model']['model_type'] == 'Mobilenet4'
        assert template['model']['model_size'] == 'large'
        assert template['model']['embeding_dim'] == 1280
        assert template['dataset'] == 'celeba_spoof'
        assert template['multi_task_learning'] == True
    
    def test_create_mobilenetv4_small_template(self):
        """Test creation of MobileNetV4-Small template."""
        template = create_config_template("Mobilenet4", "small")
        
        assert template['model']['model_type'] == 'Mobilenet4'
        assert template['model']['model_size'] == 'small'
        assert template['model']['embeding_dim'] == 1024
    
    def test_create_mobilenetv3_template(self):
        """Test creation of MobileNetV3 template."""
        template = create_config_template("Mobilenet3", "large")
        
        assert template['model']['model_type'] == 'Mobilenet3'
        assert template['model']['model_size'] == 'large'
        assert template['model']['embeding_dim'] == 1280


class TestConfigFileValidation:
    """Test configuration file validation."""
    
    def test_validate_nonexistent_file(self):
        """Test validation of non-existent configuration file."""
        results = validate_config_file("nonexistent_config.py")
        assert len(results['errors']) > 0
        assert any('Failed to load config file' in error for error in results['errors'])
    
    def test_validate_invalid_config_file(self, tmp_path):
        """Test validation of invalid configuration file."""
        config_file = tmp_path / "invalid_config.py"
        config_file.write_text("invalid python code")
        
        results = validate_config_file(config_file)
        assert len(results['errors']) > 0


class TestConfigFixing:
    """Test configuration fixing functionality."""
    
    def test_fix_valid_config(self):
        """Test fixing of valid configuration."""
        config = {
            'model': {
                'model_type': 'Mobilenet4',
                'model_size': 'large',
                'embeding_dim': 1280
            }
        }
        
        fixed_config = validate_and_fix_config(config)
        assert fixed_config == config
    
    def test_fix_config_with_warnings(self):
        """Test fixing of configuration with warnings."""
        config = {
            'model': {
                'model_type': 'Mobilenet4',
                'model_size': 'large',
                'embeding_dim': 5000  # Very large embedding dimension
            }
        }
        
        fixed_config = validate_and_fix_config(config)
        assert fixed_config['model']['embeding_dim'] == 1280  # Should be reduced
    
    def test_fix_config_with_errors(self):
        """Test fixing of configuration with errors."""
        config = {
            'model': {
                'model_type': 'InvalidModel',
                'model_size': 'large',
                'embeding_dim': 1280
            }
        }
        
        with pytest.raises(Exception):  # Should raise ConfigError
            validate_and_fix_config(config)


if __name__ == "__main__":
    pytest.main([__file__])
