"""
Unit tests for model architectures and functionality.
"""

import pytest
import torch
import torch.nn as nn

from models import (mobilenetv2, mobilenetv3_large, mobilenetv3_small,
                    mobilenetv4_large, mobilenetv4_medium, mobilenetv4_small)
from utils.error_handling import ModelError, validate_tensor


class TestModelCreation:
    """Test model creation and basic functionality."""
    
    def test_mobilenetv2_creation(self):
        """Test MobileNetV2 model creation."""
        model = mobilenetv2(width_mult=1.0, pretrained=False, embeding_dim=1280)
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'forward')
        assert hasattr(model, 'make_logits')
    
    def test_mobilenetv3_small_creation(self):
        """Test MobileNetV3-Small model creation."""
        model = mobilenetv3_small(width_mult=1.0, pretrained=False, embeding_dim=1024)
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'forward')
        assert hasattr(model, 'make_logits')
    
    def test_mobilenetv3_large_creation(self):
        """Test MobileNetV3-Large model creation."""
        model = mobilenetv3_large(width_mult=1.0, pretrained=False, embeding_dim=1280)
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'forward')
        assert hasattr(model, 'make_logits')
    
    def test_mobilenetv4_small_creation(self):
        """Test MobileNetV4-Small model creation."""
        model = mobilenetv4_small(width_mult=1.0, pretrained=False, embeding_dim=1024)
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'forward')
        assert hasattr(model, 'make_logits')
    
    def test_mobilenetv4_medium_creation(self):
        """Test MobileNetV4-Medium model creation."""
        model = mobilenetv4_medium(width_mult=1.0, pretrained=False, embeding_dim=1152)
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'forward')
        assert hasattr(model, 'make_logits')
    
    def test_mobilenetv4_large_creation(self):
        """Test MobileNetV4-Large model creation."""
        model = mobilenetv4_large(width_mult=1.0, pretrained=False, embeding_dim=1280)
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'forward')
        assert hasattr(model, 'make_logits')


class TestModelForward:
    """Test model forward pass functionality."""
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(2, 3, 224, 224)
    
    def test_mobilenetv2_forward(self, sample_input):
        """Test MobileNetV2 forward pass."""
        model = mobilenetv2(width_mult=1.0, pretrained=False, embeding_dim=1280)
        model.eval()
        
        with torch.no_grad():
            output = model(sample_input)
            validate_tensor(output, expected_shape=(2, 1280))
    
    def test_mobilenetv3_small_forward(self, sample_input):
        """Test MobileNetV3-Small forward pass."""
        model = mobilenetv3_small(width_mult=1.0, pretrained=False, embeding_dim=1024)
        model.eval()
        
        with torch.no_grad():
            output = model(sample_input)
            validate_tensor(output, expected_shape=(2, 1024))
    
    def test_mobilenetv3_large_forward(self, sample_input):
        """Test MobileNetV3-Large forward pass."""
        model = mobilenetv3_large(width_mult=1.0, pretrained=False, embeding_dim=1280)
        model.eval()
        
        with torch.no_grad():
            output = model(sample_input)
            validate_tensor(output, expected_shape=(2, 1280))
    
    def test_mobilenetv4_small_forward(self, sample_input):
        """Test MobileNetV4-Small forward pass."""
        model = mobilenetv4_small(width_mult=1.0, pretrained=False, embeding_dim=1024)
        model.eval()
        
        with torch.no_grad():
            output = model(sample_input)
            validate_tensor(output, expected_shape=(2, 1024))
    
    def test_mobilenetv4_medium_forward(self, sample_input):
        """Test MobileNetV4-Medium forward pass."""
        model = mobilenetv4_medium(width_mult=1.0, pretrained=False, embeding_dim=1152)
        model.eval()
        
        with torch.no_grad():
            output = model(sample_input)
            validate_tensor(output, expected_shape=(2, 1152))
    
    def test_mobilenetv4_large_forward(self, sample_input):
        """Test MobileNetV4-Large forward pass."""
        model = mobilenetv4_large(width_mult=1.0, pretrained=False, embeding_dim=1280)
        model.eval()
        
        with torch.no_grad():
            output = model(sample_input)
            validate_tensor(output, expected_shape=(2, 1280))


class TestModelLogits:
    """Test model logits functionality."""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features tensor."""
        return torch.randn(2, 1280)
    
    def test_mobilenetv2_logits(self, sample_features):
        """Test MobileNetV2 logits generation."""
        model = mobilenetv2(width_mult=1.0, pretrained=False, embeding_dim=1280)
        model.eval()
        
        with torch.no_grad():
            logits = model.make_logits(sample_features, all=False)
            validate_tensor(logits, expected_shape=(2, 2))
    
    def test_mobilenetv4_multi_head_logits(self, sample_features):
        """Test MobileNetV4 multi-head logits generation."""
        model = mobilenetv4_large(width_mult=1.0, pretrained=False, embeding_dim=1280, multi_heads=True)
        model.eval()
        
        with torch.no_grad():
            logits = model.make_logits(sample_features, all=True)
            assert isinstance(logits, list)
            assert len(logits) == 4  # spoof, spoof_type, lighting, attributes


class TestModelParameters:
    """Test model parameter functionality."""
    
    def test_mobilenetv2_parameters(self):
        """Test MobileNetV2 parameter count."""
        model = mobilenetv2(width_mult=1.0, pretrained=False, embeding_dim=1280)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params == total_params  # All parameters should be trainable
    
    def test_mobilenetv4_parameters(self):
        """Test MobileNetV4 parameter count."""
        model = mobilenetv4_large(width_mult=1.0, pretrained=False, embeding_dim=1280)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params == total_params


class TestModelErrors:
    """Test model error handling."""
    
    def test_invalid_input_shape(self):
        """Test model with invalid input shape."""
        model = mobilenetv2(width_mult=1.0, pretrained=False, embeding_dim=1280)
        model.eval()
        
        # Invalid input shape (wrong number of dimensions)
        invalid_input = torch.randn(2, 3, 224)  # Missing height dimension
        
        with pytest.raises((RuntimeError, ValueError)):
            with torch.no_grad():
                model(invalid_input)
    
    def test_invalid_batch_size(self):
        """Test model with invalid batch size."""
        model = mobilenetv2(width_mult=1.0, pretrained=False, embeding_dim=1280)
        model.eval()
        
        # Zero batch size
        invalid_input = torch.randn(0, 3, 224, 224)
        
        with pytest.raises((RuntimeError, ValueError)):
            with torch.no_grad():
                model(invalid_input)


if __name__ == "__main__":
    pytest.main([__file__])
