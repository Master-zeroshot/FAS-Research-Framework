"""
Unit tests for utility functions.
"""

import pytest
import torch
import torch.nn as nn

from utils.error_handling import (ModelError, ValidationError, validate_device,
                                  validate_file_path, validate_tensor)
from utils.logging_config import setup_logging, get_logger


class TestErrorHandling:
    """Test error handling utilities."""
    
    def test_validate_tensor_valid(self):
        """Test tensor validation with valid tensor."""
        tensor = torch.randn(2, 3, 224, 224)
        validate_tensor(tensor, expected_shape=(2, 3, 224, 224), name="test_tensor")
    
    def test_validate_tensor_invalid_shape(self):
        """Test tensor validation with invalid shape."""
        tensor = torch.randn(2, 3, 224, 224)
        
        with pytest.raises(ValidationError):
            validate_tensor(tensor, expected_shape=(2, 3, 224, 225), name="test_tensor")
    
    def test_validate_tensor_nan(self):
        """Test tensor validation with NaN values."""
        tensor = torch.randn(2, 3, 224, 224)
        tensor[0, 0, 0, 0] = float('nan')
        
        with pytest.raises(ValidationError):
            validate_tensor(tensor, name="test_tensor")
    
    def test_validate_tensor_inf(self):
        """Test tensor validation with infinite values."""
        tensor = torch.randn(2, 3, 224, 224)
        tensor[0, 0, 0, 0] = float('inf')
        
        with pytest.raises(ValidationError):
            validate_tensor(tensor, name="test_tensor")
    
    def test_validate_device_cpu(self):
        """Test device validation with CPU."""
        device = validate_device("cpu")
        assert device == "cpu"
    
    def test_validate_device_cuda_available(self):
        """Test device validation with CUDA (if available)."""
        if torch.cuda.is_available():
            device = validate_device("cuda:0")
            assert device == "cuda:0"
        else:
            with pytest.raises(ValidationError):
                validate_device("cuda:0")
    
    def test_validate_device_invalid(self):
        """Test device validation with invalid device."""
        with pytest.raises(ValidationError):
            validate_device("invalid_device")
    
    def test_validate_file_path_existing(self, tmp_path):
        """Test file path validation with existing file."""
        test_file = tmp_path / "test_file.txt"
        test_file.write_text("test content")
        
        path = validate_file_path(test_file, must_exist=True)
        assert path == test_file.resolve()
    
    def test_validate_file_path_nonexistent(self, tmp_path):
        """Test file path validation with non-existing file."""
        test_file = tmp_path / "nonexistent.txt"
        
        with pytest.raises(ValidationError):
            validate_file_path(test_file, must_exist=True)
    
    def test_validate_file_path_not_exist_ok(self, tmp_path):
        """Test file path validation without existence check."""
        test_file = tmp_path / "nonexistent.txt"
        
        path = validate_file_path(test_file, must_exist=False)
        assert path == test_file.resolve()


class TestLogging:
    """Test logging functionality."""
    
    def test_setup_logging(self):
        """Test logging setup."""
        logger = setup_logging(log_level="INFO", log_to_file=False, log_to_console=True)
        assert isinstance(logger, logging.Logger)
    
    def test_get_logger(self):
        """Test getting a logger."""
        logger = get_logger("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"


class TestModelForward:
    """Test model forward pass with error handling."""
    
    def test_safe_model_forward_valid(self):
        """Test safe model forward with valid input."""
        from utils.error_handling import safe_model_forward
        
        model = nn.Linear(10, 2)
        input_tensor = torch.randn(2, 10)
        
        output = safe_model_forward(model, input_tensor, device="cpu")
        assert output.shape == (2, 2)
    
    def test_safe_model_forward_invalid_input(self):
        """Test safe model forward with invalid input."""
        from utils.error_handling import safe_model_forward
        
        model = nn.Linear(10, 2)
        input_tensor = torch.randn(2, 5)  # Wrong input size
        
        with pytest.raises(ModelError):
            safe_model_forward(model, input_tensor, device="cpu")


class TestDataLoading:
    """Test data loading with error handling."""
    
    def test_safe_data_loading(self):
        """Test safe data loading."""
        from utils.error_handling import safe_data_loading
        
        # Create a simple dataset
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, size=100):
                self.data = torch.randn(size, 3, 224, 224)
                self.labels = torch.randint(0, 2, (size,))
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]
        
        dataset = SimpleDataset(100)
        dataloader = safe_data_loading(dataset, batch_size=32, num_workers=0)
        
        assert isinstance(dataloader, torch.utils.data.DataLoader)
        assert dataloader.batch_size == 32
    
    def test_safe_data_loading_invalid_workers(self):
        """Test safe data loading with invalid number of workers."""
        from utils.error_handling import safe_data_loading
        
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, size=100):
                self.data = torch.randn(size, 3, 224, 224)
                self.labels = torch.randint(0, 2, (size,))
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]
        
        dataset = SimpleDataset(100)
        
        # Test with negative number of workers
        with pytest.raises(Exception):  # Should raise some exception
            safe_data_loading(dataset, batch_size=32, num_workers=-1)


if __name__ == "__main__":
    pytest.main([__file__])
