# Setup and Installation Guide

## Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 18.04+), macOS (10.14+), or Windows 10+
- **Python**: 3.7 or higher
- **CUDA**: 10.2 or higher (for GPU acceleration)
- **Memory**: 16GB RAM minimum, 32GB recommended
- **Storage**: 50GB free space for datasets and models
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)

### Hardware Recommendations
- **Training**: NVIDIA RTX 3080/4080 or better
- **Inference**: NVIDIA GTX 1660 or better
- **CPU**: 8+ cores recommended
- **RAM**: 32GB+ for large datasets

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd FAS-Research-Framework
```

### 2. Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n face-antispoof python=3.8
conda activate face-antispoof

# Or using venv
python -m venv face-antispoof
source face-antispoof/bin/activate  # Linux/macOS
# face-antispoof\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
# Install PyTorch (choose appropriate version for your CUDA)
# For CUDA 11.6
pip install torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# For CPU only
pip install torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

### 4. Download Pretrained Models (Optional but Recommended)
```bash
# List available pretrained models
python download_pretrained_models.py --list

# Download specific models
python download_pretrained_models.py --models mobilenetv4_large_imagenet.pth.tar

# Download all available models
python download_pretrained_models.py --all
```

**Note**: Pretrained models are not included in the repository to keep it lightweight. You can:
- Download them using the script above
- Train from scratch (set `pretrained=False` in config files)
- Use your own pretrained weights

### 5. Verify Installation
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Dataset Setup

### CelebA-Spoof Dataset

#### 1. Download Dataset
```bash
# Create dataset directory
mkdir -p datasets/celeba_spoof
cd datasets/celeba_spoof

# Download from official repository
# Note: You need to request access from the official repository
# https://github.com/Davidzhangyuanhan/CelebA-Spoof
```

#### 2. Prepare Dataset Structure
```
datasets/celeba_spoof/
├── Data/
│   ├── train/
│   ├── val/
│   └── test/
└── metas/
    └── intra_test/
        ├── items_train.json
        ├── items_val.json
        └── items_test.json
```

#### 3. Generate JSON Files
```bash
# Run the preparation script
python prepare_celeba_json.py
```

### LCC FASD Dataset

#### 1. Download Dataset
```bash
# Create dataset directory
mkdir -p datasets/lcc_fasd
cd datasets/lcc_fasd

# Download from official source
# http://www.idiap.ch/dataset/lccfasd
```

#### 2. Prepare Dataset Structure
```
datasets/lcc_fasd/
├── LCC_FASD_training/
│   ├── real/
│   └── spoof/
├── LCC_FASD_development/
│   ├── real/
│   └── spoof/
└── LCC_FASD_evaluation/
    ├── real_*.jpg
    └── spoof_*.jpg
```

## Configuration

### 1. Update Dataset Paths
Edit the configuration files in `configs/` directory:

```python
# configs/config.py
datasets = dict(
    Celeba_root='./datasets/celeba_spoof',
    LCCFASD_root='./datasets/lcc_fasd',
    Casia_root='./datasets/casia'  # Optional
)
```

### 2. Configure Model Settings
```python
# Example configuration for MobileNetV3-Large
model = dict(
    model_type='Mobilenet3',
    model_size='large',
    width_mult=1.0,
    pretrained=True,
    embeding_dim=1280,
    imagenet_weights='./pretrained/MobileNet3-large-224-gaussian.pth.tar'
)
```

### 3. Set Training Parameters
```python
# Training configuration
data = dict(
    batch_size=50,
    data_loader_workers=8,
    sampler=True,
    pin_memory=True
)

optimizer = dict(
    lr=0.005,
    momentum=0.9,
    weight_decay=5e-4
)

epochs = dict(
    start_epoch=0,
    max_epoch=71
)
```

## Quick Start

### 1. Test Installation
```bash
# Test with a small configuration
python train.py --config configs/config_small.py --device cpu --test_steps 10
```

### 2. Train a Model
```bash
# Train on CelebA-Spoof dataset
python train.py --config configs/config_large.py --GPU 0

# Train on LCC FASD dataset
python train.py --config configs/config_small.py --GPU 0
```

### 3. Evaluate a Model
```bash
# Evaluate trained model
python eval_protocol.py --config configs/config_large.py --GPU 0
```

## Advanced Setup

### Multi-GPU Training
```python
# In configuration file
data_parallel = dict(
    use_parallel=True,
    parallel_params=dict(
        device_ids=[0, 1, 2, 3],
        output_device=0
    )
)
```

### Custom Datasets
1. Create a new dataset class in `datasets/`
2. Implement the required methods
3. Add dataset registration in `datasets/__init__.py`
4. Update configuration files

### Custom Models
1. Create a new model class in `models/`
2. Implement the required methods
3. Add model registration in `models/__init__.py`
4. Update configuration files

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
data = dict(batch_size=32)  # Instead of 50

# Use gradient accumulation
# Add to training script
```

#### 2. Dataset Not Found
```bash
# Check dataset paths in configuration
# Verify dataset structure
# Run dataset validation script
```

#### 3. Import Errors
```bash
# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Verify all dependencies are installed
pip list | grep torch
```

#### 4. Permission Errors
```bash
# Fix file permissions
chmod -R 755 datasets/
chmod -R 755 pretrained/
```

### Performance Optimization

#### 1. Data Loading
```python
# Increase number of workers
data = dict(
    data_loader_workers=16,  # Adjust based on CPU cores
    pin_memory=True
)
```

#### 2. Memory Optimization
```python
# Use smaller batch size
data = dict(batch_size=32)

# Enable gradient checkpointing
# Add to model configuration
```

#### 3. Training Speed
```python
# Use mixed precision training
# Add to training script
from torch.cuda.amp import autocast, GradScaler
```

## Development Setup

### 1. Install Development Dependencies
```bash
pip install -r requirements-dev.txt
```

### 2. Set Up Pre-commit Hooks
```bash
pre-commit install
```

### 3. Run Tests
```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Run with coverage
pytest --cov=. tests/
```

### 4. Code Formatting
```bash
# Format code
black .
isort .

# Lint code
pylint src/
```

## Docker Setup

### 1. Build Docker Image
```bash
docker build -t face-antispoof .
```

### 2. Run Container
```bash
# Training
docker run --gpus all -v $(pwd)/datasets:/app/datasets face-antispoof python train.py --config configs/config_large.py

# Evaluation
docker run --gpus all -v $(pwd)/datasets:/app/datasets face-antispoof python eval_protocol.py --config configs/config_large.py
```

## Cloud Setup

### AWS EC2
```bash
# Launch GPU instance (p3.2xlarge or better)
# Install dependencies
# Upload datasets to S3
# Configure for distributed training
```

### Google Cloud Platform
```bash
# Launch GPU instance
# Install dependencies
# Use Cloud Storage for datasets
# Configure for distributed training
```

### Azure
```bash
# Launch GPU instance
# Install dependencies
# Use Blob Storage for datasets
# Configure for distributed training
```

## Monitoring and Logging

### 1. TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir=./logs

# View training progress
# Open http://localhost:6006
```

### 2. Logging Configuration
```python
# Configure logging in your script
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
```

## Production Deployment

### 1. Model Export
```bash
# Export to ONNX
python export_model.py --config configs/config_large.py --output model.onnx
```

### 2. API Service
```bash
# Start API service
python api_server.py --config configs/config_large.py --port 8000
```

### 3. Docker Production
```bash
# Build production image
docker build -f Dockerfile.prod -t face-antispoof-prod .

# Run production container
docker run -p 8000:8000 face-antispoof-prod
```

## Support and Resources

### Documentation
- [Project Overview](PROJECT_OVERVIEW.md)
- [Technical Architecture](TECHNICAL_ARCHITECTURE.md)
- [API Reference](API_REFERENCE.md)
- [Development Plan](DEVELOPMENT_PLAN.md)

### Community
- GitHub Issues for bug reports
- Discussions for questions
- Pull requests for contributions

### Getting Help
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed information
4. Provide system information and error logs
