# FAS-Research-Framework API Reference

## 📚 Overview

This document provides comprehensive API reference for the FAS-Research-Framework, including all modules, classes, functions, and their usage examples.

## 🏗️ Core Modules

### 1. Model Architectures

#### MobileNetV4
```python
from models.mobilenetv4 import mobilenetv4_small, mobilenetv4_medium, mobilenetv4_large

# Create MobileNetV4 models
model_small = mobilenetv4_small(num_classes=2, pretrained=True)
model_medium = mobilenetv4_medium(num_classes=2, pretrained=True)
model_large = mobilenetv4_large(num_classes=2, pretrained=True)
```

**Parameters:**
- `num_classes` (int): Number of output classes (default: 2 for binary classification)
- `pretrained` (bool): Whether to use pretrained weights (default: True)
- `width_mult` (float): Width multiplier for model scaling (default: 1.0)

**Returns:**
- `torch.nn.Module`: MobileNetV4 model instance

#### EfficientNet
```python
from models.efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7

# Create EfficientNet models
model_b0 = efficientnet_b0(num_classes=2, pretrained=True)
model_b7 = efficientnet_b7(num_classes=2, pretrained=True)
```

#### Vision Transformer
```python
from models.vision_transformer import vit_tiny, vit_small, vit_base, vit_large, vit_huge

# Create ViT models
model_tiny = vit_tiny(num_classes=2, pretrained=True)
model_base = vit_base(num_classes=2, pretrained=True)
```

#### ResNet
```python
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

# Create ResNet models
model_18 = resnet18(num_classes=2, pretrained=True)
model_50 = resnet50(num_classes=2, pretrained=True)
```

### 2. Dataset Classes

#### CelebA-Spoof Dataset
```python
from datasets.celeba_spoof import CelebASpoofDataset

# Create dataset
dataset = CelebASpoofDataset(
    root_dir="/path/to/celeba_spoof",
    split="train",
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)

# Create data loader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

#### SiW Dataset
```python
from datasets.siw_dataset import SiWDataset

# Create SiW dataset
dataset = SiWDataset(
    root_dir="/path/to/siw",
    split="train",
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)
```

#### OULU-NPU Dataset
```python
from datasets.oulu_npu_dataset import OULUNPUDataset

# Create OULU-NPU dataset
dataset = OULUNPUDataset(
    root_dir="/path/to/oulu_npu",
    split="train",
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)
```

#### Custom Dataset Interface
```python
from datasets.custom_dataset_interface import ImageFolderDataset, JSONDataset, CSVDataset

# Image folder dataset
dataset = ImageFolderDataset(
    root_dir="/path/to/images",
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    label_mapping={'real': 0, 'spoof': 1}
)

# JSON dataset
dataset = JSONDataset(
    root_dir="/path/to/images",
    annotation_file="/path/to/annotations.json",
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)

# CSV dataset
dataset = CSVDataset(
    root_dir="/path/to/images",
    csv_file="/path/to/annotations.csv",
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)
```

### 3. Advanced Training Techniques

#### Adversarial Training
```python
from advanced_training.adversarial_training import AdversarialTrainer, AdversarialTrainingConfig

# Create adversarial training configuration
config = AdversarialTrainingConfig(
    attack_type="pgd",
    epsilon=0.03,
    alpha=0.01,
    num_steps=10,
    lambda_adv=0.5
)

# Create adversarial trainer
trainer = AdversarialTrainer(model, device="cuda", attack_type="pgd", epsilon=0.03)

# Training step
loss_dict = trainer.train_step(x, y, optimizer)
```

#### Knowledge Distillation
```python
from advanced_training.knowledge_distillation import KnowledgeDistillationTrainer, DistillationConfig

# Create distillation configuration
config = DistillationConfig(
    temperature=3.0,
    alpha=0.7,
    beta=0.3
)

# Create distillation trainer
trainer = KnowledgeDistillationTrainer(teacher_model, student_model, device="cuda")

# Distillation step
loss_dict = trainer.distillation_step(x, y, optimizer)
```

#### Self-Supervised Learning
```python
from advanced_training.self_supervised_learning import SelfSupervisedTrainer, SelfSupervisedConfig

# Create self-supervised configuration
config = SelfSupervisedConfig(
    method="simclr",
    temperature=0.07
)

# Create self-supervised trainer
trainer = SelfSupervisedTrainer(model, device="cuda", method="simclr")

# Training step
loss_dict = trainer.training_step(x, optimizer)
```

#### Curriculum Learning
```python
from advanced_training.curriculum_learning import CurriculumTrainer, CurriculumConfig

# Create curriculum configuration
config = CurriculumConfig(
    difficulty_method="loss",
    curriculum_method="linear",
    start_ratio=0.1,
    end_ratio=1.0
)

# Create curriculum trainer
trainer = CurriculumTrainer(model, difficulty_estimator, curriculum_scheduler, device="cuda")

# Training step
loss_dict = trainer.training_step(x, y, optimizer, epoch, total_epochs)
```

#### Meta-Learning
```python
from advanced_training.meta_learning import MetaLearningTrainer, MetaLearningConfig

# Create meta-learning configuration
config = MetaLearningConfig(
    method="maml",
    inner_lr=0.01,
    meta_lr=0.001,
    num_inner_steps=5
)

# Create meta-learning trainer
trainer = MetaLearningTrainer(model, method="maml", device="cuda")

# Training step
loss_dict = trainer.training_step(x, y, optimizer)
```

### 4. Visualization and Analysis

#### Attention Visualization
```python
from visualization.attention_visualization import AttentionVisualizer, AttentionAnalyzer

# Create attention visualizer
visualizer = AttentionVisualizer(model, device="cuda")
visualizer.register_hooks(['attention', 'self_attention'])

# Visualize attention
attention_maps = visualizer.visualize_attention(image, layer_name="attention")

# Create attention analyzer
analyzer = AttentionAnalyzer(model, device="cuda")
attention_stats = analyzer.analyze_attention_patterns(dataloader, num_samples=100)
```

#### Feature Analysis
```python
from visualization.feature_analysis import FeatureAnalyzer, FeatureVisualizer

# Create feature analyzer
analyzer = FeatureAnalyzer(model, device="cuda")
analyzer.register_feature_hooks(['features', 'classifier'])

# Extract features
feature_data = analyzer.extract_features(dataloader, max_samples=1000)

# Create feature visualizer
visualizer = FeatureVisualizer(model, device="cuda")

# Create t-SNE visualization
tsne_fig = visualizer.create_feature_tsne(features, labels)

# Create PCA visualization
pca_fig = visualizer.create_feature_pca(features, labels)
```

#### Model Interpretability
```python
from visualization.model_interpretability import GradCAM, IntegratedGradients, LIME, ModelInterpretabilityAnalyzer

# GradCAM
gradcam = GradCAM(model, target_layer="features", device="cuda")
cam = gradcam.generate_cam(image, class_idx=0)

# Integrated Gradients
ig = IntegratedGradients(model, device="cuda", steps=50)
attributions = ig.generate_attributions(image, class_idx=0)

# LIME
lime = LIME(model, device="cuda", num_samples=1000)
explanations = lime.generate_explanations(image, class_idx=0)

# Comprehensive interpretability analysis
analyzer = ModelInterpretabilityAnalyzer(model, device="cuda")
results = analyzer.analyze_model_interpretability(image, target_layer="features")
```

#### Failure Case Analysis
```python
from visualization.failure_case_analysis import FailureCaseAnalyzer, FailureCaseReporter

# Create failure case analyzer
analyzer = FailureCaseAnalyzer(model, device="cuda")

# Identify failure cases
failure_cases = analyzer.identify_failure_cases(dataloader, confidence_threshold=0.5)

# Analyze failure patterns
failure_patterns = analyzer.analyze_failure_patterns(failure_cases)

# Analyze confidence distribution
confidence_analysis = analyzer.analyze_confidence_distribution(dataloader)

# Create failure case reporter
reporter = FailureCaseReporter(output_dir="/path/to/output")
reporter.save_failure_report(failure_cases, failure_patterns, confidence_analysis)
```

#### Interactive Dashboards
```python
from visualization.interactive_dashboards import InteractiveDashboard, StreamlitDashboard

# Create interactive dashboard
dashboard = InteractiveDashboard(model, device="cuda")

# Create performance dashboard
perf_fig = dashboard.create_performance_dashboard(evaluation_results)

# Create feature dashboard
feature_fig = dashboard.create_feature_dashboard(feature_data)

# Create attention dashboard
attention_fig = dashboard.create_attention_dashboard(attention_data)

# Create failure analysis dashboard
failure_fig = dashboard.create_failure_analysis_dashboard(failure_data)

# Streamlit dashboard
streamlit_dashboard = StreamlitDashboard(model, device="cuda")
streamlit_dashboard.create_main_dashboard()
```

### 5. Dataset Conversion and Management

#### Dataset Conversion
```python
from datasets.dataset_conversion_tools import convert_dataset, split_dataset, merge_datasets

# Convert image folder to JSON
convert_dataset(
    source_dir="/path/to/image_folder",
    target_dir="/path/to/json_output",
    conversion_type="image_folder_to_json"
)

# Convert JSON to CSV
convert_dataset(
    source_dir="/path/to/json_input",
    target_dir="/path/to/csv_output",
    conversion_type="json_to_csv"
)

# Split dataset into train/val/test
split_dataset(
    source_dir="/path/to/dataset",
    target_dir="/path/to/splits",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    dataset_type="image_folder"
)

# Merge multiple datasets
merge_datasets(
    source_dirs=["/path/to/dataset1", "/path/to/dataset2"],
    target_dir="/path/to/merged",
    dataset_type="image_folder"
)
```

#### Cross-Dataset Evaluation
```python
from datasets.cross_dataset_evaluation import evaluate_cross_dataset, evaluate_domain_adaptation, evaluate_robustness

# Cross-dataset evaluation
results = evaluate_cross_dataset(
    model=model,
    dataloaders={
        'celeba_spoof': celeba_dataloader,
        'lcc_fasd': lcc_dataloader,
        'siw': siw_dataloader
    },
    device="cuda"
)

# Domain adaptation evaluation
adaptation_results = evaluate_domain_adaptation(
    model=model,
    source_dataloader=source_dataloader,
    target_dataloader=target_dataloader,
    device="cuda"
)

# Robustness evaluation
robustness_results = evaluate_robustness(
    model=model,
    dataloaders={
        'normal': normal_dataloader,
        'bright': bright_dataloader,
        'dark': dark_dataloader
    },
    device="cuda"
)
```

### 6. Model Comparison and Architecture Search

#### Model Comparison Framework
```python
from model_comparison_framework import ModelComparisonFramework

# Create model comparison framework
comparison = ModelComparisonFramework(
    models={
        'mobilenetv4_large': mobilenetv4_large_model,
        'efficientnet_b7': efficientnet_b7_model,
        'vit_base': vit_base_model,
        'resnet50': resnet50_model
    },
    dataloaders={
        'train': train_dataloader,
        'val': val_dataloader,
        'test': test_dataloader
    }
)

# Run comprehensive comparison
results = comparison.compare_models(
    metrics=['accuracy', 'precision', 'recall', 'f1_score', 'auc'],
    include_training_time=True,
    include_inference_time=True,
    include_memory_usage=True
)

# Generate comparison report
comparison.generate_report(output_path="/path/to/comparison_report.html")
```

#### Architecture Search Tools
```python
from architecture_search_tools import ArchitectureSearch, RandomSearch, GridSearch, EvolutionarySearch

# Random search
random_search = RandomSearch(
    search_space={
        'model_type': ['mobilenetv4', 'efficientnet', 'vit', 'resnet'],
        'model_size': ['small', 'medium', 'large'],
        'learning_rate': [0.001, 0.01, 0.1],
        'batch_size': [16, 32, 64]
    },
    max_trials=100
)

# Grid search
grid_search = GridSearch(
    search_space={
        'model_type': ['mobilenetv4', 'efficientnet'],
        'model_size': ['small', 'large'],
        'learning_rate': [0.001, 0.01]
    }
)

# Evolutionary search
evolutionary_search = EvolutionarySearch(
    population_size=50,
    generations=20,
    mutation_rate=0.1,
    crossover_rate=0.8
)

# Run architecture search
best_config = random_search.search(
    objective_function=objective_function,
    dataloader=val_dataloader
)
```

### 7. Utility Functions

#### Model Building
```python
from utils import build_model, load_checkpoint, save_checkpoint

# Build model from configuration
model = build_model(config, device="cuda", strict=True, mode="train")

# Load checkpoint
checkpoint = load_checkpoint(checkpoint_path, model, strict=True, map_location="cuda")

# Save checkpoint
save_checkpoint(model, optimizer, epoch, loss, checkpoint_path)
```

#### Configuration Management
```python
from utils import read_py_config, validate_config
from configs.paths import get_dataset_path, get_model_path, get_log_path

# Read configuration
config = read_py_config("configs/config_mobilenetv4_large.py")

# Validate configuration
validate_config(config)

# Get paths
dataset_path = get_dataset_path("celeba_spoof")
model_path = get_model_path("mobilenetv4_large.pth")
log_path = get_log_path("training.log")
```

#### Error Handling and Logging
```python
from utils.error_handling import validate_device, validate_tensor, log_system_info
from utils.logging_config import setup_logging, get_logger, log_training_start

# Setup logging
setup_logging(log_level="INFO", log_file="training.log")

# Get logger
logger = get_logger(__name__)

# Validate device
device = validate_device("cuda")

# Validate tensor
tensor = validate_tensor(tensor, expected_shape=(batch_size, channels, height, width))

# Log system info
log_system_info(logger)

# Log training start
log_training_start(logger, config, device)
```

### 8. Training and Evaluation Scripts

#### Training Script
```python
# train.py
import argparse
from utils import read_py_config, build_model, setup_logging, get_logger
from trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description='Face Anti-Spoofing Training')
    parser.add_argument('--config', type=str, required=True, help='Configuration file path')
    parser.add_argument('--GPU', type=int, default=0, help='GPU device ID')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration
    config = read_py_config(args.config)
    
    # Setup logging
    setup_logging(log_level="INFO")
    logger = get_logger(__name__)
    
    # Build model
    model = build_model(config, device=f"cuda:{args.GPU}")
    
    # Create trainer
    trainer = Trainer(model, config, device=f"cuda:{args.GPU}")
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
```

#### Evaluation Script
```python
# eval_protocol.py
import argparse
from utils import read_py_config, build_model, load_checkpoint
from eval_protocol import evaluate_model

def main():
    parser = argparse.ArgumentParser(description='Face Anti-Spoofing Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Configuration file path')
    parser.add_argument('--GPU', type=int, default=0, help='GPU device ID')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--draw_graph', action='store_true', help='Generate visualization graphs')
    
    args = parser.parse_args()
    
    # Load configuration
    config = read_py_config(args.config)
    
    # Build model
    model = build_model(config, device=f"cuda:{args.GPU}")
    
    # Load checkpoint
    load_checkpoint(args.checkpoint, model, strict=True, map_location=f"cuda:{args.GPU}")
    
    # Evaluate model
    results = evaluate_model(model, config, device=f"cuda:{args.GPU}", draw_graph=args.draw_graph)
    
    print(f"Evaluation Results: {results}")

if __name__ == "__main__":
    main()
```

## 🔧 Configuration Examples

### MobileNetV4 Configuration
```python
# configs/config_mobilenetv4_large.py
from configs.paths import get_dataset_path, get_model_path, get_log_path

config = {
    'model': {
        'model_type': 'Mobilenet4',
        'model_size': 'large',
        'width_mult': 1.0,
        'pretrained': True,
        'embeding_dim': 1280,
        'imagenet_weights': get_model_path('mobilenetv4_large_imagenet.pth.tar')
    },
    'data': {
        'dataset': 'celeba_spoof',
        'data_root': get_dataset_path('celeba_spoof'),
        'batch_size': 32,
        'data_loader_workers': 8,
        'mixup': True,
        'cutmix': True
    },
    'training': {
        'epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'optimizer': 'adam',
        'scheduler': 'cosine'
    },
    'loss': {
        'loss_type': 'amsoftmax',
        'amsoftmax': {
            'm': 0.5,
            's': 1,
            'margin_type': 'cross_entropy'
        }
    },
    'logging': {
        'log_dir': get_log_path('mobilenetv4_large'),
        'save_freq': 10,
        'eval_freq': 5
    }
}
```

### EfficientNet Configuration
```python
# configs/config_efficientnet_b0.py
config = {
    'model': {
        'model_type': 'EfficientNet',
        'model_size': 'b0',
        'pretrained': True,
        'embeding_dim': 1280,
        'imagenet_weights': get_model_path('efficientnet_b0_imagenet.pth.tar')
    },
    'data': {
        'dataset': 'celeba_spoof',
        'data_root': get_dataset_path('celeba_spoof'),
        'batch_size': 32,
        'data_loader_workers': 8
    },
    'training': {
        'epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'optimizer': 'adam',
        'scheduler': 'cosine'
    },
    'loss': {
        'loss_type': 'amsoftmax',
        'amsoftmax': {
            'm': 0.5,
            's': 1,
            'margin_type': 'cross_entropy'
        }
    }
}
```

## 📊 Usage Examples

### Complete Training Pipeline
```python
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# 1. Load configuration
from utils import read_py_config
config = read_py_config("configs/config_mobilenetv4_large.py")

# 2. Create dataset
from datasets.celeba_spoof import CelebASpoofDataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CelebASpoofDataset(
    root_dir=config['data']['data_root'],
    split="train",
    transform=transform
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=config['data']['batch_size'],
    shuffle=True,
    num_workers=config['data']['data_loader_workers']
)

# 3. Build model
from utils import build_model
model = build_model(config, device="cuda")

# 4. Create trainer
from trainer import Trainer
trainer = Trainer(model, config, device="cuda")

# 5. Start training
trainer.train()
```

### Complete Evaluation Pipeline
```python
# 1. Load model
from utils import build_model, load_checkpoint
model = build_model(config, device="cuda")
load_checkpoint("checkpoints/best_model.pth", model, strict=True)

# 2. Create test dataset
test_dataset = CelebASpoofDataset(
    root_dir=config['data']['data_root'],
    split="test",
    transform=transform
)

test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. Evaluate model
from eval_protocol import evaluate_model
results = evaluate_model(model, test_dataloader, device="cuda")

print(f"Accuracy: {results['accuracy']:.4f}")
print(f"AUC: {results['auc']:.4f}")
print(f"EER: {results['eer']:.4f}")
```

### Advanced Training with Adversarial Training
```python
# 1. Create adversarial training configuration
from advanced_training.adversarial_training import AdversarialTrainingConfig
adv_config = AdversarialTrainingConfig(
    attack_type="pgd",
    epsilon=0.03,
    alpha=0.01,
    num_steps=10,
    lambda_adv=0.5
)

# 2. Create adversarial trainer
from advanced_training.adversarial_training import AdversarialTrainer
adv_trainer = AdversarialTrainer(model, device="cuda", attack_type="pgd", epsilon=0.03)

# 3. Training loop with adversarial training
for epoch in range(num_epochs):
    for batch in train_dataloader:
        x, y = batch['image'], batch['label']
        
        # Adversarial training step
        loss_dict = adv_trainer.train_step(x, y, optimizer)
        
        print(f"Epoch {epoch}, Loss: {loss_dict['total_loss']:.4f}")
```

This API reference provides comprehensive documentation for all major components of the face anti-spoofing system, enabling users to effectively utilize the project's capabilities.
