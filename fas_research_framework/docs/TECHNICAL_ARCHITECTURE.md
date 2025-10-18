# FAS-Research-Framework Technical Architecture

## 🏗️ System Architecture Overview

This document provides a comprehensive technical overview of the FAS-Research-Framework architecture, including model designs, training pipelines, evaluation frameworks, and deployment considerations.

## 📐 Core Architecture Components

### 1. Model Architecture Stack

```
Input Image (224x224x3)
    ↓
Feature Extraction Backbone
    ↓
Global Average Pooling
    ↓
Multi-Head Classifier
    ├── Spoof Detection (Binary)
    ├── Spoof Type Classification (Multi-class)
    ├── Lighting Condition (Multi-class)
    └── Face Attributes (Multi-label)
```

### 2. Model Architecture Details

#### MobileNetV4 Architecture
```python
class MobileNetV4(nn.Module):
    def __init__(self, model_size='large', num_classes=2):
        super().__init__()
        
        # Configuration based on model size
        configs = {
            'small': {'width': 1024, 'depth': 16, 'blocks': [2, 4, 4, 2]},
            'medium': {'width': 1152, 'depth': 18, 'blocks': [2, 4, 4, 2]},
            'large': {'width': 1280, 'depth': 20, 'blocks': [2, 4, 4, 2]}
        }
        
        self.config = configs[model_size]
        self.backbone = self._build_backbone()
        self.classifier = MultiHeadClassifier(self.config['width'], num_classes)
    
    def _build_backbone(self):
        """Build MobileNetV4 backbone with UIB blocks"""
        layers = []
        in_channels = 3
        
        # Initial convolution
        layers.append(nn.Conv2d(3, 32, 3, 2, 1))
        
        # UIB blocks
        for stage, block_count in enumerate(self.config['blocks']):
            out_channels = 32 * (2 ** (stage + 1))
            for i in range(block_count):
                layers.append(UIBBlock(in_channels, out_channels))
                in_channels = out_channels
        
        return nn.Sequential(*layers)
```

#### Universal Inverted Bottleneck (UIB) Block
```python
class UIBBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio=6, stride=1):
        super().__init__()
        
        self.expand = nn.Conv2d(in_channels, in_channels * expansion_ratio, 1)
        self.depthwise = nn.Conv2d(
            in_channels * expansion_ratio,
            in_channels * expansion_ratio,
            3, stride, 1,
            groups=in_channels * expansion_ratio
        )
        self.project = nn.Conv2d(in_channels * expansion_ratio, out_channels, 1)
        self.se = SEBlock(out_channels)
        self.activation = nn.ReLU6(inplace=True)
        
    def forward(self, x):
        identity = x
        
        x = self.expand(x)
        x = self.activation(x)
        x = self.depthwise(x)
        x = self.activation(x)
        x = self.project(x)
        x = self.se(x)
        
        if x.shape == identity.shape:
            x += identity
        
        return x
```

### 3. Multi-Head Classification System

```python
class MultiHeadClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Task-specific heads
        self.spoof_head = nn.Linear(512, 2)  # Binary classification
        self.spoof_type_head = nn.Linear(512, 5)  # 5 spoof types
        self.lighting_head = nn.Linear(512, 3)  # 3 lighting conditions
        self.attributes_head = nn.Linear(512, 40)  # 40 face attributes
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        return {
            'spoof': self.spoof_head(features),
            'spoof_type': self.spoof_type_head(features),
            'lighting': self.lighting_head(features),
            'attributes': self.attributes_head(features)
        }
```

## 🚀 Advanced Training Architecture

### 1. Adversarial Training Pipeline

```python
class AdversarialTrainingPipeline:
    def __init__(self, model, attack_type='fgsm', epsilon=0.03):
        self.model = model
        self.attack_type = attack_type
        self.epsilon = epsilon
        
    def adversarial_training_step(self, x, y, optimizer):
        # Generate adversarial examples
        x_adv = self.generate_adversarial_examples(x, y)
        
        # Compute losses
        loss_clean = self.compute_loss(x, y)
        loss_adv = self.compute_loss(x_adv, y)
        
        # Combined loss
        total_loss = loss_clean + 0.5 * loss_adv
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()
```

### 2. Knowledge Distillation Architecture

```python
class KnowledgeDistillationPipeline:
    def __init__(self, teacher_model, student_model, temperature=3.0):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        
    def distillation_loss(self, student_logits, teacher_logits, targets):
        # Softmax with temperature
        student_soft = F.softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # Distillation loss (KL divergence)
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            teacher_soft,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Student loss (cross entropy)
        student_loss = F.cross_entropy(student_logits, targets)
        
        return 0.7 * distillation_loss + 0.3 * student_loss
```

### 3. Self-Supervised Learning Architecture

```python
class SelfSupervisedLearningPipeline:
    def __init__(self, model, method='simclr', temperature=0.07):
        self.model = model
        self.method = method
        self.temperature = temperature
        
    def contrastive_loss(self, features1, features2):
        # Normalize features
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features1, features2.T) / self.temperature
        
        # Create labels for positive pairs
        labels = torch.arange(features1.size(0), device=features1.device)
        
        # Compute loss
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss
```

## 📊 Evaluation Architecture

### 1. Comprehensive Evaluation Pipeline

```python
class EvaluationPipeline:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.metrics = {}
        
    def evaluate_model(self, dataloader):
        """Comprehensive model evaluation"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                images, targets, metadata = batch
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                predictions = outputs.argmax(dim=1)
                probabilities = F.softmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Compute metrics
        metrics = self.compute_metrics(all_predictions, all_targets, all_probabilities)
        return metrics
    
    def compute_metrics(self, predictions, targets, probabilities):
        """Compute comprehensive evaluation metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.metrics import roc_auc_score, roc_curve
        
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, average='weighted'),
            'recall': recall_score(targets, predictions, average='weighted'),
            'f1_score': f1_score(targets, predictions, average='weighted'),
            'auc': roc_auc_score(targets, probabilities[:, 1])
        }
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(targets, probabilities[:, 1])
        metrics['roc_curve'] = (fpr, tpr)
        
        # EER (Equal Error Rate)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        metrics['eer'] = eer
        
        return metrics
```

### 2. Cross-Dataset Evaluation Architecture

```python
class CrossDatasetEvaluation:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        
    def evaluate_cross_dataset(self, dataloaders):
        """Evaluate model across multiple datasets"""
        results = {}
        
        for dataset_name, dataloader in dataloaders.items():
            # Evaluate on each dataset
            dataset_results = self.evaluate_single_dataset(dataloader)
            results[dataset_name] = dataset_results
        
        # Compute cross-dataset statistics
        cross_dataset_stats = self.compute_cross_dataset_statistics(results)
        results['cross_dataset_statistics'] = cross_dataset_stats
        
        return results
```

## 🔍 Visualization and Analysis Architecture

### 1. Attention Visualization Architecture

```python
class AttentionVisualization:
    def __init__(self, model, target_layer, device='cuda'):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        
        # Register hooks for attention maps
        self.attention_maps = {}
        self.register_hooks()
    
    def register_hooks(self):
        """Register hooks for capturing attention maps"""
        def hook_fn(module, input, output):
            self.attention_maps[self.target_layer] = output.detach()
        
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(hook_fn)
                break
    
    def generate_attention_map(self, image):
        """Generate attention map for input image"""
        self.model.eval()
        
        with torch.no_grad():
            _ = self.model(image)
            attention_map = self.attention_maps[self.target_layer]
            
            # Process attention map
            attention_map = attention_map.squeeze()
            if attention_map.dim() == 3:  # Multi-head attention
                attention_map = attention_map.mean(dim=0)
            
            # Normalize attention map
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
            
        return attention_map.cpu().numpy()
```

### 2. Feature Analysis Architecture

```python
class FeatureAnalysis:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        
    def extract_features(self, dataloader, max_samples=1000):
        """Extract features from dataset"""
        self.model.eval()
        
        all_features = []
        all_labels = []
        
        sample_count = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if sample_count >= max_samples:
                    break
                
                images, labels, metadata = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Extract features
                features = self.model.backbone(images)
                features = F.adaptive_avg_pool2d(features, 1)
                features = features.view(features.size(0), -1)
                
                all_features.append(features.cpu())
                all_labels.extend(labels.cpu().numpy())
                
                sample_count += images.size(0)
        
        return torch.cat(all_features, dim=0), np.array(all_labels)
    
    def analyze_features(self, features, labels):
        """Analyze extracted features"""
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        
        # t-SNE visualization
        tsne = TSNE(n_components=2, random_state=42)
        features_tsne = tsne.fit_transform(features.numpy())
        
        # PCA analysis
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features.numpy())
        
        # Clustering analysis
        kmeans = KMeans(n_clusters=2, random_state=42)
        cluster_labels = kmeans.fit_predict(features.numpy())
        
        return {
            'tsne': features_tsne,
            'pca': features_pca,
            'clusters': cluster_labels,
            'explained_variance': pca.explained_variance_ratio_
        }
```

## 🚀 Deployment Architecture

### 1. Model Serving Architecture

```python
class ModelServing:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self.load_model(model_path)
        self.preprocessor = self.setup_preprocessor()
        
    def load_model(self, model_path):
        """Load trained model"""
        model = MobileNetV4(model_size='large')
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model
    
    def setup_preprocessor(self):
        """Setup image preprocessing"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image):
        """Make prediction on input image"""
        # Preprocess image
        image_tensor = self.preprocessor(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            prediction = outputs.argmax(dim=1).item()
            confidence = F.softmax(outputs, dim=1).max(dim=1)[0].item()
        
        return {
            'prediction': 'real' if prediction == 0 else 'spoof',
            'confidence': confidence,
            'probability': F.softmax(outputs, dim=1).cpu().numpy()[0]
        }
```

### 2. API Architecture

```python
from flask import Flask, request, jsonify
import base64
from PIL import Image
import io

app = Flask(__name__)
model_serving = ModelServing('path/to/model.pth')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for face anti-spoofing prediction"""
    try:
        # Get image from request
        data = request.get_json()
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        
        # Make prediction
        result = model_serving.predict(image)
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## 🔧 Configuration Architecture

### 1. Configuration Management

```python
class ConfigurationManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self):
        """Load configuration from file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def get_model_config(self):
        """Get model configuration"""
        return self.config['model']
    
    def get_training_config(self):
        """Get training configuration"""
        return self.config['training']
    
    def get_data_config(self):
        """Get data configuration"""
        return self.config['data']
```

### 2. Configuration Schema

```yaml
# config.yaml
model:
  model_type: "Mobilenet4"
  model_size: "large"
  pretrained: true
  embedding_dim: 1280

training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100
  optimizer: "adam"
  scheduler: "cosine"

data:
  dataset: "celeba_spoof"
  data_root: "/path/to/dataset"
  batch_size: 32
  num_workers: 8
  augmentation:
    mixup: true
    cutmix: true
    random_crop: true
    random_flip: true

loss:
  loss_type: "amsoftmax"
  amsoftmax:
    margin: 0.5
    scale: 1.0
    margin_type: "cross_entropy"
```

## 📊 Performance Monitoring Architecture

### 1. Training Monitoring

```python
class TrainingMonitor:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        
    def log_metrics(self, epoch, metrics):
        """Log training metrics"""
        for metric_name, value in metrics.items():
            self.writer.add_scalar(f'train/{metric_name}', value, epoch)
    
    def log_model_weights(self, model, epoch):
        """Log model weights distribution"""
        for name, param in model.named_parameters():
            self.writer.add_histogram(f'weights/{name}', param, epoch)
            self.writer.add_histogram(f'gradients/{name}', param.grad, epoch)
```

### 2. Evaluation Monitoring

```python
class EvaluationMonitor:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        
    def log_evaluation_results(self, epoch, results):
        """Log evaluation results"""
        for metric_name, value in results.items():
            self.writer.add_scalar(f'eval/{metric_name}', value, epoch)
    
    def log_confusion_matrix(self, confusion_matrix, epoch):
        """Log confusion matrix"""
        self.writer.add_image('confusion_matrix', 
                            confusion_matrix.astype(np.uint8), epoch)
```

## 🔒 Security Architecture

### 1. Model Security

```python
class ModelSecurity:
    def __init__(self, model):
        self.model = model
        
    def validate_input(self, image):
        """Validate input image"""
        # Check image dimensions
        if image.size != (224, 224):
            raise ValueError("Image must be 224x224 pixels")
        
        # Check image format
        if image.mode != 'RGB':
            raise ValueError("Image must be RGB format")
        
        return True
    
    def sanitize_input(self, image):
        """Sanitize input image"""
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if necessary
        if image.size != (224, 224):
            image = image.resize((224, 224))
        
        return image
```

### 2. API Security

```python
class APISecurity:
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.input_validator = InputValidator()
        
    def rate_limit(self, client_ip):
        """Rate limiting for API requests"""
        return self.rate_limiter.is_allowed(client_ip)
    
    def validate_request(self, request):
        """Validate API request"""
        # Check request size
        if len(request.data) > 10 * 1024 * 1024:  # 10MB limit
            raise ValueError("Request too large")
        
        # Check content type
        if request.content_type != 'application/json':
            raise ValueError("Invalid content type")
        
        return True
```

## 📈 Scalability Architecture

### 1. Distributed Training

```python
class DistributedTraining:
    def __init__(self, model, world_size, rank):
        self.model = model
        self.world_size = world_size
        self.rank = rank
        
        # Initialize distributed training
        torch.distributed.init_process_group(backend='nccl')
        self.model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank]
        )
    
    def train_step(self, batch):
        """Distributed training step"""
        # Forward pass
        outputs = self.model(batch['images'])
        loss = self.compute_loss(outputs, batch['labels'])
        
        # Backward pass
        loss.backward()
        
        # Synchronize gradients
        torch.distributed.barrier()
        
        return loss.item()
```

### 2. Model Parallelism

```python
class ModelParallelism:
    def __init__(self, model, device_ids):
        self.model = model
        self.device_ids = device_ids
        
        # Split model across devices
        self.model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    def forward(self, x):
        """Forward pass with model parallelism"""
        return self.model(x)
```

This technical architecture provides a comprehensive foundation for the face anti-spoofing system, covering all aspects from model design to deployment and monitoring.
