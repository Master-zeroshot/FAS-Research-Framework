"""
Optimized MobileNetV4 Configuration for Face Anti-Spoofing

This configuration file contains optimized hyperparameters specifically tuned
for MobileNetV4 models on face anti-spoofing tasks.
"""

from .paths import get_dataset_path, get_log_path, get_model_path

# Experiment configuration
exp_num = 0
dataset = "celeba_spoof"
multi_task_learning = True
evaluation = True
test_steps = None
random_seed = 42

# Dataset paths
datasets = dict(
    Celeba_root=get_dataset_path("celeba_spoof"),
    Casia_root=get_dataset_path("casia"),
    LCCFASD_root=get_dataset_path("lcc_fasd"),
)

# Image normalization (optimized for face anti-spoofing)
img_norm_cfg = dict(mean=[0.5931, 0.4690, 0.4229], std=[0.2471, 0.2214, 0.2157])

# Optimized optimizer settings for MobileNetV4
optimizer = dict(
    lr=0.003,  # Lower learning rate for MobileNetV4
    momentum=0.9,
    weight_decay=1e-4,  # Reduced weight decay
    nesterov=True  # Enable Nesterov momentum
)

# Learning rate scheduler (optimized for MobileNetV4)
scheduler = dict(
    milestones=[15, 30, 45],  # Adjusted milestones for MobileNetV4
    gamma=0.3,  # More aggressive decay
    warmup_epochs=5,  # Add warmup
    warmup_factor=0.1
)

# Data loading configuration
data = dict(
    batch_size=24,  # Optimized batch size for MobileNetV4
    data_loader_workers=8,
    sampler=True,
    pin_memory=True,
    drop_last=True  # Ensure consistent batch sizes
)

# Image resize configuration
resize = dict(height=224, width=224)

# Checkpoint configuration
checkpoint = dict(
    snapshot_name="mobilenetv4_optimized_224_antispoof.pth.tar",
    experiment_path=get_log_path("logs_mobilenetv4_optimized"),
)

# Loss configuration (optimized for MobileNetV4)
loss = dict(
    loss_type="amsoftmax",
    amsoftmax=dict(
        m=0.4,  # Reduced margin for MobileNetV4
        s=15,  # Increased scale
        margin_type="cos",  # Use cosine margin
        label_smooth=True,
        smoothing=0.1,
        ratio=[1, 1],
        gamma=0.1  # Add gamma for better convergence
    ),
)

# Training epochs (optimized for MobileNetV4)
epochs = dict(start_epoch=0, max_epoch=60)  # Reduced epochs due to better convergence

# MobileNetV4 model configuration
model = dict(
    model_type="Mobilenet4",
    model_size="large",  # Use large variant for best performance
    width_mult=1.0,
    pretrained=True,  # Use pretrained weights
    embeding_dim=1280,
    imagenet_weights=get_model_path("mobilenetv4_large"),
    # MobileNetV4 specific parameters
    use_attention=True,  # Enable attention mechanisms
    use_squeeze_excitation=True,  # Enable SE blocks
    dropout_rate=0.2,  # Optimized dropout rate
)

# Data augmentation (optimized for MobileNetV4)
aug = dict(
    type_aug="mixup",  # Use mixup for better generalization
    alpha=0.4,  # Optimized alpha for MobileNetV4
    beta=0.4,  # Optimized beta for MobileNetV4
    aug_prob=0.8,  # Higher augmentation probability
    # Additional MobileNetV4 specific augmentations
    use_cutmix=True,
    cutmix_alpha=1.0,
    cutmix_prob=0.5,
    use_autoaugment=True,
    autoaugment_policy="imagenet"
)

# Visualization curves
curves = dict(
    det_curve="det_curve_mobilenetv4_optimized.png",
    roc_curve="roc_curve_mobilenetv4_optimized.png"
)

# Dropout configuration (optimized for MobileNetV4)
dropout = dict(
    prob_dropout=0.2,  # Optimized dropout probability
    classifier=0.3,  # Higher classifier dropout
    type="gaussian",  # Use Gaussian dropout
    mu=0.5,
    sigma=0.2,  # Reduced sigma for better regularization
    # MobileNetV4 specific dropout
    attention_dropout=0.1,
    path_dropout=0.1
)

# RSC (Random Subspace Classifier) configuration
RSC = dict(
    use_rsc=True,  # Enable RSC for MobileNetV4
    p=0.3,  # Optimized p value
    b=0.3,  # Optimized b value
    rsc_epochs=10  # Start RSC after 10 epochs
)

# Test dataset configuration
test_dataset = dict(type="LCC_FASD")

# Convolutional channel dropout
conv_cd = dict(theta=0.1)  # Enable channel dropout

# Test file name
test_file_name = "test_mobilenetv4_optimized"

# MobileNetV4 specific training parameters
mobilenetv4_params = dict(
    # Block configuration
    use_universal_blocks=True,
    use_extra_depthwise=True,
    use_feed_forward=True,
    
    # Training optimizations
    use_gradient_clipping=True,
    max_grad_norm=1.0,
    use_ema=True,  # Exponential moving average
    ema_decay=0.999,
    
    # Learning rate scheduling
    use_cosine_annealing=True,
    cosine_restarts=True,
    t_max=60,
    
    # Regularization
    use_label_smoothing=True,
    label_smoothing_factor=0.1,
    use_mixup=True,
    mixup_alpha=0.4,
    
    # Performance optimizations
    use_amp=True,  # Automatic mixed precision
    use_compile=True,  # PyTorch 2.0 compilation
    use_channels_last=True,  # Memory layout optimization
)

# Advanced training techniques
advanced_training = dict(
    # Knowledge distillation
    use_distillation=False,
    teacher_model=None,
    distillation_alpha=0.7,
    distillation_temperature=3.0,
    
    # Adversarial training
    use_adversarial=False,
    adversarial_alpha=0.1,
    adversarial_epsilon=0.01,
    
    # Self-supervised learning
    use_self_supervised=False,
    self_supervised_weight=0.1,
    
    # Meta-learning
    use_meta_learning=False,
    meta_lr=0.01,
    meta_steps=5
)

# Evaluation configuration
evaluation_config = dict(
    # Metrics to compute
    metrics=["accuracy", "auc", "eer", "apcer", "bpcer", "acer", "hter"],
    
    # Evaluation frequency
    eval_frequency=5,  # Evaluate every 5 epochs
    
    # Best model selection
    best_metric="auc",
    best_mode="max",
    
    # Early stopping
    use_early_stopping=True,
    patience=10,
    min_delta=0.001
)

# Logging configuration
logging_config = dict(
    # Logging frequency
    log_frequency=100,  # Log every 100 batches
    
    # TensorBoard logging
    use_tensorboard=True,
    log_dir="./logs_mobilenetv4_optimized/tensorboard",
    
    # WandB logging
    use_wandb=False,
    wandb_project="face-anti-spoofing-mobilenetv4",
    wandb_entity=None,
    
    # Model checkpointing
    save_frequency=5,  # Save every 5 epochs
    keep_checkpoints=3,  # Keep last 3 checkpoints
)

# Performance monitoring
performance_config = dict(
    # Memory monitoring
    monitor_memory=True,
    memory_threshold=0.9,  # Alert at 90% memory usage
    
    # GPU monitoring
    monitor_gpu=True,
    gpu_memory_threshold=0.9,
    
    # Training speed monitoring
    monitor_speed=True,
    target_speed=100,  # Target samples per second
    
    # Automatic optimization
    auto_optimize=True,
    optimization_frequency=10  # Optimize every 10 epochs
)
