"""
EfficientNet-B0 Configuration for Face Anti-Spoofing

This configuration file is optimized for EfficientNet-B0 on face anti-spoofing tasks.
EfficientNet-B0 provides a good balance between accuracy and efficiency.
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

# Optimizer settings for EfficientNet
optimizer = dict(
    lr=0.001,  # Lower learning rate for EfficientNet
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)

# Learning rate scheduler
scheduler = dict(
    milestones=[20, 40, 60],
    gamma=0.1,
    warmup_epochs=5,
    warmup_factor=0.1
)

# Data loading configuration
data = dict(
    batch_size=32,  # EfficientNet can handle larger batches
    data_loader_workers=8,
    sampler=True,
    pin_memory=True,
    drop_last=True
)

# Image resize configuration
resize = dict(height=224, width=224)

# Checkpoint configuration
checkpoint = dict(
    snapshot_name="efficientnet_b0_224_antispoof.pth.tar",
    experiment_path=get_log_path("logs_efficientnet_b0"),
)

# Loss configuration
loss = dict(
    loss_type="amsoftmax",
    amsoftmax=dict(
        m=0.3,
        s=15,
        margin_type="cos",
        label_smooth=True,
        smoothing=0.1,
        ratio=[1, 1],
        gamma=0.1
    ),
)

# Training epochs
epochs = dict(start_epoch=0, max_epoch=80)

# EfficientNet-B0 model configuration
model = dict(
    model_type="EfficientNet",
    model_size="b0",
    pretrained=True,
    embeding_dim=1280,
    imagenet_weights=get_model_path("efficientnet_b0"),
)

# Data augmentation
aug = dict(
    type_aug="mixup",
    alpha=0.4,
    beta=0.4,
    aug_prob=0.8,
    use_cutmix=True,
    cutmix_alpha=1.0,
    cutmix_prob=0.5
)

# Visualization curves
curves = dict(
    det_curve="det_curve_efficientnet_b0.png",
    roc_curve="roc_curve_efficientnet_b0.png"
)

# Dropout configuration
dropout = dict(
    prob_dropout=0.2,
    classifier=0.3,
    type="gaussian",
    mu=0.5,
    sigma=0.2
)

# RSC configuration
RSC = dict(
    use_rsc=True,
    p=0.3,
    b=0.3,
    rsc_epochs=10
)

# Test dataset configuration
test_dataset = dict(type="LCC_FASD")

# Convolutional channel dropout
conv_cd = dict(theta=0.1)

# Test file name
test_file_name = "test_efficientnet_b0"
