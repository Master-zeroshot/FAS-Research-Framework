"""
EfficientNet Implementation for Face Anti-Spoofing

This module implements EfficientNet architectures (B0-B7) optimized for face anti-spoofing tasks.
EfficientNet uses compound scaling to achieve better accuracy and efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
import math


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""
    
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for EfficientNet"""
    
    def __init__(self, in_channels: int, se_ratio: float = 0.25):
        super().__init__()
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, se_channels, 1),
            Swish(),
            nn.Conv2d(se_channels, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution block for EfficientNet"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, expand_ratio: int, se_ratio: float = 0.25,
                 drop_connect_rate: float = 0.0):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, expanded_channels, 1, bias=False)
            self.expand_bn = nn.BatchNorm2d(expanded_channels)
            self.expand_swish = Swish()
        else:
            self.expand_conv = None
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(
            expanded_channels, expanded_channels, kernel_size, stride,
            padding=kernel_size//2, groups=expanded_channels, bias=False
        )
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels)
        self.depthwise_swish = Swish()
        
        # Squeeze-and-Excitation
        self.se = SqueezeExcitation(expanded_channels, se_ratio)
        
        # Projection phase
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
        # Drop connect
        self.drop_connect_rate = drop_connect_rate
    
    def forward(self, x):
        # Expansion
        if self.expand_conv is not None:
            x = self.expand_conv(x)
            x = self.expand_bn(x)
            x = self.expand_swish(x)
        
        # Depthwise convolution
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_swish(x)
        
        # Squeeze-and-Excitation
        x = self.se(x)
        
        # Projection
        x = self.project_conv(x)
        x = self.project_bn(x)
        
        # Skip connection
        if self.stride == 1 and self.in_channels == self.out_channels:
            if self.drop_connect_rate > 0:
                x = self._drop_connect(x)
            x = x + x  # Skip connection
        
        return x
    
    def _drop_connect(self, x):
        """Drop connect implementation"""
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        random_tensor = keep_prob + torch.rand(x.shape[0], 1, 1, 1, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class EfficientNet(nn.Module):
    """EfficientNet base class"""
    
    def __init__(self, width_coefficient: float = 1.0, depth_coefficient: float = 1.0,
                 dropout_rate: float = 0.2, drop_connect_rate: float = 0.2,
                 embeding_dim: int = 1280, num_classes: int = 2, multi_heads: bool = False):
        super().__init__()
        
        self.embeding_dim = embeding_dim
        self.num_classes = num_classes
        self.multi_heads = multi_heads
        
        # Calculate scaled dimensions
        def round_filters(filters):
            multiplier = width_coefficient
            divisor = 8
            min_depth = None
            new_filters = int(filters * multiplier)
            if min_depth is not None:
                new_filters = max(min_depth, new_filters)
            return int(new_filters + divisor / 2) // divisor * divisor
        
        def round_repeats(repeats):
            return int(math.ceil(depth_coefficient * repeats))
        
        # Stem
        out_channels = round_filters(32)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            Swish()
        )
        
        # Build blocks
        blocks = []
        in_channels = out_channels
        
        # Block configurations (expand_ratio, kernel_size, stride, repeats, se_ratio)
        block_configs = [
            (1, 3, 1, 1, 0.25),  # MBConv1
            (6, 3, 2, 2, 0.25),  # MBConv6
            (6, 5, 2, 2, 0.25),  # MBConv6
            (6, 3, 2, 3, 0.25),  # MBConv6
            (6, 5, 1, 3, 0.25),  # MBConv6
            (6, 5, 2, 4, 0.25),  # MBConv6
            (6, 3, 1, 1, 0.25),  # MBConv6
        ]
        
        total_blocks = sum(round_repeats(repeats) for _, _, _, repeats, _ in block_configs)
        block_idx = 0
        
        for expand_ratio, kernel_size, stride, repeats, se_ratio in block_configs:
            repeats = round_repeats(repeats)
            for i in range(repeats):
                out_channels = round_filters(16 * (2 ** (block_idx // 2)))
                stride = stride if i == 0 else 1
                
                block = MBConvBlock(
                    in_channels, out_channels, kernel_size, stride,
                    expand_ratio, se_ratio, drop_connect_rate * block_idx / total_blocks
                )
                blocks.append(block)
                
                in_channels = out_channels
                block_idx += 1
        
        self.blocks = nn.Sequential(*blocks)
        
        # Head
        final_channels = round_filters(1280)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, bias=False),
            nn.BatchNorm2d(final_channels),
            Swish()
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classifier
        self.classifier = nn.Linear(final_channels, embeding_dim)
        
        # Multi-head outputs for face anti-spoofing
        if multi_heads:
            self.spoofer = nn.Sequential(
                nn.Linear(embeding_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 2)  # Spoof/Real classification
            )
            
            self.spoof_type = nn.Sequential(
                nn.Linear(embeding_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 40)  # Spoof type classification
            )
            
            self.lighting = nn.Sequential(
                nn.Linear(embeding_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 3)  # Lighting condition classification
            )
            
            self.attributes = nn.Sequential(
                nn.Linear(embeding_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 40)  # Attribute classification
            )
        else:
            self.spoofer = nn.Linear(embeding_dim, num_classes)
    
    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # Blocks
        x = self.blocks(x)
        
        # Head
        x = self.head(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Dropout
        x = self.dropout(x)
        
        # Features
        features = self.classifier(x)
        
        if self.multi_heads:
            # Multi-task learning
            spoofer_logits = self.spoofer(features)
            spoof_type_logits = self.spoof_type(features)
            lighting_logits = self.lighting(features)
            attributes_logits = self.attributes(features)
            
            return (spoofer_logits, spoof_type_logits, lighting_logits, attributes_logits)
        else:
            # Single task learning
            spoofer_logits = self.spoofer(features)
            return spoofer_logits
    
    def make_logits(self, features, all=False):
        """Make logits from features for multi-head learning"""
        if not self.multi_heads:
            return self.spoofer(features)
        
        if all:
            return [
                self.spoofer(features),
                self.spoof_type(features),
                self.lighting(features),
                self.attributes(features)
            ]
        else:
            return self.spoofer(features)


def efficientnet_b0(embeding_dim: int = 1280, num_classes: int = 2, multi_heads: bool = False, **kwargs):
    """EfficientNet-B0"""
    return EfficientNet(
        width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2,
        embeding_dim=embeding_dim, num_classes=num_classes, multi_heads=multi_heads, **kwargs
    )


def efficientnet_b1(embeding_dim: int = 1280, num_classes: int = 2, multi_heads: bool = False, **kwargs):
    """EfficientNet-B1"""
    return EfficientNet(
        width_coefficient=1.0, depth_coefficient=1.1, dropout_rate=0.2,
        embeding_dim=embeding_dim, num_classes=num_classes, multi_heads=multi_heads, **kwargs
    )


def efficientnet_b2(embeding_dim: int = 1280, num_classes: int = 2, multi_heads: bool = False, **kwargs):
    """EfficientNet-B2"""
    return EfficientNet(
        width_coefficient=1.1, depth_coefficient=1.2, dropout_rate=0.3,
        embeding_dim=embeding_dim, num_classes=num_classes, multi_heads=multi_heads, **kwargs
    )


def efficientnet_b3(embeding_dim: int = 1280, num_classes: int = 2, multi_heads: bool = False, **kwargs):
    """EfficientNet-B3"""
    return EfficientNet(
        width_coefficient=1.2, depth_coefficient=1.4, dropout_rate=0.3,
        embeding_dim=embeding_dim, num_classes=num_classes, multi_heads=multi_heads, **kwargs
    )


def efficientnet_b4(embeding_dim: int = 1280, num_classes: int = 2, multi_heads: bool = False, **kwargs):
    """EfficientNet-B4"""
    return EfficientNet(
        width_coefficient=1.4, depth_coefficient=1.8, dropout_rate=0.4,
        embeding_dim=embeding_dim, num_classes=num_classes, multi_heads=multi_heads, **kwargs
    )


def efficientnet_b5(embeding_dim: int = 1280, num_classes: int = 2, multi_heads: bool = False, **kwargs):
    """EfficientNet-B5"""
    return EfficientNet(
        width_coefficient=1.6, depth_coefficient=2.2, dropout_rate=0.4,
        embeding_dim=embeding_dim, num_classes=num_classes, multi_heads=multi_heads, **kwargs
    )


def efficientnet_b6(embeding_dim: int = 1280, num_classes: int = 2, multi_heads: bool = False, **kwargs):
    """EfficientNet-B6"""
    return EfficientNet(
        width_coefficient=1.8, depth_coefficient=2.6, dropout_rate=0.5,
        embeding_dim=embeding_dim, num_classes=num_classes, multi_heads=multi_heads, **kwargs
    )


def efficientnet_b7(embeding_dim: int = 1280, num_classes: int = 2, multi_heads: bool = False, **kwargs):
    """EfficientNet-B7"""
    return EfficientNet(
        width_coefficient=2.0, depth_coefficient=3.1, dropout_rate=0.5,
        embeding_dim=embeding_dim, num_classes=num_classes, multi_heads=multi_heads, **kwargs
    )


# Convenience function to get EfficientNet by name
def get_efficientnet(model_name: str, **kwargs):
    """Get EfficientNet model by name"""
    models = {
        'efficientnet_b0': efficientnet_b0,
        'efficientnet_b1': efficientnet_b1,
        'efficientnet_b2': efficientnet_b2,
        'efficientnet_b3': efficientnet_b3,
        'efficientnet_b4': efficientnet_b4,
        'efficientnet_b5': efficientnet_b5,
        'efficientnet_b6': efficientnet_b6,
        'efficientnet_b7': efficientnet_b7,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown EfficientNet model: {model_name}")
    
    return models[model_name](**kwargs)
