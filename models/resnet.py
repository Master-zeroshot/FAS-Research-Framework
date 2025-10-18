"""
ResNet Implementation for Face Anti-Spoofing

This module implements ResNet architectures (18, 34, 50, 101, 152) optimized for face anti-spoofing tasks.
ResNet uses residual connections to enable training of very deep networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable


class BasicBlock(nn.Module):
    """Basic ResNet block for ResNet-18 and ResNet-34"""
    
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """Bottleneck ResNet block for ResNet-50, ResNet-101, and ResNet-152"""
    
    expansion = 4
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class ResNet(nn.Module):
    """ResNet base class"""
    
    def __init__(self, block, layers, num_classes: int = 2, embeding_dim: int = 512, 
                 multi_heads: bool = False, zero_init_residual: bool = False):
        super().__init__()
        
        self.embeding_dim = embeding_dim
        self.num_classes = num_classes
        self.multi_heads = multi_heads
        
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Feature extraction
        self.features = nn.Linear(512 * block.expansion, embeding_dim)
        
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
        
        # Initialize weights
        self._init_weights(zero_init_residual)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        """Make a ResNet layer"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, zero_init_residual):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Features
        features = self.features(x)
        
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


def resnet18(embeding_dim: int = 512, num_classes: int = 2, multi_heads: bool = False, **kwargs):
    """ResNet-18"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, embeding_dim, multi_heads, **kwargs)


def resnet34(embeding_dim: int = 512, num_classes: int = 2, multi_heads: bool = False, **kwargs):
    """ResNet-34"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, embeding_dim, multi_heads, **kwargs)


def resnet50(embeding_dim: int = 2048, num_classes: int = 2, multi_heads: bool = False, **kwargs):
    """ResNet-50"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, embeding_dim, multi_heads, **kwargs)


def resnet101(embeding_dim: int = 2048, num_classes: int = 2, multi_heads: bool = False, **kwargs):
    """ResNet-101"""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, embeding_dim, multi_heads, **kwargs)


def resnet152(embeding_dim: int = 2048, num_classes: int = 2, multi_heads: bool = False, **kwargs):
    """ResNet-152"""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, embeding_dim, multi_heads, **kwargs)


# Convenience function to get ResNet by name
def get_resnet(model_name: str, **kwargs):
    """Get ResNet model by name"""
    models = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown ResNet model: {model_name}")
    
    return models[model_name](**kwargs)
