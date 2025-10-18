"""
Vision Transformer (ViT) Implementation for Face Anti-Spoofing

This module implements Vision Transformer architectures optimized for face anti-spoofing tasks.
ViT applies transformer architecture to image patches for computer vision tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PatchEmbedding(nn.Module):
    """Patch embedding layer for Vision Transformer"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, 
                 embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, n_patches, n_patches)
        x = self.projection(x)
        # Flatten to (B, embed_dim, n_patches^2)
        x = x.flatten(2)
        # Transpose to (B, n_patches^2, embed_dim)
        x = x.transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """Multi-layer perceptron for transformer blocks"""
    
    def __init__(self, embed_dim: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP"""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int = 4, 
                 dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
    
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # MLP with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class VisionTransformer(nn.Module):
    """Vision Transformer for face anti-spoofing"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12, 
                 mlp_ratio: int = 4, dropout: float = 0.1, drop_path: float = 0.1,
                 embeding_dim: int = 768, num_classes: int = 2, multi_heads: bool = False):
        super().__init__()
        
        self.embeding_dim = embeding_dim
        self.num_classes = num_classes
        self.multi_heads = multi_heads
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, dpr[i])
            for i in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Head
        self.head = nn.Linear(embed_dim, embeding_dim)
        
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
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights_module)
    
    def _init_weights_module(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Use cls token for classification
        cls_token = x[:, 0]  # (B, embed_dim)
        
        # Features
        features = self.head(cls_token)
        
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


def vit_tiny(embeding_dim: int = 768, num_classes: int = 2, multi_heads: bool = False, **kwargs):
    """ViT-Tiny"""
    return VisionTransformer(
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        embeding_dim=embeding_dim, num_classes=num_classes, multi_heads=multi_heads, **kwargs
    )


def vit_small(embeding_dim: int = 768, num_classes: int = 2, multi_heads: bool = False, **kwargs):
    """ViT-Small"""
    return VisionTransformer(
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        embeding_dim=embeding_dim, num_classes=num_classes, multi_heads=multi_heads, **kwargs
    )


def vit_base(embeding_dim: int = 768, num_classes: int = 2, multi_heads: bool = False, **kwargs):
    """ViT-Base"""
    return VisionTransformer(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        embeding_dim=embeding_dim, num_classes=num_classes, multi_heads=multi_heads, **kwargs
    )


def vit_large(embeding_dim: int = 1024, num_classes: int = 2, multi_heads: bool = False, **kwargs):
    """ViT-Large"""
    return VisionTransformer(
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        embeding_dim=embeding_dim, num_classes=num_classes, multi_heads=multi_heads, **kwargs
    )


def vit_huge(embeding_dim: int = 1280, num_classes: int = 2, multi_heads: bool = False, **kwargs):
    """ViT-Huge"""
    return VisionTransformer(
        embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        embeding_dim=embeding_dim, num_classes=num_classes, multi_heads=multi_heads, **kwargs
    )


# Convenience function to get ViT by name
def get_vit(model_name: str, **kwargs):
    """Get Vision Transformer model by name"""
    models = {
        'vit_tiny': vit_tiny,
        'vit_small': vit_small,
        'vit_base': vit_base,
        'vit_large': vit_large,
        'vit_huge': vit_huge,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown ViT model: {model_name}")
    
    return models[model_name](**kwargs)
