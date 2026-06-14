"""
Baseline Multimodal Model for Summer School Challenge

Simplified architecture for building damage assessment using:
- Satellite imagery (pre/post-disaster images)
- Climate data (ERA-5 environmental variables)
- Attention mechanism for interpretability (NO event masks)

This model is designed to be:
1. Simple and understandable
2. Efficient to train
3. Interpretable with attention weights
4. A good starting point for participants to improve upon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .shared_components import CNNBackbone, DepthwiseSeparableConv2d


class SimpleClimateEncoder(nn.Module):
    """
    Simple climate feature encoder.
    
    Processes ERA-5 climate variables without event masking.
    Input:  (B, C=10, T, H=20, W=20)
    Output: (B, T, climate_dim)
    """
    
    def __init__(self, in_channels: int = 10, out_dim: int = 64):
        super().__init__()
        # Spatial encoding: per-timestep independent 2D convolutions
        self.spatial_encoder = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels, 32),
            DepthwiseSeparableConv2d(32, out_dim),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, H, W)
        Returns: (B, T, out_dim)
        """
        B, C, T, H, W = x.shape
        # Process each timestep independently
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.spatial_encoder(x)
        x = x.reshape(B, T, self.out_dim)
        return x


class SimpleTemporalEncoder(nn.Module):
    """
    Simple temporal encoder using 1D convolutions.
    
    Processes climate time series without recurrence.
    Input:  (B, T, in_dim)
    Output: (B, T, out_dim)
    """
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        
        # Two simple 1D conv layers with increasing dilation
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_dim)
        
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, in_dim)
        Returns: (B, T, out_dim)
        """
        x = self.proj(x)  # (B, T, out_dim)
        x = x.transpose(1, 2)  # (B, out_dim, T)
        
        # Conv block 1
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + res
        
        # Conv block 2
        res = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + res
        
        x = x.transpose(1, 2)  # (B, T, out_dim)
        return x


class SimpleAttention(nn.Module):
    """
    Simple self-attention for climate time series.
    
    Learns to focus on relevant timesteps automatically (no labels required).
    Returns both context vector and attention weights for visualization.
    """
    
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, dim)
        Returns: context (B, dim), att_weights (B, T)
        """
        B, T, D = x.shape
        H = self.num_heads
        head_dim = D // H

        # Self-attention: query from mean of sequence
        query = x.mean(dim=1)  # (B, D)
        
        q = self.q_proj(query).reshape(B, H, head_dim)  # (B, H, head_dim)
        k = self.k_proj(x).reshape(B, T, H, head_dim).permute(0, 2, 1, 3)  # (B, H, T, hd)
        v = self.v_proj(x).reshape(B, T, H, head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        attn = (q.unsqueeze(2) @ k.transpose(-1, -2)) * self.scale  # (B, H, 1, T)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Context: weighted sum of values
        ctx_heads = (attn @ v).squeeze(2)  # (B, H, head_dim)
        ctx = self.out_proj(ctx_heads.reshape(B, D))
        ctx = self.norm(ctx)

        # Average attention weights across heads
        att_weights = attn.mean(dim=1).squeeze(1)  # (B, T)
        
        return ctx, att_weights


class BaselineMultimodalModel(nn.Module):
    """
    Baseline Multimodal Model for Summer School Challenge.
    
    Architecture:
    - Visual branch: CNN encoding of pre/post satellite images
    - Climate branch: Simple temporal encoder + attention
    - Fusion: Concatenate and classify
    
    Key features:
    - NO event masks (uses only satellite + climate)
    - Simple and understandable architecture
    - Attention mechanism for interpretability
    - Good baseline for participants to improve
    
    Parameters
    ----------
    num_classes : int
        Number of damage classes (default 4)
    dropout_rate : float
        Dropout rate for regularization
    backbone : str
        Visual backbone type ('cnn' only)
    vis_dim : int
        Visual features dimension
    climate_dim : int
        Climate features dimension
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        dropout_rate: float = 0.2,
        backbone: str = 'cnn',
        vis_dim: int = 256,
        climate_dim: int = 128,
    ):
        super().__init__()
        
        assert backbone == 'cnn', "Only CNN backbone supported in baseline"
        
        # Visual branch: shared CNN for pre and post images
        self.visual_backbone = CNNBackbone(out_dim=vis_dim)
        
        # Climate branch: encoder + temporal processing + attention
        spatial_dim = 64
        self.climate_spatial = SimpleClimateEncoder(in_channels=10, out_dim=spatial_dim)
        self.climate_temporal = SimpleTemporalEncoder(in_dim=spatial_dim, out_dim=climate_dim, 
                                                      dropout=dropout_rate * 0.5)
        self.climate_attention = SimpleAttention(dim=climate_dim, num_heads=max(1, climate_dim // 32),
                                                 dropout=dropout_rate * 0.5)
        
        # Classifier: simple MLPs
        fusion_dim = vis_dim * 2 + climate_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes),
        )
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x_pre: torch.Tensor,
        x_post: torch.Tensor,
        x_climate: torch.Tensor,
        event_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the baseline model.
        
        Parameters
        ----------
        x_pre : torch.Tensor
            Pre-disaster satellite image, shape (B, 3, 64, 64)
        x_post : torch.Tensor
            Post-disaster satellite image, shape (B, 3, 64, 64)
        x_climate : torch.Tensor
            Climate time series, shape (B, 10, T, 20, 20)
        event_labels : torch.Tensor, optional
            Event labels (not used in baseline, for interface compatibility)
        
        Returns
        -------
        logits : torch.Tensor
            Class logits, shape (B, num_classes)
        att_weights : torch.Tensor
            Attention weights, shape (B, T) for visualization
        """
        # Extract visual features from satellite images
        f_pre = self.visual_backbone(x_pre)    # (B, vis_dim)
        f_post = self.visual_backbone(x_post)  # (B, vis_dim)
        
        # Process climate data
        z = self.climate_spatial(x_climate)     # (B, T, 64)
        z = self.climate_temporal(z)            # (B, T, climate_dim)
        f_climate, att_weights = self.climate_attention(z)  # (B, climate_dim), (B, T)
        
        # Fusion: concatenate all features
        x = torch.cat([f_pre, f_post, f_climate], dim=1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits, att_weights


def create_baseline_model(
    num_classes: int = 4,
    dropout_rate: float = 0.2,
    backbone: str = 'cnn',
    vis_dim: int = 256,
    climate_dim: int = 128,
    **kwargs
) -> BaselineMultimodalModel:
    """
    Factory function to create baseline model.
    
    Parameters
    ----------
    num_classes : int
        Number of output classes
    dropout_rate : float
        Dropout rate
    backbone : str
        Visual backbone ('cnn' only)
    vis_dim : int
        Visual features dimension
    climate_dim : int
        Climate features dimension
    **kwargs
        Additional arguments (ignored for compatibility)
    
    Returns
    -------
    model : BaselineMultimodalModel
        Instantiated baseline model
    """
    return BaselineMultimodalModel(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        backbone=backbone,
        vis_dim=vis_dim,
        climate_dim=climate_dim,
    )
