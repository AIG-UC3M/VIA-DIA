"""
Shared components used across all DIA model architectures.

Contains visual backbones and common utility layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Visual backbones (shared across all architectures)
# ──────────────────────────────────────────────────────────────────────────────

class DamageConvBlock(nn.Module):
    """Basic conv block: Conv2d → BN → ReLU × 2."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class CNNBackbone(nn.Module):
    """
    Lightweight CNN backbone for satellite image encoding.
    
    Input:  (B, 3, H, W)
    Output: (B, out_dim)
    """
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            DamageConvBlock(3, 64),
            nn.MaxPool2d(2),
            DamageConvBlock(64, 128),
            nn.MaxPool2d(2),
            DamageConvBlock(128, out_dim),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.features(x).flatten(1)   # (B, out_dim)


# ──────────────────────────────────────────────────────────────────────────────
# Common utilities
# ──────────────────────────────────────────────────────────────────────────────

class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise-separable 2-D convolution.
    ~9× fewer MACs than standard Conv2d with minimal accuracy loss.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel, stride=stride,
                            padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))), inplace=True)


def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Stochastic depth (drop path) for residual blocks."""
    if drop_prob == 0. or not training:
        return x
    keep = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep).div_(keep)
    return x * mask


class DropPath(nn.Module):
    """Stochastic depth wrapper."""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return _drop_path(x, self.drop_prob, self.training)
