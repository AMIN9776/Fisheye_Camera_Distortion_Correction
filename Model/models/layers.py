"""
Custom layers for Fisheye Rectification Model.

This module contains specialized neural network layers including:
- Deformable Convolution for handling geometric distortions
- Color-Aware Attention for enhanced color preservation
- Gated Skip Connections for selective feature propagation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops
from typing import Tuple, Optional


class DeformableConv2d(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = False
    ):
        super(DeformableConv2d, self).__init__()
        
        # Ensure kernel_size is a tuple
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = padding
        self.dilation = dilation
        
        # Offset convolution - learns 2D offsets for each kernel position
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size[0] * kernel_size[1],  # 2 coords (x,y) per kernel element
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True
        )
        
        # Initialize offsets to zero (no deformation initially)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        # Modulation convolution - learns importance weights for each offset
        self.modulator_conv = nn.Conv2d(
            in_channels,
            kernel_size[0] * kernel_size[1],  # One weight per kernel element
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True
        )
        
        # Initialize modulation to zero (sigmoid(0) = 0.5)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        # Regular convolution for feature transformation
        self.regular_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Learn offsets for sampling positions
        offset = self.offset_conv(x)
        
        # Learn modulation weights (importance of each offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        # Apply deformable convolution using torchvision ops
        x = torchvision.ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
            mask=modulator,
            stride=self.stride,
            dilation=self.dilation
        )
        
        return x


class ColorAwareAttention(nn.Module):
    
    def __init__(self, channels: int):
        super(ColorAwareAttention, self).__init__()
        self.channels = channels
        
        # Attention branch for green-like channels (middle channels)
        self.green_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Attention branch for other channels
        self.other_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute attention maps
        green_attention = self.green_attention(x)
        other_attention = self.other_attention(x)
        
        # Create weighted attention focusing on middle channels
        # (assuming they correspond to green-like features)
        n_channels = x.size(1)
        mid_point = n_channels // 2
        
        # Create channel-wise weight tensor
        attention = torch.ones_like(x)
        
        # Boost attention for middle channels (green-like)
        # This gives 20% more weight to middle channels
        attention[:, mid_point - n_channels//6 : mid_point + n_channels//6, :, :] *= 1.2
        
        # Combine both attention mechanisms
        final_attention = (green_attention * attention + other_attention * (2 - attention)) / 2
        
        # Apply attention to input
        return x * final_attention


class GatedSkipConnection(nn.Module):
    
    def __init__(self, in_channels: int):
        super(GatedSkipConnection, self).__init__()
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Learn gate values (0-1) for each feature
        gate_weights = self.gate(x)
        
        # Apply gate to features
        return x * gate_weights


class SpatialAttentionBlock(nn.Module):
    
    def __init__(self, in_channels: int, reduction_ratio: int = 8):
        super(SpatialAttentionBlock, self).__init__()
        
        # Channel reduction for efficiency
        reduced_channels = max(in_channels // reduction_ratio, 1)
        
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_map = self.attention(x)
        return x * attention_map


class ResidualDeformableBlock(nn.Module):
    
    def __init__(self, channels: int, use_attention: bool = True):
        super(ResidualDeformableBlock, self).__init__()
        
        self.conv1 = DeformableConv2d(channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = DeformableConv2d(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = ColorAwareAttention(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_attention:
            out = self.attention(out)
        
        # Residual connection
        out = out + identity
        out = self.relu(out)
        
        return out