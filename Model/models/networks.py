"""
Neural network architectures for Fisheye Rectification.

This module contains the main network architectures:
- CoarseRectificationNet: Initial coarse geometric correction
- EnhancedFisheyeRectificationModel: Fine-grained rectification with attention
- CascadedRectificationModel: Two-stage cascaded approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .layers import (
    DeformableConv2d,
    ColorAwareAttention,
    GatedSkipConnection,
    ResidualDeformableBlock
)


class CoarseRectificationNet(nn.Module):
    """
    Coarse rectification network for initial geometric correction.
    
    This network performs a rough geometric correction of fisheye distortion
    by learning displacement fields that warp the input image. It operates
    at a lower resolution for efficiency and provides a good initialization
    for subsequent refinement.
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        feature_channels: Base number of feature channels (default: 64)
        downsample_factor: Factor by which the displacement field is downsampled (default: 16)
    """
    
    def __init__(
        self, 
        in_channels: int = 3,
        feature_channels: int = 64,
        downsample_factor: int = 16
    ):
        super(CoarseRectificationNet, self).__init__()
        self.downsample_factor = downsample_factor
        
        # Progressive feature extraction with increasing receptive field
        self.conv_layers = nn.Sequential(
            # Initial feature extraction
            nn.Conv2d(in_channels, feature_channels, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            
            # Downsample and increase features
            nn.Conv2d(feature_channels, feature_channels * 2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            
            # Further downsampling
            nn.Conv2d(feature_channels * 2, feature_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # Final downsampling
            nn.Conv2d(feature_channels * 4, feature_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # Predict displacement field (2 channels for x,y displacements)
            nn.Conv2d(feature_channels * 4, 2, kernel_size=3, stride=1, padding=1)
        )
        
        # Use tanh to bound displacements
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Predict coarse displacement field
        disp_coarse = self.conv_layers(x)
        
        # Upsample displacement field to input resolution
        disp = F.interpolate(disp_coarse, size=(H, W), mode='bilinear', align_corners=True)
        
        # Scale displacements (0.1 limits maximum displacement to 10% of image size)
        disp = self.tanh(disp) * 0.1
        
        # Create sampling grid
        grid = self._create_grid(B, H, W, x.device)
        
        # Apply displacement to grid
        grid = grid + disp.permute(0, 2, 3, 1)
        
        # Warp input image using displaced grid
        warped = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        return warped
    
    def _create_grid(self, B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        Create a normalized coordinate grid for sampling.
        
        Args:
            B: Batch size
            H: Height
            W: Width
            device: Device to create tensor on
            
        Returns:
            Grid tensor of shape (B, H, W, 2) with coordinates in [-1, 1]
        """
        # Create normalized coordinates
        xs = torch.linspace(-1, 1, W, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
        
        # Create mesh grid
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        
        # Stack and expand for batch
        grid = torch.stack((grid_x, grid_y), dim=-1)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        
        return grid


class EnhancedFisheyeRectificationModel(nn.Module):
    """
    Enhanced fisheye rectification model with attention mechanisms.
    
    This model performs fine-grained rectification using deformable convolutions,
    color-aware attention, and gated skip connections. It follows an encoder-decoder
    architecture with progressive feature extraction and reconstruction.
    
    Args:
        config: Configuration object containing model parameters
    """
    
    def __init__(self, config):
        super(EnhancedFisheyeRectificationModel, self).__init__()
        
        self.image_size = config.image_size
        self.num_blocks = config.num_blocks
        self.num_skip_connections = config.num_skip_connections
        
        # Calculate channel progression
        channels = [
            config.initial_channels * (config.growth_rate ** i) 
            for i in range(config.num_blocks)
        ]
        
        # Build encoder blocks
        self.encoder_blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            in_channels = 3 if i == 0 else channels[i - 1]
            out_channels = channels[i]
            self.encoder_blocks.append(
                self._make_encoder_block(in_channels, out_channels)
            )
        
        # Gated skip connections for selected encoder features
        self.gated_skips = nn.ModuleList([
            GatedSkipConnection(ch) 
            for ch in reversed(channels[-self.num_skip_connections:])
        ])
        
        # Build decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for i in range(self.num_skip_connections):
            encoder_channel = channels[-(i + 1)]
            if (i + 1) < self.num_skip_connections:
                out_channels = channels[-(i + 2)]
            else:
                out_channels = 3  # Final output channels
            
            self.decoder_blocks.append(
                self._make_decoder_block(encoder_channel, out_channels)
            )
        
        # Border attention for handling image boundaries
        self.border_attention = self._make_attention_block()
        
        # Color refinement for final output
        self.color_refinement = self._make_color_refinement_block()
    
    def _make_encoder_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create an encoder block with deformable convolutions and attention."""
        return nn.Sequential(
            DeformableConv2d(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ColorAwareAttention(out_channels),
            DeformableConv2d(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_block(self, encoder_channel: int, out_channels: int) -> nn.Module:
        """Create a decoder block for feature reconstruction."""
        # Input channels = encoder_channel (upsampled) + encoder_channel (skip)
        in_channels = 2 * encoder_channel
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ColorAwareAttention(out_channels)
        )
    
    def _make_attention_block(self) -> nn.Module:
        """Create attention block for border handling."""
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def _make_color_refinement_block(self) -> nn.Module:
        """Create color refinement block for final output."""
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ColorAwareAttention(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply border attention to focus on valid image regions
        attention = self.border_attention(x)
        x = x * attention
        
        # Encoder pass with feature storage
        features = []
        for block in self.encoder_blocks:
            x = block(x)
            features.append(x)
            x = F.max_pool2d(x, 2)
        
        # Select features for skip connections
        skip_features = features[-self.num_skip_connections:]
        
        # Decoder pass with skip connections
        for i in range(self.num_skip_connections):
            # Upsample
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            
            # Get and process skip features
            skip_feature = skip_features[-(i + 1)]
            skip_feature = self.gated_skips[i](skip_feature)
            
            # Concatenate with skip features
            x = torch.cat([x, skip_feature], dim=1)
            
            # Apply decoder block
            x = self.decoder_blocks[i](x)
        
        # Ensure output matches target size
        x = F.interpolate(
            x, 
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        )
        
        # Apply color refinement
        x = self.color_refinement(x)
        
        return x


class CascadedRectificationModel(nn.Module):
    """
    Cascaded model combining coarse and fine rectification stages.
    
    This model uses a two-stage approach:
    1. Coarse geometric correction for rough alignment
    2. Fine-grained refinement for detail preservation
    
    Args:
        config: Configuration object containing model parameters
    """
    
    def __init__(self, config):
        super(CascadedRectificationModel, self).__init__()
        
        # Stage 1: Coarse rectification
        self.coarse_net = CoarseRectificationNet(
            in_channels=3,
            feature_channels=config.initial_channels,
            downsample_factor=16
        )
        
        # Stage 2: Fine refinement
        self.refinement_net = EnhancedFisheyeRectificationModel(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1: Coarse geometric correction
        coarse = self.coarse_net(x)
        
        # Stage 2: Fine-grained refinement
        refined = self.refinement_net(coarse)
        
        return refined
    
    def get_intermediate_outputs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coarse = self.coarse_net(x)
        refined = self.refinement_net(coarse)
        return coarse, refined