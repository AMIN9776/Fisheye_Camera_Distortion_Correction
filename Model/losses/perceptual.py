import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Optional


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG-16 features.
    
    This loss computes the feature-space distance between predicted and target
    images using features extracted from different layers of a pre-trained VGG-16
    network. This helps maintain perceptual quality during rectification.
    
    Args:
        layer_weights: Weights for different VGG layers (default: [1.0, 1.0, 1.0, 1.0])
        use_normalized_features: Whether to normalize features before computing loss
        device: Device to load the model on
    """
    
    def __init__(
        self,
        layer_weights: Optional[List[float]] = None,
        use_normalized_features: bool = False,
        device: Optional[str] = None
    ):
        super(PerceptualLoss, self).__init__()
        
        # Set default layer weights if not provided
        if layer_weights is None:
            layer_weights = [1.0, 1.0, 1.0, 1.0]
        
        self.layer_weights = layer_weights
        self.use_normalized_features = use_normalized_features
        
        # Load pre-trained VGG-16 and extract feature layers
        vgg = models.vgg16(pretrained=True).features
        
        # Extract specific layers for perceptual loss
        # These correspond to relu outputs after different conv blocks
        self.blocks = nn.ModuleList([
            vgg[:4],    # relu1_2 (64 channels)
            vgg[4:9],   # relu2_2 (128 channels)
            vgg[9:16],  # relu3_3 (256 channels)
            vgg[16:23]  # relu4_3 (512 channels)
        ])
        
        # Freeze VGG parameters (we don't want to train VGG)
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False
        
        # Move to appropriate device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.blocks = self.blocks.to(device)
        
        # Set to evaluation mode
        self.eval()
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        
        # Process both images through VGG blocks
        x = predicted
        y = target
        
        for block, weight in zip(self.blocks, self.layer_weights):
            # Extract features
            x = block(x)
            y = block(y)
            
            # Normalize features if requested
            if self.use_normalized_features:
                x = F.normalize(x, dim=1)
                y = F.normalize(y, dim=1)
            
            # Compute feature distance (MSE)
            loss += weight * F.mse_loss(x, y)
        
        return loss
    
    def get_feature_maps(
        self, 
        image: torch.Tensor
    ) -> List[torch.Tensor]:
        features = []
        x = image
        
        for block in self.blocks:
            x = block(x)
            features.append(x.clone())
        
        return features


class MultiScalePerceptualLoss(nn.Module):
    
    def __init__(
        self,
        scales: Optional[List[float]] = None,
        layer_weights: Optional[List[float]] = None
    ):
        super(MultiScalePerceptualLoss, self).__init__()
        
        if scales is None:
            scales = [1.0, 0.5, 0.25]
        
        self.scales = scales
        self.perceptual_loss = PerceptualLoss(layer_weights)
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        
        for scale in self.scales:
            if scale != 1.0:
                # Downsample images for multi-scale processing
                scaled_size = (
                    int(predicted.shape[2] * scale),
                    int(predicted.shape[3] * scale)
                )
                
                scaled_pred = F.interpolate(
                    predicted,
                    size=scaled_size,
                    mode='bilinear',
                    align_corners=False
                )
                
                scaled_target = F.interpolate(
                    target,
                    size=scaled_size,
                    mode='bilinear',
                    align_corners=False
                )
            else:
                scaled_pred = predicted
                scaled_target = target
            
            # Weight smaller scales less
            scale_weight = scale
            loss += scale_weight * self.perceptual_loss(scaled_pred, scaled_target)
        
        # Normalize by sum of scale weights
        total_weight = sum(s for s in self.scales)
        loss = loss / total_weight
        
        return loss