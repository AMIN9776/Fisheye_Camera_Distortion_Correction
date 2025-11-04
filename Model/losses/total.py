import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .perceptual import PerceptualLoss, MultiScalePerceptualLoss
from .color import ImprovedColorLoss, ChromaticAberrationLoss, ColorConsistencyLoss
from .basic import ssim_loss, gradient_loss, EdgeAwareLoss, SmoothL1Loss


class TotalLoss(nn.Module):
    """
    Combined loss function for fisheye rectification training.
    
    This loss combines multiple loss components:
    - Content loss (L1 or Smooth L1)
    - Perceptual loss (VGG-based)
    - SSIM loss (structural similarity)
    - Gradient loss (edge preservation)
    - Color loss (color fidelity)
    
    Args:
        perceptual_weight: Weight for perceptual loss (default: 0.7)
        content_weight: Weight for content loss (default: 1.0)
        ssim_weight: Weight for SSIM loss (default: 0.2)
        grad_weight: Weight for gradient loss (default: 0.4)
        color_weight: Weight for color loss (default: 0.4)
        use_smooth_l1: Use Smooth L1 instead of L1 for content (default: False)
        use_edge_aware: Use edge-aware loss (default: False)
        use_multi_scale_perceptual: Use multi-scale perceptual loss (default: False)
        use_chromatic_aberration: Include chromatic aberration loss (default: False)
    """
    
    def __init__(
        self,
        perceptual_weight: float = 0.7,
        content_weight: float = 1.0,
        ssim_weight: float = 0.2,
        grad_weight: float = 0.4,
        color_weight: float = 0.4,
        use_smooth_l1: bool = False,
        use_edge_aware: bool = False,
        use_multi_scale_perceptual: bool = False,
        use_chromatic_aberration: bool = False
    ):
        super(TotalLoss, self).__init__()
        
        # Store weights
        self.perceptual_weight = perceptual_weight
        self.content_weight = content_weight
        self.ssim_weight = ssim_weight
        self.grad_weight = grad_weight
        self.color_weight = color_weight
        
        # Initialize loss components
        
        # Perceptual loss
        if use_multi_scale_perceptual:
            self.perceptual_loss = MultiScalePerceptualLoss()
        else:
            self.perceptual_loss = PerceptualLoss()
        
        # Content loss
        if use_smooth_l1:
            self.content_loss = SmoothL1Loss(beta=0.1)
        else:
            self.content_loss = nn.L1Loss()
        
        # Color loss
        self.color_loss = ImprovedColorLoss()
        
        # Optional losses
        self.edge_aware_loss = EdgeAwareLoss() if use_edge_aware else None
        self.chromatic_loss = ChromaticAberrationLoss() if use_chromatic_aberration else None
        
        # Track loss history for monitoring
        self.loss_history = {
            'total': [],
            'perceptual': [],
            'content': [],
            'ssim': [],
            'gradient': [],
            'color': []
        }
    
    def forward(
        self, 
        output: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Calculate individual loss components
        loss_components = {}
        
        # Content loss
        content_loss = self.content_loss(output, target)
        loss_components['content'] = content_loss.item()
        
        # Perceptual loss
        perceptual_loss = self.perceptual_loss(output, target)
        loss_components['perceptual'] = perceptual_loss.item()
        
        # SSIM loss
        ssim_loss_val = ssim_loss(output, target)
        loss_components['ssim'] = ssim_loss_val.item()
        
        # Gradient loss
        grad_loss = gradient_loss(output, target)
        loss_components['gradient'] = grad_loss.item()
        
        # Color loss
        color_loss = self.color_loss(output, target)
        loss_components['color'] = color_loss.item()
        
        # Combine losses with weights
        total_loss = (
            self.content_weight * content_loss +
            self.perceptual_weight * perceptual_loss +
            self.ssim_weight * ssim_loss_val +
            self.grad_weight * grad_loss +
            self.color_weight * color_loss
        )
        
        # Add optional losses if enabled
        if self.edge_aware_loss is not None:
            edge_loss = self.edge_aware_loss(output, target)
            total_loss += 0.1 * edge_loss
            loss_components['edge_aware'] = edge_loss.item()
        
        if self.chromatic_loss is not None:
            chromatic_loss = self.chromatic_loss(output, target)
            total_loss += 0.1 * chromatic_loss
            loss_components['chromatic'] = chromatic_loss.item()
        
        loss_components['total'] = total_loss.item()
        
        # Update history
        for key, value in loss_components.items():
            if key in self.loss_history:
                self.loss_history[key].append(value)
        
        return total_loss, loss_components
    
    def get_loss_weights(self) -> Dict[str, float]:
        return {
            'perceptual': self.perceptual_weight,
            'content': self.content_weight,
            'ssim': self.ssim_weight,
            'gradient': self.grad_weight,
            'color': self.color_weight
        }
    
    def update_weights(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown weight parameter '{key}'")
    
    def reset_history(self) -> None:
        """Reset loss history tracking."""
        for key in self.loss_history:
            self.loss_history[key] = []
    
    def get_history_statistics(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        
        for key, values in self.loss_history.items():
            if values:
                tensor_values = torch.tensor(values)
                stats[key] = {
                    'mean': tensor_values.mean().item(),
                    'std': tensor_values.std().item(),
                    'min': tensor_values.min().item(),
                    'max': tensor_values.max().item()
                }
        
        return stats


class AdaptiveTotalLoss(TotalLoss):
    """
    Adaptive total loss with dynamic weight adjustment.
    
    This variant of TotalLoss automatically adjusts weights during training
    based on the relative magnitudes of different loss components, helping
    to balance their contributions.
    
    Args:
        initial_weights: Initial weights for loss components
        adaptation_rate: Rate of weight adaptation (default: 0.01)
        balance_method: Method for balancing ('normalize' or 'gradient')
    """
    
    def __init__(
        self,
        initial_weights: Optional[Dict[str, float]] = None,
        adaptation_rate: float = 0.01,
        balance_method: str = 'normalize'
    ):
        # Set initial weights
        if initial_weights is None:
            initial_weights = {
                'perceptual_weight': 0.7,
                'content_weight': 1.0,
                'ssim_weight': 0.2,
                'grad_weight': 0.4,
                'color_weight': 0.4
            }
        
        super().__init__(**initial_weights)
        
        self.adaptation_rate = adaptation_rate
        self.balance_method = balance_method
        
        # Running averages for loss components
        self.running_means = {
            'perceptual': 1.0,
            'content': 1.0,
            'ssim': 1.0,
            'gradient': 1.0,
            'color': 1.0
        }
    
    def adapt_weights(self, loss_components: Dict[str, float]) -> None:
        if self.balance_method == 'normalize':
            # Update running means
            for key in self.running_means:
                if key in loss_components:
                    self.running_means[key] = (
                        0.99 * self.running_means[key] + 
                        0.01 * loss_components[key]
                    )
            
            # Normalize weights based on running means
            total_mean = sum(self.running_means.values())
            
            if total_mean > 0:
                target_contribution = 1.0 / len(self.running_means)
                
                # Adjust weights to balance contributions
                self.perceptual_weight *= (
                    1 + self.adaptation_rate * 
                    (target_contribution - self.running_means['perceptual'] / total_mean)
                )
                self.content_weight *= (
                    1 + self.adaptation_rate * 
                    (target_contribution - self.running_means['content'] / total_mean)
                )
                self.ssim_weight *= (
                    1 + self.adaptation_rate * 
                    (target_contribution - self.running_means['ssim'] / total_mean)
                )
                self.grad_weight *= (
                    1 + self.adaptation_rate * 
                    (target_contribution - self.running_means['gradient'] / total_mean)
                )
                self.color_weight *= (
                    1 + self.adaptation_rate * 
                    (target_contribution - self.running_means['color'] / total_mean)
                )
    
    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss, loss_components = super().forward(output, target)
        
        # Adapt weights based on current losses
        self.adapt_weights(loss_components)
        
        return total_loss, loss_components