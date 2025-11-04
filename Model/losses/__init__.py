
# Import perceptual losses
from .perceptual import (
    PerceptualLoss,
    MultiScalePerceptualLoss
)

# Import color losses
from .color import (
    ImprovedColorLoss,
    ChromaticAberrationLoss,
    ColorConsistencyLoss
)

# Import basic losses
from .basic import (
    ssim_loss,
    gradient_loss,
    first_order_gradient_loss,
    second_order_gradient_loss,
    EdgeAwareLoss,
    SmoothL1Loss
)

# Import total loss
from .total import (
    TotalLoss,
    AdaptiveTotalLoss
)

__all__ = [
    # Perceptual
    'PerceptualLoss',
    'MultiScalePerceptualLoss',
    
    # Color
    'ImprovedColorLoss',
    'ChromaticAberrationLoss',
    'ColorConsistencyLoss',
    
    # Basic
    'ssim_loss',
    'gradient_loss',
    'first_order_gradient_loss',
    'second_order_gradient_loss',
    'EdgeAwareLoss',
    'SmoothL1Loss',
    
    # Total
    'TotalLoss',
    'AdaptiveTotalLoss',
]