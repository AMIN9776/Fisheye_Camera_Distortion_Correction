import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
try:
    from pytorch_msssim import ssim, ms_ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    print("Warning: pytorch-msssim not found. SSIM loss will use basic implementation.")


def ssim_loss(
    output: torch.Tensor, 
    target: torch.Tensor,
    data_range: float = 1.0,
    size_average: bool = True
) -> torch.Tensor:
    """
    Structural Similarity Index (SSIM) loss.
    
    SSIM loss = 1 - SSIM, so perfect similarity gives 0 loss.
    
    Args:
        output: Predicted image tensor of shape (B, C, H, W)
        target: Target image tensor of shape (B, C, H, W)
        data_range: Range of the image data (default: 1.0)
        size_average: Whether to average the loss (default: True)
        
    Returns:
        SSIM loss value
    """
    if SSIM_AVAILABLE:
        # Use pytorch-msssim if available
        # Scale from [-1, 1] to [0, 1] if needed
        output_scaled = (output + 1) / 2 if output.min() < 0 else output
        target_scaled = (target + 1) / 2 if target.min() < 0 else target
        
        ssim_value = ssim(
            output_scaled, 
            target_scaled, 
            data_range=data_range, 
            size_average=size_average
        )
        return 1 - ssim_value
    else:
        # Basic SSIM implementation
        return basic_ssim_loss(output, target)


def basic_ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5
) -> torch.Tensor:
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Create Gaussian window
    window = _create_gaussian_window(window_size, pred.size(1), sigma).to(pred.device)
    
    # Compute means
    mu1 = F.conv2d(pred, window, padding=window_size//2, groups=pred.size(1))
    mu2 = F.conv2d(target, window, padding=window_size//2, groups=target.size(1))
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=pred.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=target.size(1)) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=pred.size(1)) - mu1_mu2
    
    # Compute SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return 1 - ssim_map.mean()


def _create_gaussian_window(window_size: int, channel: int, sigma: float) -> torch.Tensor:
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size//2)**2 / (2.0 * sigma**2)))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    
    return window


def gradient_loss(
    output: torch.Tensor, 
    target: torch.Tensor,
    edge_order: int = 1
) -> torch.Tensor:
    if edge_order == 1:
        return first_order_gradient_loss(output, target)
    elif edge_order == 2:
        return second_order_gradient_loss(output, target)
    else:
        raise ValueError(f"Unsupported edge_order: {edge_order}")


def first_order_gradient_loss(
    output: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    # Define Sobel kernels
    sobel_x = torch.tensor([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ], dtype=output.dtype, device=output.device).view(1, 1, 3, 3)
    
    sobel_y = torch.tensor([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ], dtype=output.dtype, device=output.device).view(1, 1, 3, 3)
    
    # Expand kernels for all channels
    channels = output.shape[1]
    sobel_x = sobel_x.repeat(channels, 1, 1, 1)
    sobel_y = sobel_y.repeat(channels, 1, 1, 1)
    
    # Compute gradients
    grad_out_x = F.conv2d(output, sobel_x, padding=1, groups=channels)
    grad_out_y = F.conv2d(output, sobel_y, padding=1, groups=channels)
    
    grad_tar_x = F.conv2d(target, sobel_x, padding=1, groups=channels)
    grad_tar_y = F.conv2d(target, sobel_y, padding=1, groups=channels)
    
    # Compute L1 loss on gradients
    loss = F.l1_loss(grad_out_x, grad_tar_x) + F.l1_loss(grad_out_y, grad_tar_y)
    
    return loss


def second_order_gradient_loss(
    output: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    # Define Laplacian kernel
    laplacian = torch.tensor([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=output.dtype, device=output.device).view(1, 1, 3, 3)
    
    # Expand kernel for all channels
    channels = output.shape[1]
    laplacian = laplacian.repeat(channels, 1, 1, 1)
    
    # Compute Laplacian
    lap_out = F.conv2d(output, laplacian, padding=1, groups=channels)
    lap_tar = F.conv2d(target, laplacian, padding=1, groups=channels)
    
    # Compute L1 loss
    loss = F.l1_loss(lap_out, lap_tar)
    
    return loss


class EdgeAwareLoss(nn.Module):
    """
    Edge-aware loss that focuses more on edge regions.
    
    This loss applies stronger penalties to errors in edge regions,
    which are particularly important for maintaining image sharpness.
    
    Args:
        edge_threshold: Threshold for edge detection (default: 0.1)
        edge_weight: Weight multiplier for edge regions (default: 2.0)
    """
    
    def __init__(
        self,
        edge_threshold: float = 0.1,
        edge_weight: float = 2.0
    ):
        super(EdgeAwareLoss, self).__init__()
        
        self.edge_threshold = edge_threshold
        self.edge_weight = edge_weight
        
        # Sobel kernels for edge detection
        self.register_buffer('sobel_x', torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.tensor([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Convert to grayscale for edge detection
        target_gray = target.mean(dim=1, keepdim=True)
        
        # Detect edges in target
        edge_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(target_gray, self.sobel_y, padding=1)
        edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        
        # Create edge mask
        edge_mask = (edge_magnitude > self.edge_threshold).float()
        
        # Apply different weights to edge and non-edge regions
        weights = 1.0 + edge_mask * (self.edge_weight - 1.0)
        
        # Compute weighted L1 loss
        diff = torch.abs(output - target)
        weighted_diff = diff * weights
        
        return weighted_diff.mean()


class SmoothL1Loss(nn.Module):
    
    def __init__(self, beta: float = 1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta)
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.smooth_l1(output, target)