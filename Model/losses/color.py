import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ImprovedColorLoss(nn.Module):
    """
    Color loss with enhanced weight for green channel.
    
    This loss function specifically weights the green channel higher than
    red and blue channels, reflecting the importance of green in both
    sensor design (Bayer pattern) and human perception.
    
    Args:
        green_weight: Weight multiplier for green channel (default: 1.5)
        use_correlation: Whether to include correlation loss (default: True)
        correlation_weight: Weight for correlation component (default: 0.1)
    """
    
    def __init__(
        self,
        green_weight: float = 1.5,
        use_correlation: bool = True,
        correlation_weight: float = 0.1
    ):
        super(ImprovedColorLoss, self).__init__()
        
        self.green_weight = green_weight
        self.use_correlation = use_correlation
        self.correlation_weight = correlation_weight
        
        # Create channel weights tensor
        self.register_buffer(
            'channel_weights',
            torch.tensor([1.0, green_weight, 1.0], dtype=torch.float32)
        )
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Split into color channels
        output_r, output_g, output_b = torch.split(output, 1, dim=1)
        target_r, target_g, target_b = torch.split(target, 1, dim=1)
        
        # Compute channel-wise differences
        diff_r = torch.abs(output_r - target_r)
        diff_g = torch.abs(output_g - target_g)
        diff_b = torch.abs(output_b - target_b)
        
        # Stack differences
        channel_diffs = torch.cat([diff_r, diff_g, diff_b], dim=1)
        
        # Apply channel weights
        weights = self.channel_weights.view(1, 3, 1, 1)
        weighted_diff = channel_diffs * weights
        
        # Compute mean loss
        loss = weighted_diff.mean()
        
        # Add correlation loss if enabled
        if self.use_correlation:
            # Focus on green channel correlation
            g_corr_loss = self._correlation_loss(output_g, target_g)
            loss = loss + self.correlation_weight * g_corr_loss
        
        return loss
    
    def _correlation_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Flatten spatial dimensions
        x_flat = x.view(x.size(0), -1)
        y_flat = y.view(y.size(0), -1)
        
        # Center the variables
        x_centered = x_flat - x_flat.mean(dim=1, keepdim=True)
        y_centered = y_flat - y_flat.mean(dim=1, keepdim=True)
        
        # Compute correlation
        numerator = (x_centered * y_centered).sum(dim=1)
        denominator = torch.sqrt(
            (x_centered ** 2).sum(dim=1) * (y_centered ** 2).sum(dim=1)
        )
        
        correlation = numerator / (denominator + 1e-6)
        
        # Return 1 - correlation (so perfect correlation gives 0 loss)
        return (1 - correlation).mean()


class ChromaticAberrationLoss(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super(ChromaticAberrationLoss, self).__init__()
        
        self.kernel_size = kernel_size
        
        # Create Sobel kernels for edge detection
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
        loss = 0.0
        
        # Process each color channel
        for i in range(3):
            # Extract channel
            output_ch = output[:, i:i+1, :, :]
            target_ch = target[:, i:i+1, :, :]
            
            # Compute gradients
            output_grad_x = F.conv2d(output_ch, self.sobel_x, padding=1)
            output_grad_y = F.conv2d(output_ch, self.sobel_y, padding=1)
            
            target_grad_x = F.conv2d(target_ch, self.sobel_x, padding=1)
            target_grad_y = F.conv2d(target_ch, self.sobel_y, padding=1)
            
            # Compute gradient magnitude
            output_grad_mag = torch.sqrt(output_grad_x**2 + output_grad_y**2 + 1e-6)
            target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-6)
            
            # Add to loss
            loss += F.l1_loss(output_grad_mag, target_grad_mag)
        
        # Compare gradient consistency across channels
        for i in range(2):
            for j in range(i+1, 3):
                # Get channel gradients
                ch_i = output[:, i:i+1, :, :]
                ch_j = output[:, j:j+1, :, :]
                
                grad_i_x = F.conv2d(ch_i, self.sobel_x, padding=1)
                grad_i_y = F.conv2d(ch_i, self.sobel_y, padding=1)
                
                grad_j_x = F.conv2d(ch_j, self.sobel_x, padding=1)
                grad_j_y = F.conv2d(ch_j, self.sobel_y, padding=1)
                
                # Penalize gradient differences between channels
                loss += 0.1 * (
                    F.l1_loss(grad_i_x, grad_j_x) + 
                    F.l1_loss(grad_i_y, grad_j_y)
                )
        
        return loss


class ColorConsistencyLoss(nn.Module):
    """
    Loss for maintaining color consistency across the image.
    
    This loss helps preserve color relationships and prevents color shifts
    during rectification.
    
    Args:
        patch_size: Size of patches for local color statistics (default: 16)
    """
    
    def __init__(self, patch_size: int = 16):
        super(ColorConsistencyLoss, self).__init__()
        self.patch_size = patch_size
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute color consistency loss.
        
        Args:
            output: Predicted image tensor of shape (B, 3, H, W)
            target: Target image tensor of shape (B, 3, H, W)
            
        Returns:
            Color consistency loss value
        """
        # Compute global color statistics
        output_mean = output.mean(dim=[2, 3], keepdim=True)
        output_std = output.std(dim=[2, 3], keepdim=True)
        
        target_mean = target.mean(dim=[2, 3], keepdim=True)
        target_std = target.std(dim=[2, 3], keepdim=True)
        
        # Global statistics loss
        global_loss = (
            F.l1_loss(output_mean, target_mean) +
            F.l1_loss(output_std, target_std)
        )
        
        # Local patch statistics
        local_loss = self._compute_local_statistics_loss(output, target)
        
        return global_loss + 0.5 * local_loss
    
    def _compute_local_statistics_loss(
        self, 
        output: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        B, C, H, W = output.shape
        
        # Use average pooling to compute local statistics
        kernel_size = min(self.patch_size, H, W)
        
        # Local means
        output_local_mean = F.avg_pool2d(output, kernel_size, stride=kernel_size//2)
        target_local_mean = F.avg_pool2d(target, kernel_size, stride=kernel_size//2)
        
        # Local variance (using E[X^2] - E[X]^2)
        output_sq = F.avg_pool2d(output ** 2, kernel_size, stride=kernel_size//2)
        target_sq = F.avg_pool2d(target ** 2, kernel_size, stride=kernel_size//2)
        
        output_local_var = output_sq - output_local_mean ** 2
        target_local_var = target_sq - target_local_mean ** 2
        
        # Compute losses
        mean_loss = F.l1_loss(output_local_mean, target_local_mean)
        var_loss = F.l1_loss(output_local_var, target_local_var)
        
        return mean_loss + var_loss