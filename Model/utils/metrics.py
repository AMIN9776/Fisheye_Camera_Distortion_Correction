import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional, Tuple, List, Dict
from scipy import signal
import warnings


def calculate_psnr(
    img1: Union[torch.Tensor, np.ndarray],
    img2: Union[torch.Tensor, np.ndarray],
    max_val: float = 1.0
) -> float:
    # Convert to numpy if needed
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Calculate MSE
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    
    return float(psnr)


def calculate_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0
) -> float:
    try:
        from pytorch_msssim import ssim
        return float(ssim(img1, img2, data_range=data_range, size_average=True).item())
    except ImportError:
        # Fallback to custom implementation
        return float(basic_ssim(img1, img2, window_size, sigma, data_range))


def basic_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0
) -> torch.Tensor:

    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    
    # Create Gaussian window
    window = create_gaussian_window(window_size, img1.size(1), sigma).to(img1.device)
    
    # Calculate means
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2
    
    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()


def create_gaussian_window(window_size: int, channel: int, sigma: float) -> torch.Tensor:
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([
            np.exp(-(x - window_size//2)**2 / float(2*sigma**2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    
    return window


def calculate_lpips(
    img1: torch.Tensor,
    img2: torch.Tensor,
    net: str = 'alex',
    device: Optional[str] = None
) -> float:
    try:
        import lpips
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize LPIPS model
        lpips_fn = lpips.LPIPS(net=net).to(device)
        lpips_fn.eval()
        
        # Move tensors to device
        img1 = img1.to(device)
        img2 = img2.to(device)
        
        # Calculate LPIPS
        with torch.no_grad():
            distance = lpips_fn(img1, img2)
        
        return float(distance.mean().item())
        
    except ImportError:
        warnings.warn("LPIPS not available. Install with: pip install lpips")
        return 0.0


def calculate_fid(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    batch_size: int = 50,
    device: Optional[str] = None
) -> float:
    try:
        from pytorch_fid import fid_score
        
        # This is a simplified version - normally you'd save images to disk
        # and use fid_score.calculate_fid_given_paths
        warnings.warn("FID calculation requires saving images to disk. Using placeholder.")
        return 0.0
        
    except ImportError:
        warnings.warn("pytorch-fid not available. Install with: pip install pytorch-fid")
        return 0.0


def calculate_mae(
    img1: Union[torch.Tensor, np.ndarray],
    img2: Union[torch.Tensor, np.ndarray]
) -> float:
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    return float(np.mean(np.abs(img1 - img2)))


def calculate_rmse(
    img1: Union[torch.Tensor, np.ndarray],
    img2: Union[torch.Tensor, np.ndarray]
) -> float:
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    return float(np.sqrt(np.mean((img1 - img2) ** 2)))


def calculate_gradient_similarity(
    img1: torch.Tensor,
    img2: torch.Tensor
) -> float:
    # Sobel filters
    sobel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=torch.float32).view(1, 1, 3, 3).to(img1.device)
    
    sobel_y = torch.tensor([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=torch.float32).view(1, 1, 3, 3).to(img1.device)
    
    # Expand for all channels
    channels = img1.shape[1]
    sobel_x = sobel_x.repeat(channels, 1, 1, 1)
    sobel_y = sobel_y.repeat(channels, 1, 1, 1)
    
    # Calculate gradients
    grad1_x = F.conv2d(img1, sobel_x, padding=1, groups=channels)
    grad1_y = F.conv2d(img1, sobel_y, padding=1, groups=channels)
    grad1_mag = torch.sqrt(grad1_x**2 + grad1_y**2 + 1e-6)
    
    grad2_x = F.conv2d(img2, sobel_x, padding=1, groups=channels)
    grad2_y = F.conv2d(img2, sobel_y, padding=1, groups=channels)
    grad2_mag = torch.sqrt(grad2_x**2 + grad2_y**2 + 1e-6)
    
    # Calculate similarity
    similarity = 1.0 - F.l1_loss(grad1_mag, grad2_mag)
    
    return float(similarity.item())


def calculate_color_accuracy(
    img1: torch.Tensor,
    img2: torch.Tensor
) -> Dict[str, float]:
    # Calculate per-channel metrics
    metrics = {}
    
    # Mean color difference per channel
    for i, channel in enumerate(['red', 'green', 'blue']):
        channel_diff = torch.abs(img1[:, i] - img2[:, i]).mean()
        metrics[f'{channel}_mae'] = float(channel_diff.item())
    
    # Color distribution metrics
    img1_mean = img1.mean(dim=[2, 3])  # (B, C)
    img2_mean = img2.mean(dim=[2, 3])  # (B, C)
    
    img1_std = img1.std(dim=[2, 3])
    img2_std = img2.std(dim=[2, 3])
    
    # Mean color difference
    metrics['mean_color_diff'] = float(F.l1_loss(img1_mean, img2_mean).item())
    
    # Std color difference
    metrics['std_color_diff'] = float(F.l1_loss(img1_std, img2_std).item())
    
    # Color correlation
    for i, channel in enumerate(['red', 'green', 'blue']):
        ch1 = img1[:, i].flatten()
        ch2 = img2[:, i].flatten()
        
        # Normalize
        ch1 = (ch1 - ch1.mean()) / (ch1.std() + 1e-6)
        ch2 = (ch2 - ch2.mean()) / (ch2.std() + 1e-6)
        
        correlation = (ch1 * ch2).mean()
        metrics[f'{channel}_correlation'] = float(correlation.item())
    
    return metrics


def evaluate_batch(
    output: torch.Tensor,
    target: torch.Tensor,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    if metrics is None:
        metrics = ['psnr', 'ssim', 'mae', 'rmse']
    
    results = {}
    
    # Ensure tensors are in [0, 1] range
    output = torch.clamp(output, 0, 1)
    target = torch.clamp(target, 0, 1)
    
    # Calculate requested metrics
    if 'psnr' in metrics:
        psnr_values = []
        for i in range(output.shape[0]):
            psnr = calculate_psnr(
                output[i].cpu().numpy().transpose(1, 2, 0),
                target[i].cpu().numpy().transpose(1, 2, 0)
            )
            psnr_values.append(psnr)
        results['psnr'] = np.mean(psnr_values)
        results['psnr_std'] = np.std(psnr_values)
    
    if 'ssim' in metrics:
        results['ssim'] = calculate_ssim(output, target)
    
    if 'mae' in metrics:
        results['mae'] = calculate_mae(output, target)
    
    if 'rmse' in metrics:
        results['rmse'] = calculate_rmse(output, target)
    
    if 'lpips' in metrics:
        results['lpips'] = calculate_lpips(output, target)
    
    if 'gradient' in metrics:
        results['gradient_sim'] = calculate_gradient_similarity(output, target)
    
    if 'color' in metrics and output.shape[1] == 3:
        color_metrics = calculate_color_accuracy(output, target)
        results.update(color_metrics)
    
    return results


class MetricsTracker:
    
    def __init__(self, metrics_to_track: Optional[List[str]] = None):
        if metrics_to_track is None:
            metrics_to_track = ['psnr', 'ssim', 'mae']
        
        self.metrics_to_track = metrics_to_track
        self.history = {metric: [] for metric in metrics_to_track}
        self.best = {metric: None for metric in metrics_to_track}
        self.best_epoch = {metric: None for metric in metrics_to_track}
    
    def update(self, metrics: Dict[str, float], epoch: Optional[int] = None) -> None:
        for metric in self.metrics_to_track:
            if metric in metrics:
                value = metrics[metric]
                self.history[metric].append(value)
                
                # Update best value
                if self.best[metric] is None or self._is_better(metric, value, self.best[metric]):
                    self.best[metric] = value
                    self.best_epoch[metric] = epoch
    
    def _is_better(self, metric: str, new_val: float, old_val: float) -> bool:
        # Higher is better for these metrics
        higher_better = ['psnr', 'ssim', 'gradient_sim']
        
        if metric in higher_better:
            return new_val > old_val
        else:
            return new_val < old_val
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all tracked metrics.
        
        Returns:
            Dictionary with statistics for each metric
        """
        summary = {}
        
        for metric in self.metrics_to_track:
            if self.history[metric]:
                values = np.array(self.history[metric])
                summary[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'best': self.best[metric],
                    'best_epoch': self.best_epoch[metric],
                    'latest': float(values[-1])
                }
        
        return summary
    
    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.history = {metric: [] for metric in self.metrics_to_track}
        self.best = {metric: None for metric in self.metrics_to_track}
        self.best_epoch = {metric: None for metric in self.metrics_to_track}