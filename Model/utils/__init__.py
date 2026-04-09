"""
Utility functions for Fisheye Rectification Model.

This module provides utilities for:
- Visualization and plotting
- Metrics computation
- General helper functions
"""

# Import visualization utilities
from .visualization import (
    plot_training_history,
    visualize_batch_results,
    visualize_model_comparison,
    plot_loss_components_breakdown,
    create_before_after_grid,
    tensor_to_image,
    save_image_tensor
)

# Import metrics utilities
from .metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_lpips,
    calculate_fid,
    calculate_mae,
    calculate_rmse,
    calculate_gradient_similarity,
    calculate_color_accuracy,
    evaluate_batch,
    MetricsTracker
)


__all__ = [
    # Visualization
    'plot_training_history',
    'visualize_batch_results',
    'visualize_model_comparison',
    'plot_loss_components_breakdown',
    'create_before_after_grid',
    'tensor_to_image',
    'save_image_tensor',
    
    # Metrics
    'calculate_psnr',
    'calculate_ssim',
    'calculate_lpips',
    'calculate_fid',
    'calculate_mae',
    'calculate_rmse',
    'calculate_gradient_similarity',
    'calculate_color_accuracy',
    'evaluate_batch',
    'MetricsTracker'
]