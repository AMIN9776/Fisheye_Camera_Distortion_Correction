import os
from typing import List, Optional, Dict, Tuple, Union
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
import seaborn as sns
from PIL import Image

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    loss_components: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    # Create figure with subplots
    if loss_components:
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Main loss plot (larger)
        ax_main = fig.add_subplot(gs[0:2, :])
    else:
        fig, ax_main = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot main losses
    epochs = range(1, len(train_losses) + 1)
    ax_main.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2.5)
    ax_main.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2.5)
    ax_main.set_xlabel('Epoch', fontsize=12)
    ax_main.set_ylabel('Loss', fontsize=12)
    ax_main.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax_main.legend(loc='upper right', fontsize=11)
    ax_main.grid(True, alpha=0.3)
    
    # Mark best validation loss
    best_val_idx = np.argmin(val_losses)
    ax_main.plot(best_val_idx + 1, val_losses[best_val_idx], 'r*', 
                markersize=15, label=f'Best Val: {val_losses[best_val_idx]:.4f}')
    
    # Plot component losses if provided
    if loss_components:
        components = ['perceptual', 'content', 'ssim', 'gradient', 'color']
        positions = [(2, 0), (2, 1), (2, 2)]
        
        for idx, comp in enumerate(components[:3]):
            if comp in loss_components.get('train', {}):
                ax = fig.add_subplot(gs[positions[idx][0], positions[idx][1]])
                
                train_comp = loss_components['train'].get(comp, [])
                val_comp = loss_components['val'].get(comp, [])
                
                if train_comp:
                    ax.plot(epochs[:len(train_comp)], train_comp, 
                           label=f'Train', linewidth=1.5)
                    ax.plot(epochs[:len(val_comp)], val_comp, 
                           label=f'Val', linewidth=1.5)
                    ax.set_xlabel('Epoch', fontsize=10)
                    ax.set_ylabel('Loss', fontsize=10)
                    ax.set_title(f'{comp.capitalize()} Loss', fontsize=11)
                    ax.legend(loc='upper right', fontsize=9)
                    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Training History Overview', fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def visualize_batch_results(
    fisheye_batch: torch.Tensor,
    target_batch: torch.Tensor,
    output_batch: torch.Tensor,
    num_samples: int = 4,
    save_path: Optional[str] = None,
    show: bool = True,
    titles: Optional[List[str]] = None
) -> Figure:
    batch_size = min(fisheye_batch.shape[0], num_samples)
    
    fig, axes = plt.subplots(3, batch_size, figsize=(4 * batch_size, 12))
    
    if batch_size == 1:
        axes = axes.reshape(3, 1)
    
    row_titles = ['Input (Fisheye)', 'Ground Truth', 'Model Output']
    
    for i in range(batch_size):
        # Convert tensors to displayable format
        img_fisheye = tensor_to_image(fisheye_batch[i])
        img_target = tensor_to_image(target_batch[i])
        img_output = tensor_to_image(output_batch[i])
        
        # Display images
        axes[0, i].imshow(img_fisheye)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel(row_titles[0], fontsize=12, fontweight='bold')
        if titles and i < len(titles):
            axes[0, i].set_title(titles[i], fontsize=10)
        
        axes[1, i].imshow(img_target)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel(row_titles[1], fontsize=12, fontweight='bold')
        
        axes[2, i].imshow(img_output)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel(row_titles[2], fontsize=12, fontweight='bold')
        
        # Add PSNR/SSIM metrics if possible
        try:
            from ..utils.metrics import calculate_psnr, calculate_ssim
            psnr = calculate_psnr(img_output, img_target)
            ssim = calculate_ssim(output_batch[i:i+1], target_batch[i:i+1])
            axes[2, i].set_xlabel(f'PSNR: {psnr:.2f}\nSSIM: {ssim:.3f}', fontsize=9)
        except:
            pass
    
    plt.suptitle('Batch Results Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Batch visualization saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def visualize_model_comparison(
    fisheye: torch.Tensor,
    outputs: Dict[str, torch.Tensor],
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Compare outputs from different models or checkpoints.
    
    Args:
        fisheye: Input fisheye image tensor (C, H, W)
        outputs: Dictionary mapping model names to output tensors
        save_path: Path to save the figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    num_models = len(outputs)
    fig, axes = plt.subplots(1, num_models + 1, figsize=(5 * (num_models + 1), 5))
    
    # Display input
    img_fisheye = tensor_to_image(fisheye)
    axes[0].imshow(img_fisheye)
    axes[0].set_title('Input (Fisheye)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Display outputs from different models
    for idx, (model_name, output) in enumerate(outputs.items(), 1):
        img_output = tensor_to_image(output)
        axes[idx].imshow(img_output)
        axes[idx].set_title(model_name, fontsize=12)
        axes[idx].axis('off')
    
    plt.suptitle('Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Model comparison saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_loss_components_breakdown(
    loss_components: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Prepare data
    components = ['perceptual', 'content', 'ssim', 'gradient', 'color']
    epochs = range(1, len(loss_components['train'][components[0]]) + 1)
    
    # Training loss breakdown
    train_data = []
    for comp in components:
        if comp in loss_components['train']:
            train_data.append(loss_components['train'][comp])
    
    ax1.stackplot(epochs, *train_data, labels=components, alpha=0.7)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss Contribution', fontsize=11)
    ax1.set_title('Training Loss Components Breakdown', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Validation loss breakdown
    val_data = []
    for comp in components:
        if comp in loss_components['val']:
            val_data.append(loss_components['val'][comp])
    
    ax2.stackplot(epochs, *val_data, labels=components, alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss Contribution', fontsize=11)
    ax2.set_title('Validation Loss Components Breakdown', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Loss Components Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Loss breakdown plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_before_after_grid(
    input_dir: str,
    output_dir: str,
    num_samples: int = 9,
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    # Get matching files
    input_files = sorted(Path(input_dir).glob('*.jpg'))[:num_samples]
    
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size * 2, figsize=(4 * grid_size, 2 * grid_size))
    axes = axes.flatten()
    
    for idx, input_file in enumerate(input_files):
        output_file = Path(output_dir) / f"{input_file.stem}_rectified.jpg"
        
        if output_file.exists():
            # Load images
            input_img = Image.open(input_file)
            output_img = Image.open(output_file)
            
            # Display before
            axes[idx * 2].imshow(input_img)
            axes[idx * 2].set_title('Before', fontsize=9)
            axes[idx * 2].axis('off')
            
            # Display after
            axes[idx * 2 + 1].imshow(output_img)
            axes[idx * 2 + 1].set_title('After', fontsize=9)
            axes[idx * 2 + 1].axis('off')
    
    # Hide unused subplots
    for idx in range(len(input_files) * 2, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Rectification Results Grid', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Results grid saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    # Handle different tensor formats
    if isinstance(tensor, torch.Tensor):
        img = tensor.detach().cpu().numpy()
    else:
        img = tensor
    
    # Handle channel positioning
    if img.shape[0] in [1, 3]:  # (C, H, W)
        img = np.transpose(img, (1, 2, 0))
    
    # Remove single channel dimension if grayscale
    if img.shape[2] == 1:
        img = img.squeeze(2)
    
    # Denormalize if needed
    if img.min() < 0:
        img = (img + 1) / 2  # From [-1, 1] to [0, 1]
    
    # Clip to valid range
    img = np.clip(img, 0, 1)
    
    return img


def save_image_tensor(
    tensor: torch.Tensor,
    path: str,
    denormalize: bool = True
) -> None:
    """
    Save a tensor as an image file.
    
    Args:
        tensor: Image tensor to save
        path: Path to save the image
        denormalize: Whether to denormalize the tensor
    """
    img = tensor_to_image(tensor)
    
    # Convert to uint8
    img = (img * 255).astype(np.uint8)
    
    # Handle grayscale
    if len(img.shape) == 2:
        img = Image.fromarray(img, mode='L')
    else:
        img = Image.fromarray(img, mode='RGB')
    
    img.save(path)
    print(f"Image saved to {path}")