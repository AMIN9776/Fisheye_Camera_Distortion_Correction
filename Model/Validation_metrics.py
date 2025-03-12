import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import matplotlib.pyplot as plt
from datetime import datetime
import os
import yaml
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
import argparse

# Import your existing modules
from datasets import BatchImagePairLoader

# Built-in PSNR function to avoid dependency on scikit-image
def psnr_func(gt, pred):
    """Custom PSNR implementation"""
    mse = np.mean((gt - pred) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Optional dependencies with fallbacks
try:
    from pytorch_msssim import ssim
    SSIM_AVAILABLE = True
except ImportError:
    print("Warning: pytorch-msssim not found. SSIM metrics will be unavailable.")
    SSIM_AVAILABLE = False

try:
    from pytorch_fid import fid_score
    FID_AVAILABLE = True
except ImportError:
    print("Warning: pytorch-fid not found. FID metrics will be unavailable.")
    FID_AVAILABLE = False

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    print("Warning: lpips not found. LPIPS metrics will be unavailable.")
    LPIPS_AVAILABLE = False

# Including the Config class from your original code
default_config = """
model:
  initial_channels: 128
  num_blocks: 5
  growth_rate: 2
  image_size: 256
  num_skip_connections: 4
  load_checkpoint: True
  checkpoint_path: '/home/amin/Projects/Amin/Fisheye/NewModel/checkpoints_larger_continue_same_method/best_model.pth'
  reset_optimizer: True
  model_type: 'cascaded'  # can be 'cascaded' or 'enhanced'

training:
  batch_size: 1
  learning_rate: 0.00045
  num_epochs: 500
  plot_interval: 200
  validation_split: 0.02
  scheduler_step_size: 5
  scheduler_gamma: 0.9
  early_stopping_patience: 100
  start_epoch: 0

loss:
  perceptual_weight: 0.7 #0.6
  content_weight: 1.0
  ssim_weight: 0.2
  grad_weight: 0.4
  color_weight: 0.4 

paths:
  checkpoint_dir: './checkpoints_newModel2_enhanced2_color_continue1'
  data_dir: '/home/amin/Projects/Amin/Fisheye/PreProcess'
  train_fisheye_dir: '/home/amin/Projects/Amin/Fisheye/PreProcess/crop_distorted_pad_train_1024'
  train_rectified_dir: '/home/amin/Projects/Amin/Fisheye/PreProcess/rectified_train_1024'
  val_fisheye_dir: '/home/amin/Projects/Amin/Fisheye/PreProcess/Main_dataset/crop_distorted_pad_val_1024'
  val_rectified_dir: '/home/amin/Projects/Amin/Fisheye/PreProcess/Main_dataset/rectified_val_1024'
  results_dir: './results2_enhanced2_color_continue1'
  output_dir: './validation_results_continue_same_method'
"""

class Config:
    def __init__(self, config_path=None):
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = yaml.safe_load(default_config)
        
        # Model parameters
        self.initial_channels = config['model']['initial_channels']
        self.num_blocks = config['model']['num_blocks']
        self.growth_rate = config['model']['growth_rate']
        self.image_size = config['model']['image_size']
        self.num_skip_connections = config['model'].get('num_skip_connections', config['model']['num_blocks'])
        self.model_type = config['model'].get('model_type', 'cascaded')
        
        # Loading parameters
        self.load_checkpoint = config['model'].get('load_checkpoint', False)
        self.checkpoint_path = config['model'].get('checkpoint_path', None)
        self.reset_optimizer = config['model'].get('reset_optimizer', False)
        self.start_epoch = config['training'].get('start_epoch', 0)
        
        # Training parameters
        self.batch_size = config['training']['batch_size']
        self.learning_rate = config['training']['learning_rate']
        self.num_epochs = config['training']['num_epochs']
        self.plot_interval = config['training']['plot_interval']
        self.scheduler_step_size = config['training']['scheduler_step_size']
        self.scheduler_gamma = config['training']['scheduler_gamma']
        self.early_stopping_patience = config['training']['early_stopping_patience']
        
        # Loss weights
        self.perceptual_weight = config['loss']['perceptual_weight']
        self.content_weight = config['loss']['content_weight']
        self.ssim_weight = config['loss'].get('ssim_weight', 0.1)
        self.grad_weight = config['loss'].get('grad_weight', 0.1)
        self.color_weight = config['loss'].get('color_weight', 0.5)
        
        # Paths
        self.checkpoint_dir = config['paths']['checkpoint_dir']
        self.data_dir = config['paths']['data_dir']
        self.train_fisheye_dir = config['paths']['train_fisheye_dir']
        self.train_rectified_dir = config['paths']['train_rectified_dir']
        self.val_fisheye_dir = config['paths']['val_fisheye_dir']
        self.val_rectified_dir = config['paths']['val_rectified_dir']
        self.results_dir = config['paths']['results_dir']
        self.output_dir = config['paths'].get('output_dir', './validation_results')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

# Validation script configuration
# You can modify these values directly
VALIDATION_CONFIG = {
    "compute_psnr": True,
    "compute_ssim": True and SSIM_AVAILABLE,
    "compute_fid": True and FID_AVAILABLE,
    "compute_lpips": True and LPIPS_AVAILABLE,
    "visualize_results": True,
    "num_visualization_samples": 10,
    "use_cuda": True,
    "batch_size_override": None  # If set, will override the config's batch_size
}

def denormalize(tensor):
    """Denormalize tensor from [-1, 1] to [0, 1]"""
    return (tensor).clamp(0, 1)
    #return (tensor*0.75 + 0.1).clamp(0, 1)

def tensor_to_numpy(tensor):
    """Convert tensor to numpy for visualization/saving"""
    img = denormalize(tensor).cpu().numpy().transpose(1, 2, 0)
    return img

def save_image(tensor, path):
    """Save tensor as image"""
    img_np = tensor_to_numpy(tensor)
    img_np = (img_np*255.0).astype(np.uint8)
    Image.fromarray(img_np).save(path)

def compute_psnr(img1, img2):
    """Compute PSNR between two images"""
    return psnr_func(img1, img2)

def compute_ssim(img1_tensor, img2_tensor):
    """Compute SSIM between two tensor images"""
    if not SSIM_AVAILABLE:
        return 0.0
    return ssim(img1_tensor.unsqueeze(0), img2_tensor.unsqueeze(0), data_range=1.0).item()

def compute_lpips(img1_tensor, img2_tensor, lpips_fn):
    """Compute LPIPS between two tensor images"""
    if not LPIPS_AVAILABLE:
        return 0.0
    return lpips_fn(img1_tensor.unsqueeze(0), img2_tensor.unsqueeze(0)).item()

def create_directories(base_dir):
    """Create output directories"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, f'results')
    
    generated_dir = os.path.join(results_dir, 'generated')
    input_dir = os.path.join(results_dir, 'input')
    gt_dir = os.path.join(results_dir, 'ground_truth')
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    
    return {
        'results': results_dir,
        'generated': generated_dir,
        'input': input_dir,
        'ground_truth': gt_dir
    }

def save_config(config, validation_config, results_dir):
    """Save configuration to a file"""
    # Combine configurations for saving
    combined_config = {
        'model_config': {k: v for k, v in config.__dict__.items() if not k.startswith('__')},
        'validation_config': validation_config
    }
    
    with open(os.path.join(results_dir, 'validation_config.json'), 'w') as f:
        json.dump(combined_config, f, indent=4)

def visualize_results(dirs, metrics, num_samples=10):
    """Create visualization of input, ground truth and generated images"""
    print("Generating visualization...")
    
    # Get all filenames
    all_files = sorted(os.listdir(dirs['generated']))
    
    # Select samples for visualization
    if num_samples < len(all_files):
        # Take evenly spaced samples
        indices = np.linspace(0, len(all_files)-1, num_samples, dtype=int)
        selected_files = [all_files[i] for i in indices]
    else:
        selected_files = all_files
    
    # Create figure
    fig, axes = plt.subplots(len(selected_files), 3, figsize=(15, 5*len(selected_files)))
    if len(selected_files) == 1:
        axes = axes.reshape(1, 3)
    
    for i, filename in enumerate(selected_files):
        # Get indices in the metrics arrays
        file_idx = all_files.index(filename)
        
        # Load images
        input_img = np.array(Image.open(os.path.join(dirs['input'], filename))) 
        gt_img = np.array(Image.open(os.path.join(dirs['ground_truth'], filename))) 
        gen_img = np.array(Image.open(os.path.join(dirs['generated'], filename))) 
        
        # Get metrics for this image
        metrics_text = ""
        if 'psnr_values' in metrics and len(metrics['psnr_values']) > file_idx:
            metrics_text += f"PSNR: {metrics['psnr_values'][file_idx]:.2f}  "
        if 'ssim_values' in metrics and len(metrics['ssim_values']) > file_idx:
            metrics_text += f"SSIM: {metrics['ssim_values'][file_idx]:.2f}  "
        
        # Display images
        axes[i, 0].imshow(input_img)
        axes[i, 0].set_title("Input (Fisheye)")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gt_img)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(gen_img)
        axes[i, 2].set_title(f"Generated\n{metrics_text}")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(dirs['results'], 'comparison.png'), dpi=300)
    plt.close()
    
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
def main():
    # Create a minimal argparse to override configuration
    parser = argparse.ArgumentParser(description='Fisheye Image Rectification Validation')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, help='Override checkpoint path')
    parser.add_argument('--val_fisheye_dir', type=str, help='Override validation fisheye directory')
    parser.add_argument('--val_rectified_dir', type=str, help='Override validation rectified directory')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--model_type', type=str, choices=['cascaded', 'enhanced'], help='Override model type')
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Override configuration with command line arguments
    if args.checkpoint:
        config.checkpoint_path = args.checkpoint
    if args.val_fisheye_dir:
        config.val_fisheye_dir = args.val_fisheye_dir
    if args.val_rectified_dir:
        config.val_rectified_dir = args.val_rectified_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.batch_size:
        VALIDATION_CONFIG["batch_size_override"] = args.batch_size
    if args.model_type:
        config.model_type = args.model_type
    
    # Determine batch size for validation
    batch_size = VALIDATION_CONFIG["batch_size_override"] if VALIDATION_CONFIG["batch_size_override"] else config.batch_size
    
    # Create output directories
    dirs = create_directories(config.output_dir)
    
    # Save configuration
    save_config(config, VALIDATION_CONFIG, dirs['results'])
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and VALIDATION_CONFIG["use_cuda"] else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading checkpoint from {config.checkpoint_path}")
    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    
    # Import your model definitions
    from main_newModel_enhanced_color import CascadedRectificationModel
    from main_newModel_enhanced_color import EnhancedFisheyeRectificationModel
    
    # Load the appropriate model type
    if config.model_type == "cascaded":
        model = CascadedRectificationModel(config).to(device)
        print("Using CascadedRectificationModel")
    else:  # "enhanced"
        model = EnhancedFisheyeRectificationModel(config).to(device)
        print("Using EnhancedFisheyeRectificationModel")
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Setup LPIPS if needed
    lpips_fn = None
    if VALIDATION_CONFIG["compute_lpips"] and LPIPS_AVAILABLE:
        lpips_fn = lpips.LPIPS(net='alex').to(device)
    
    # Create dataset using your BatchImagePairLoader
    dataset = BatchImagePairLoader(
        config.val_fisheye_dir,
        config.val_rectified_dir,
        config.image_size,
        config.image_size,
        batch_size,
        shuffle=False
    )
    
    # Initialize metrics storage
    metrics = {
        'psnr_values': [],
        'ssim_values': [],
        'lpips_values': []
    }
    
    # Process validation data
    print(f"Running validation on {len(dataset)} batches")
    with torch.no_grad():
        for batch_idx in tqdm(range(len(dataset)), desc="Generating rectified images"):
            # Get batch
            fisheye_batch, rectified_batch = dataset[batch_idx]
            fisheye_batch = fisheye_batch.to(device)
            rectified_batch = rectified_batch.to(device)
            
            # Generate rectified images
            output_batch = model(fisheye_batch)
            
            # Calculate metrics and save images for each item in batch
            for i in range(fisheye_batch.size(0)):
                # Extract single images
                fisheye = fisheye_batch[i]
                rectified = rectified_batch[i]
                output = output_batch[i]
                
                # Generate filename based on batch and position
                filename = f"img_batch{batch_idx:03d}_item{i:03d}.png"
                
                # Save images
                save_image(fisheye, os.path.join(dirs['input'], filename))
                save_image(rectified, os.path.join(dirs['ground_truth'], filename))
                save_image(output, os.path.join(dirs['generated'], filename))
                
                # Calculate metrics
                if VALIDATION_CONFIG["compute_psnr"]:
                    output_np = tensor_to_numpy(output)
                    target_np = tensor_to_numpy(rectified)
                    psnr_value = compute_psnr(target_np, output_np)
                    metrics['psnr_values'].append(psnr_value)
                
                if VALIDATION_CONFIG["compute_ssim"] and SSIM_AVAILABLE:
                    output_norm = denormalize(output)
                    target_norm = denormalize(rectified)
                    ssim_value = compute_ssim(output_norm, target_norm)
                    metrics['ssim_values'].append(ssim_value)
                
                if VALIDATION_CONFIG["compute_lpips"] and LPIPS_AVAILABLE:
                    lpips_value = compute_lpips(output, rectified, lpips_fn)
                    metrics['lpips_values'].append(lpips_value)
    
    # Calculate average metrics
    results = {}
    
    if VALIDATION_CONFIG["compute_psnr"]:
        results['PSNR'] = np.mean(metrics['psnr_values'])
    
    if VALIDATION_CONFIG["compute_ssim"] and SSIM_AVAILABLE:
        results['SSIM'] = np.mean(metrics['ssim_values'])
    
    if VALIDATION_CONFIG["compute_lpips"] and LPIPS_AVAILABLE:
        results['LPIPS'] = np.mean(metrics['lpips_values'])
    
    # Compute FID if enabled
    if VALIDATION_CONFIG["compute_fid"] and FID_AVAILABLE:
        print("Computing FID score...")
        fid = fid_score.calculate_fid_given_paths(
            [dirs['generated'], dirs['ground_truth']],
            batch_size=min(batch_size, 50),
            device=device,
            dims=2048
        )
        results['FID'] = fid
    
    # Visualize results if enabled
    if VALIDATION_CONFIG["visualize_results"]:
        visualize_results(
            dirs, 
            metrics, 
            num_samples=VALIDATION_CONFIG["num_visualization_samples"]
        )
    
    # Save metrics to file
    with open(os.path.join(dirs['results'], 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    
    # Print metrics
    print("\nValidation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    print(f"\nValidation completed. Results saved to {dirs['results']}")

if __name__ == "__main__":
    main()