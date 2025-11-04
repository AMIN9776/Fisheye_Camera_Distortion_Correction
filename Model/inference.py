import argparse
import os
import time
from pathlib import Path
from typing import Optional, Tuple, List, Union

from matplotlib.figure import Figure
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import model components
from config import Config
from models import CascadedRectificationModel, EnhancedFisheyeRectificationModel
from utils import tensor_to_image, calculate_psnr, calculate_ssim


class FisheyeRectifier:

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        config_path: Optional[str] = None,
        quick_mode: bool = False
    ):

        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Using device: {self.device}")
        
        # Load checkpoint
        print(f"📁 Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load or create configuration
        if quick_mode:
            # Use minimal config for quick inference
            self.config = self._create_quick_config(checkpoint)
        elif config_path and os.path.exists(config_path):
            self.config = Config(config_path)
        elif 'config' in checkpoint:
            # Load config from checkpoint
            self.config = self._load_config_from_checkpoint(checkpoint['config'])
        else:
            raise ValueError("No configuration found. Provide config_path or use --quick mode")
        
        # Create and load model
        self.model = self._create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully ({self.config.model_type})")
        
        # Setup image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
        ])
    
    def _create_quick_config(self, checkpoint: dict) -> Config:
        """Create a minimal config for quick inference."""
        config = Config()
        
        # Extract essential parameters from checkpoint if available
        if 'config' in checkpoint:
            ckpt_config = checkpoint['config']
            config.model_type = ckpt_config.get('model_type', 'cascaded')
            config.image_size = ckpt_config.get('image_size', 256)
            config.initial_channels = ckpt_config.get('initial_channels', 128)
            config.num_blocks = ckpt_config.get('num_blocks', 5)
            config.growth_rate = ckpt_config.get('growth_rate', 2)
            config.num_skip_connections = ckpt_config.get('num_skip_connections', 4)
        
        return config
    
    def _load_config_from_checkpoint(self, config_dict: dict) -> Config:
        """Load configuration from checkpoint dictionary."""
        config = Config()
        for key, value in config_dict.items():
            setattr(config, key, value)
        return config
    
    def _create_model(self) -> torch.nn.Module:
        """Create the appropriate model based on configuration."""
        if self.config.model_type == 'cascaded':
            return CascadedRectificationModel(self.config)
        else:
            return EnhancedFisheyeRectificationModel(self.config)
    
    @torch.no_grad()
    def rectify_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        resize_back: bool = True
    ) -> np.ndarray:

        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8))
        
        original_size = image.size
        
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        output = self.model(input_tensor)
        
        # Convert to numpy
        rectified = tensor_to_image(output[0])
        
        # Resize back to original size if requested
        if resize_back:
            rectified_pil = Image.fromarray((rectified * 255).astype(np.uint8))
            rectified_pil = rectified_pil.resize(original_size, Image.LANCZOS)
            rectified = np.array(rectified_pil) / 255.0
        
        return rectified
    
    @torch.no_grad()
    def rectify_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 8,
        show_progress: bool = True
    ) -> List[np.ndarray]:
        """
        Rectify multiple images efficiently in batches.
        
        Args:
            images: List of images (paths or PIL Images)
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            List of rectified images as numpy arrays
        """
        rectified_images = []
        
        # Create progress bar if requested
        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Processing images")
        
        for i in iterator:
            batch_images = images[i:i+batch_size]
            batch_tensors = []
            original_sizes = []
            
            # Prepare batch
            for img in batch_images:
                if isinstance(img, (str, Path)):
                    pil_img = Image.open(img).convert('RGB')
                else:
                    pil_img = img
                
                original_sizes.append(pil_img.size)
                tensor = self.transform(pil_img)
                batch_tensors.append(tensor)
            
            # Stack and process
            batch = torch.stack(batch_tensors).to(self.device)
            outputs = self.model(batch)
            
            # Convert outputs
            for j, output in enumerate(outputs):
                rectified = tensor_to_image(output)
                
                # Resize to original
                original_size = original_sizes[j]
                rectified_pil = Image.fromarray((rectified * 255).astype(np.uint8))
                rectified_pil = rectified_pil.resize(original_size, Image.LANCZOS)
                rectified = np.array(rectified_pil) / 255.0
                
                rectified_images.append(rectified)
        
        return rectified_images
    
    def visualize_comparison(
        self,
        image_path: str,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Figure:
        """
        Create a side-by-side comparison visualization.
        
        Args:
            image_path: Path to input fisheye image
            save_path: Optional path to save visualization
            show: Whether to display the figure
            
        Returns:
            Matplotlib figure object
        """
        # Load original
        original = Image.open(image_path).convert('RGB')
        original_np = np.array(original) / 255.0
        
        # Rectify
        rectified = self.rectify_image(image_path)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(original_np)
        ax1.set_title('Original Fisheye', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2.imshow(rectified)
        ax2.set_title('Rectified Image', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        plt.suptitle('Fisheye Rectification Result', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Visualization saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig


def process_single_image(
    rectifier: FisheyeRectifier,
    input_path: Path,
    output_dir: Path,
    visualize: bool = False
) -> None:
    """Process a single image."""
    print(f"\n Processing: {input_path.name}")
    
    # Measure time
    start_time = time.time()
    
    # Rectify image
    rectified = rectifier.rectify_image(str(input_path))
    
    # Save rectified image
    output_path = output_dir / f"{input_path.stem}_rectified{input_path.suffix}"
    rectified_img = Image.fromarray((rectified * 255).astype(np.uint8))
    rectified_img.save(output_path)
    
    elapsed = time.time() - start_time
    print(f"Saved to: {output_path.name} ({elapsed:.2f}s)")
    
    # Create visualization if requested
    if visualize:
        vis_path = output_dir / f"{input_path.stem}_comparison.png"
        rectifier.visualize_comparison(str(input_path), str(vis_path), show=False)


def process_directory(
    rectifier: FisheyeRectifier,
    input_dir: Path,
    output_dir: Path,
    batch_size: int = 8,
    visualize: bool = False
) -> None:
    """Process all images in a directory."""
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = [p for p in input_dir.iterdir() 
                   if p.suffix.lower() in image_extensions]
    
    if not image_paths:
        print(" No images found in directory")
        return
    
    print(f"\nFound {len(image_paths)} images")
    
    # Batch process
    start_time = time.time()
    rectified_images = rectifier.rectify_batch(
        [str(p) for p in image_paths],
        batch_size=batch_size,
        show_progress=True
    )
    
    # Save results
    print("\n Saving results...")
    for img_path, rectified in zip(image_paths, rectified_images):
        output_path = output_dir / f"{img_path.stem}_rectified.png"
        rectified_img = Image.fromarray((rectified * 255).astype(np.uint8))
        rectified_img.save(output_path)
        
        # Create visualization for first few images
        if visualize and image_paths.index(img_path) < 3:
            vis_path = output_dir / f"{img_path.stem}_comparison.png"
            original = np.array(Image.open(img_path)) / 255.0
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ax1.imshow(original)
            ax1.set_title('Original')
            ax1.axis('off')
            ax2.imshow(rectified)
            ax2.set_title('Rectified')
            ax2.axis('off')
            plt.tight_layout()
            plt.savefig(vis_path, dpi=100)
            plt.close()
    
    elapsed = time.time() - start_time
    print(f"\n Processed {len(image_paths)} images in {elapsed:.1f}s")
    print(f"Results saved to: {output_dir}")


def main():
    """Main entry point for inference script."""
    
    parser = argparse.ArgumentParser(
        description='Fisheye Image Rectification - Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s model.pth image.jpg                    # Quick inference on single image
  %(prog)s model.pth input_dir/ --batch_size 16   # Batch process directory
  %(prog)s model.pth image.jpg --visualize        # With comparison visualization
        """
    )
    
    parser.add_argument('checkpoint', type=str,
                       help='Path to trained model checkpoint')
    parser.add_argument('input', type=str,
                       help='Input image or directory')
    parser.add_argument('--output', type=str, default='./inference_output',
                       help='Output directory (default: ./inference_output)')
    parser.add_argument('--config', type=str,
                       help='Config file (uses checkpoint config if not provided)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for directory processing (default: 8)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create comparison visualizations')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='Device to use (auto-detect if not specified)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with minimal configuration')
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "="*50)
    print("Fisheye Rectification - Inference Mode")
    print("="*50)
    
    # Setup paths
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize rectifier
    print("\nLoading model...")
    rectifier = FisheyeRectifier(
        checkpoint_path=args.checkpoint,
        device=args.device,
        config_path=args.config,
        quick_mode=args.quick
    )
    
    # Process based on input type
    if input_path.is_file():
        process_single_image(rectifier, input_path, output_dir, args.visualize)
    elif input_path.is_dir():
        process_directory(rectifier, input_path, output_dir, 
                         args.batch_size, args.visualize)
    else:
        print(f"Error: {input_path} not found")
        return 1
    
    print("\n" + "="*50)
    print("✨ Inference complete!")
    print("="*50 + "\n")
    
    return 0


if __name__ == "__main__":
    exit(main())