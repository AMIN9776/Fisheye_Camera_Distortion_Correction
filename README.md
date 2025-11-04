# Fisheye_Camera_Distortion_Correction# Fisheye Camera Distortion Correction

A deep learning approach for rectifying fisheye camera distortion using cascaded neural networks with deformable convolutions and attention mechanisms.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Model Architectures](#model-architectures)
- [Configuration](#configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a state-of-the-art deep learning solution for correcting geometric distortions in fisheye camera images. The system uses advanced neural network architectures combining deformable convolutions, attention mechanisms, and multi-scale processing to achieve high-quality rectification while preserving image details and color fidelity.

### Key Innovations

- **Cascaded Architecture**: Two-stage processing with coarse geometric correction followed by fine-grained refinement
- **Deformable Convolutions**: Adaptive spatial sampling to handle varying distortion patterns
- **Color-Aware Attention**: Special attention to green channel for better color preservation (important for Bayer sensors)
- **Gated Skip Connections**: Selective feature propagation between encoder and decoder
- **Multi-Component Loss**: Combines perceptual, structural, gradient, and color losses for optimal results

## Features

- **Multiple Model Architectures**: Cascaded and Enhanced models for different use cases
- **Comprehensive Loss Functions**: Perceptual (VGG), SSIM, gradient, and color-aware losses
- **Flexible Training Pipeline**: Support for custom datasets, checkpointing, and early stopping
- **Efficient Inference**: Batch processing, GPU acceleration, and memory optimization
- **Extensive Preprocessing**: Synthetic fisheye generation, padding, and augmentation tools
- **Professional Code Structure**: Modular design with clear separation of concerns
- **Comprehensive Evaluation**: Multiple metrics including PSNR, SSIM, LPIPS, and FID

## Architecture

The project offers two main model architectures:

### 1. Cascaded Rectification Model
- **Stage 1**: Coarse geometric correction using displacement field learning
- **Stage 2**: Fine-grained refinement with enhanced feature extraction
- **Advantages**: Faster training, lower memory usage, progressive refinement

### 2. Enhanced Rectification Model  
- **Single-stage** processing with advanced attention mechanisms
- **Deformable convolutions** for adaptive feature extraction
- **Color-aware attention** for better color preservation
- **Advantages**: Higher quality output, better detail preservation

## Installation

### Requirements

- Python 3.8 or higher
- CUDA 11.0+ (for GPU support)
- 8GB+ GPU memory recommended

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fisheye-rectification.git
cd fisheye-rectification
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify installation:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Project Structure

```
.
├── Model/
│   ├── config/                 # Configuration management
│   │   ├── config.py          # Configuration class
│   │   └── config.yaml        # Default configuration
│   ├── data/                  # Data loading utilities
│   │   └── dataloader.py      # Batch image pair loader
│   ├── models/                # Neural network architectures
│   │   ├── layers.py          # Custom layers (deformable conv, attention)
│   │   └── networks.py        # Main network architectures
│   ├── losses/                # Loss functions
│   │   ├── perceptual.py      # VGG-based perceptual loss
│   │   ├── color.py           # Color-aware losses
│   │   ├── basic.py           # SSIM and gradient losses
│   │   └── total.py           # Combined loss function
│   ├── training/              # Training utilities
│   │   ├── trainer.py         # Main trainer class
│   │   └── callbacks.py       # Training callbacks
│   ├── utils/                 # Utility functions
│   │   ├── visualization.py   # Plotting and visualization
│   │   ├── metrics.py         # Evaluation metrics
│   │   └── helpers.py         # General utilities
│   ├── main.py                # Main entry point
│   ├── train_simple.py        # Simple training script
│   ├── train_example.py       # Example training script
│   └── inference.py           # Inference script
├── PreProcess/                # Data preprocessing tools
│   ├── Adding_distortion.py   # Synthetic fisheye generation
│   ├── AddPadding.py          # Black border padding
│   ├── crop_FD_black_margine.py  # Crop black margins
│   ├── create_dataset.py      # Dataset creation utilities
│   └── ValSplit.py            # Train/validation splitting
└── requirements.txt           # Python dependencies
```

## Data Preparation

### Dataset Structure

Organize your dataset in the following structure:
```
data/
├── train/
│   ├── fisheye/       # Distorted training images
│   └── rectified/     # Ground truth rectified images
└── val/
    ├── fisheye/       # Distorted validation images
    └── rectified/     # Ground truth rectified images
```

### Preprocessing Pipeline

#### 1. Generate Synthetic Fisheye Images

If you only have normal images, create synthetic fisheye distortions:

```bash
cd PreProcess
python Adding_distortion.py
```

Configuration in `Adding_distortion.py`:
- `downsamplex`, `downsampley`: Output image dimensions
- `input_folder`: Source undistorted images
- `output_folder`: Synthetic fisheye outputs
- `original_folder`: Resized originals

The script applies a 4-term polynomial distortion model:
```
r_distorted = k1*r + k2*r^3 + k3*r^5 + k4*r^7
```

#### 2. Handle Black Borders

Remove or pad black borders in fisheye images:

```bash
python AddPadding.py
```

This script:
- Detects black borders using threshold detection
- Fills borders by sampling from deeper image regions
- Uses erosion to ensure source pixels are valid

#### 3. Crop Black Margins

Remove black margins and resize:

```bash
python crop_FD_black_margine.py
```

#### 4. Create Dataset from Multiple Sources

Combine images from multiple directories:

```bash
python create_dataset.py
```

Parameters:
- `N`: Total number of images to select
- `M`: Number of source subfolders
- `seed`: Random seed for reproducible selection

#### 5. Split Data for Training/Validation

```bash
python ValSplit.py
```

Creates train/validation splits while maintaining paired relationships.

## Training

### Quick Start

#### Option 1: Simple Training (Beginners)

```bash
python train_simple.py
```

Modify constants in the script for basic configuration:
- `BATCH_SIZE`: Training batch size (default: 4)
- `IMAGE_SIZE`: Image dimensions (default: 256)
- `NUM_EPOCHS`: Training epochs (default: 100)

#### Option 2: Standard Training

```bash
python train_example.py \
    --data_dir ./data \
    --epochs 200 \
    --batch_size 8 \
    --model cascaded \
    --lr 0.001
```

#### Option 3: Advanced Training

```bash
python main.py train \
    --config config/config.yaml \
    --experiment_name fisheye_experiment \
    --epochs 300 \
    --tensorboard \
    --backup_code
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_dir` | Dataset directory | `./data` |
| `--epochs` | Number of training epochs | 100 |
| `--batch_size` | Batch size | 8 |
| `--lr` | Learning rate | 0.001 |
| `--model` | Model type (`cascaded`/`enhanced`) | `cascaded` |
| `--checkpoint` | Resume from checkpoint | None |
| `--device` | Device (`cuda`/`cpu`) | Auto-detect |

### Resume Training

```bash
python train_example.py \
    --checkpoint checkpoints/latest_checkpoint.pth \
    --epochs 100
```

### Monitor Training

Training automatically saves:
- `checkpoints/best_model.pth`: Best model based on validation loss
- `checkpoints/latest_checkpoint.pth`: Most recent checkpoint
- `results/training_history.png`: Loss curves
- `results/metrics.csv`: Detailed metrics per epoch

## Inference

### Single Image Inference

```bash
python inference.py checkpoints/best_model.pth input.jpg --output results/
```

### Batch Processing

```bash
python inference.py checkpoints/best_model.pth input_dir/ \
    --output results/ \
    --batch_size 16
```

### With Visualization

```bash
python inference.py checkpoints/best_model.pth input.jpg \
    --visualize \
    --output results/
```

### Quick Mode (No Configuration File)

```bash
python inference.py checkpoints/best_model.pth input.jpg --quick
```

### Python API

```python
from inference import FisheyeRectifier
import numpy as np
from PIL import Image

# Initialize rectifier
rectifier = FisheyeRectifier('checkpoints/best_model.pth', quick_mode=True)

# Single image
rectified = rectifier.rectify_image('fisheye.jpg')

# Batch processing
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
rectified_batch = rectifier.rectify_batch(image_paths, batch_size=8)

# Save results
for i, img in enumerate(rectified_batch):
    Image.fromarray((img * 255).astype(np.uint8)).save(f'result_{i}.jpg')
```

## Model Architectures

### Cascaded Rectification Model

```python
Stage 1: Coarse Rectification
- Input: Fisheye image (B, 3, H, W)
- Downsampling: Progressive feature extraction
- Output: Displacement field for initial correction

Stage 2: Enhanced Refinement
- Input: Coarsely rectified image
- Processing: Deformable convolutions + Attention
- Output: Final rectified image
```

### Enhanced Fisheye Rectification Model

```python
Components:
- Encoder: 5 blocks with deformable convolutions
- Decoder: 4 blocks with skip connections
- Attention: Border attention + Color-aware attention
- Skip Connections: Gated for selective feature propagation
```

### Custom Layers

1. **DeformableConv2d**: Learns spatial offsets for adaptive sampling
2. **ColorAwareAttention**: Enhanced attention for green channel
3. **GatedSkipConnection**: Selective feature propagation
4. **ResidualDeformableBlock**: Residual connections with deformable convolutions

## Configuration

### Configuration File (config.yaml)

```yaml
model:
  model_type: 'cascaded'        # 'cascaded' or 'enhanced'
  initial_channels: 128          # Starting channel count
  num_blocks: 5                  # Number of encoder/decoder blocks
  growth_rate: 2                 # Channel growth factor
  image_size: 256               # Training image size
  num_skip_connections: 4        # Skip connections count

training:
  batch_size: 8
  learning_rate: 0.001
  num_epochs: 200
  early_stopping_patience: 20
  scheduler_step_size: 30
  scheduler_gamma: 0.5

loss:
  perceptual_weight: 0.7        # VGG perceptual loss
  content_weight: 1.0           # L1 content loss
  ssim_weight: 0.2             # Structural similarity
  grad_weight: 0.4             # Gradient loss
  color_weight: 0.4            # Color preservation

paths:
  checkpoint_dir: './checkpoints'
  results_dir: './results'
  train_fisheye_dir: './data/train/fisheye'
  train_rectified_dir: './data/train/rectified'
  val_fisheye_dir: './data/val/fisheye'
  val_rectified_dir: './data/val/rectified'
```

### Loss Function Weights

Adjust weights to emphasize different aspects:

| Weight | Purpose | Recommended Range |
|--------|---------|-------------------|
| `perceptual_weight` | Overall visual quality | 0.5 - 1.0 |
| `content_weight` | Pixel-level accuracy | 0.8 - 1.2 |
| `ssim_weight` | Structural preservation | 0.1 - 0.3 |
| `grad_weight` | Edge sharpness | 0.2 - 0.5 |
| `color_weight` | Color fidelity | 0.3 - 0.5 |

## Evaluation Metrics

The system computes multiple metrics for comprehensive evaluation:

### Image Quality Metrics

1. **PSNR (Peak Signal-to-Noise Ratio)**
   - Measures pixel-level reconstruction quality
   - Higher is better (typical range: 25-40 dB)

2. **SSIM (Structural Similarity Index)**
   - Measures structural preservation
   - Range: 0-1 (higher is better)

3. **LPIPS (Learned Perceptual Image Patch Similarity)**
   - Perceptual distance using deep features
   - Lower is better

4. **FID (Fréchet Inception Distance)**
   - Measures distribution similarity
   - Lower is better

### Running Evaluation

```bash
python main.py evaluate \
    --checkpoint checkpoints/best_model.pth \
    --val_fisheye_dir data/val/fisheye \
    --val_rectified_dir data/val/rectified \
    --compute_lpips \
    --save_visualizations
```

## Results

### Expected Performance

| Model | PSNR | SSIM | LPIPS | FID | Training Time |
|-------|------|------|-------|-----|---------------|
| Cascaded | 31.32 dB | 0.91 | 0.044 | 13.69 | ~4 hours |
| Enhanced | 32.79 dB | 0.90 | 0.052 | 25.28 | ~6 hours |

*Results on validation set with 256x256 images, trained for 200 epochs on NVIDIA RTX 3090*

### Sample Results

The model successfully corrects various types of fisheye distortions:
- Barrel distortion
- Pincushion effects
- Complex radial distortions
- Color aberrations

## Technical Details

### Deformable Convolution Implementation

The deformable convolution learns offsets and modulation masks:

```python
offset = self.offset_conv(x)           # 2D offsets for each kernel position
modulator = 2 * sigmoid(self.modulator_conv(x))  # Importance weights
output = deform_conv2d(x, offset, weight, mask=modulator)
```

### Color-Aware Attention Mechanism

Special attention to green channel (important for Bayer pattern sensors):

```python
green_attention = self.green_attention(x)
other_attention = self.other_attention(x)
# Boost middle channels by 20%
attention[:, mid_channels, :, :] *= 1.2
final_attention = (green_attention * attention + other_attention) / 2
```

### Polynomial Distortion Model

Synthetic fisheye generation uses a 4-term polynomial:

```python
r_distorted = k1*r + k2*r^3 + k3*r^5 + k4*r^7
```

Where:
- `r`: Original radius from image center
- `k1, k2, k3, k4`: Distortion coefficients
- `r_distorted`: Distorted radius

## Troubleshooting

### Common Issues and Solutions

#### Out of Memory Error

**Problem**: CUDA out of memory during training

**Solutions**:
- Reduce batch size: `--batch_size 2`
- Reduce image size: `--image_size 128`
- Use gradient accumulation
- Switch to cascaded model (uses less memory)

#### Slow Training

**Problem**: Training is taking too long

**Solutions**:
- Verify GPU usage: `nvidia-smi`
- Enable mixed precision training
- Reduce validation frequency
- Use cascaded model for faster training

#### Poor Convergence

**Problem**: Loss not decreasing or unstable

**Solutions**:
- Reduce learning rate: `--lr 0.0001`
- Check data normalization
- Verify paired images are correctly matched
- Increase batch size for more stable gradients

#### Black Borders in Output

**Problem**: Rectified images have black borders

**Solutions**:
- Preprocess with `AddPadding.py`
- Adjust padding parameters in preprocessing
- Check input image quality

### System Requirements

**Minimum Requirements**:
- 8GB RAM
- 4GB GPU memory
- 50GB disk space

**Recommended Requirements**:
- 16GB RAM
- 8GB+ GPU memory (RTX 3070 or better)
- 100GB disk space
- CUDA 11.0+

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Add type hints for function parameters
- Include docstrings for all classes and functions
- Write unit tests for new features

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- VGG network for perceptual loss from torchvision
- Deformable convolution implementation from torchvision.ops
- SSIM implementation from pytorch-msssim
- Dataset augmentation inspired by standard fisheye correction literature

## Citation

If you use this code in your research, please cite:

```bibtex
@software{fisheye_rectification,
  title={Fisheye Camera Distortion Correction},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/fisheye-rectification}
}
```

## Contact

For questions or support, please open an issue on GitHub or contact [your-email@example.com]