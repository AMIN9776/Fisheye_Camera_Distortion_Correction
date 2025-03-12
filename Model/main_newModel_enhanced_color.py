import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import matplotlib.pyplot as plt
from datetime import datetime
import os
import yaml
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import torchvision.ops
from tqdm import tqdm  
from pytorch_msssim import ssim

default_config = """
model:
  initial_channels: 128
  num_blocks: 5
  growth_rate: 2
  image_size: 256
  num_skip_connections: 4
  load_checkpoint: True
  checkpoint_path: '/home/amin/Projects/Amin/Fisheye/NewModel/checkpoints_larger_continue_same_method/best_model.pth'
  reset_optimizer: False

training:
  batch_size: 15
  learning_rate: 0.0003
  num_epochs: 500
  plot_interval: 655
  validation_split: 0.02
  scheduler_step_size: 5
  scheduler_gamma: 0.9
  early_stopping_patience: 100
  start_epoch: 0

loss:
  perceptual_weight: 0.7 #0.6
  content_weight: 1.0
  ssim_weight: 0.3 #0.2
  grad_weight: 0.4
  color_weight: 0.4 

paths:
  checkpoint_dir: './checkpoints_larger_continue_same_method'
  data_dir: '/home/amin/Projects/Amin/Fisheye/PreProcess'
  train_fisheye_dir: '/home/amin/Projects/Amin/Fisheye/PreProcess/Main_dataset/crop_distorted_pad_train_1024'
  train_rectified_dir: '/home/amin/Projects/Amin/Fisheye/PreProcess/Main_dataset/rectified_train_1024'
  val_fisheye_dir: '/home/amin/Projects/Amin/Fisheye/PreProcess/Main_dataset/crop_distorted_pad_val_1024'
  val_rectified_dir: '/home/amin/Projects/Amin/Fisheye/PreProcess/Main_dataset/rectified_val_1024'
  
  
  results_dir: './results_larger_continue_same_method'
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
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        bias=True)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      bias=bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation)
        return x

class ColorAwareAttention(nn.Module):
    def __init__(self, channels):
        super(ColorAwareAttention, self).__init__()
        self.channels = channels
        
        # Attention for green channel
        self.green_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid()
        )
        
        # Attention for other channels
        self.other_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Apply attention to entire feature map
        green_attention = self.green_attention(x)
        other_attention = self.other_attention(x)
        
        # Combine attentions with a focus on middle channels (assuming RGB-like structure)
        n_channels = x.size(1)
        mid_point = n_channels // 2
        
        # Create a weighted attention mask
        attention = torch.ones_like(x)
        attention[:, mid_point, :, :] = attention[:, mid_point, :, :] * 1.2  # Boost middle channels
        
        # Combine both attentions
        final_attention = (green_attention * attention + other_attention * (1 - attention)) / 2
        
        return x * final_attention

# Modify your EnhancedFisheyeRectificationModel class
class EnhancedFisheyeRectificationModel(nn.Module):
    def __init__(self, config):
        super(EnhancedFisheyeRectificationModel, self).__init__()
        self.image_size = config.image_size
        channels = [config.initial_channels * (config.growth_rate ** i) for i in range(config.num_blocks)]
        
        # Enhanced encoder blocks with color attention
        self.encoder_blocks = nn.ModuleList([
            self._make_encoder_block(3 if i == 0 else channels[i - 1], channels[i])
            for i in range(config.num_blocks)
        ])
        
        self.num_skip_connections = config.num_skip_connections
        self.gated_skips = nn.ModuleList([
            GatedSkipConnection(ch) for ch in reversed(channels[-self.num_skip_connections:])
        ])
        
        # Enhanced decoder blocks with color attention
        self.decoder_blocks = nn.ModuleList([
            self._make_decoder_block(
                encoder_channel=channels[-(i + 1)],
                out_channels=channels[-(i + 2)] if (i + 1) < self.num_skip_connections else 3
            )
            for i in range(self.num_skip_connections)
        ])
        
        self.border_attention = self._make_attention_block()
        self.color_refinement = self._make_color_refinement_block()

    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            DeformableConv2d(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            ColorAwareAttention(out_channels),
            DeformableConv2d(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def _make_decoder_block(self, encoder_channel, out_channels):
        return nn.Sequential(
            nn.Conv2d(2 * encoder_channel, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            ColorAwareAttention(out_channels)
        )

    def _make_attention_block(self):
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def _make_color_refinement_block(self):
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            ColorAwareAttention(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        attention = self.border_attention(x)
        x = x * attention
        
        features = []
        for block in self.encoder_blocks:
            x = block(x)
            features.append(x)
            x = F.max_pool2d(x, 2)
            
        skip_features = features[-self.num_skip_connections:]
        
        for i in range(self.num_skip_connections):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            skip_feature = skip_features[-(i + 1)]
            skip_feature = self.gated_skips[i](skip_feature)
            x = torch.cat([x, skip_feature], dim=1)
            x = self.decoder_blocks[i](x)
            
        x = F.interpolate(x, size=(self.image_size, self.image_size), 
                         mode='bilinear', align_corners=False)
        return self.color_refinement(x)
    
    
class ImprovedColorLoss(nn.Module):
    def __init__(self):
        super(ImprovedColorLoss, self).__init__()
        
    def forward(self, output, target):
        # Split into channels
        output_r, output_g, output_b = torch.split(output, 1, dim=1)
        target_r, target_g, target_b = torch.split(target, 1, dim=1)
        
        # Higher weight for green channel
        green_weight = 1.5
        rgb_weights = torch.tensor([1.0, green_weight, 1.0], device=output.device).view(1, 3, 1, 1)
        
        # Calculate channel-wise differences
        diff_r = torch.abs(output_r - target_r)
        diff_g = torch.abs(output_g - target_g)
        diff_b = torch.abs(output_b - target_b)
        
        # Combine differences with weights
        weighted_diff = torch.cat([diff_r, diff_g, diff_b], dim=1) * rgb_weights
        
        # Add correlation loss for green channel
        g_correlation = self._correlation_loss(output_g, target_g)
        
        return weighted_diff.mean() + 0.0 * g_correlation
        
    def _correlation_loss(self, x, y):
        # Normalize the inputs
        x = x - x.mean()
        y = y - y.mean()
        
        # Calculate correlation coefficient
        numerator = (x * y).sum()
        denominator = torch.sqrt((x ** 2).sum() * (y ** 2).sum())
        correlation = numerator / (denominator + 1e-6)
        
        return 1 - correlation

# Update the TotalLoss class with color-aware loss
class TotalLoss(nn.Module):
    def __init__(self, perceptual_weight=0.7, content_weight=1.0, 
                 ssim_weight=0.2, grad_weight=0.4, color_weight=0.4):
        super(TotalLoss, self).__init__()
        
        # Initialize all loss components
        self.perceptual_loss = PerceptualLoss()
        self.content_loss = nn.L1Loss()
        self.improved_color_loss = ImprovedColorLoss()
        
        # Store weights for each loss component
        self.perceptual_weight = perceptual_weight
        self.content_weight = content_weight
        self.ssim_weight = ssim_weight
        self.grad_weight = grad_weight
        self.color_weight = color_weight

    def forward(self, output, target):
        # Calculate individual loss components
        p_loss = self.perceptual_loss(output, target)
        c_loss = self.content_loss(output, target)
        s_loss = ssim_loss(output, target)
        g_loss = gradient_loss(output, target)
        col_loss = self.improved_color_loss(output, target)
        
        # Combine all losses with their respective weights
        total_loss = (
            self.content_weight * c_loss +
            self.perceptual_weight * p_loss +
            self.ssim_weight * s_loss +
            self.grad_weight * g_loss +
            self.color_weight * col_loss
        )
        
        # Optional: Create a dictionary of individual losses for monitoring
        loss_components = {
            'perceptual': p_loss.item(),
            'content': c_loss.item(),
            'ssim': s_loss.item(),
            'gradient': g_loss.item(),
            'color': col_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_components


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features.eval()
        self.blocks = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:16], # relu3_3
            vgg[16:23] # relu4_3
        ])
        for bl in self.blocks:
            for p in bl.parameters():
                p.requires_grad = False
        if torch.cuda.is_available():
            self.blocks = self.blocks.cuda()

    def forward(self, x, y):
        loss = 0
        weights = [1.0, 1.0, 1.0, 1.0]
        for block, weight in zip(self.blocks, weights):
            x = block(x)
            y = block(y)
            loss += weight * F.mse_loss(x, y)
        return loss

class GatedSkipConnection(nn.Module):
    def __init__(self, in_channels):
        super(GatedSkipConnection, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        gate = self.conv(x)
        return x * gate

class CoarseRectificationNet(nn.Module):
    def __init__(self, in_channels=3, feature_channels=64, downsample_factor=16):
        super(CoarseRectificationNet, self).__init__()
        self.downsample_factor = downsample_factor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels * 2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(feature_channels * 2, feature_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_channels * 4, feature_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_channels * 4, 2, kernel_size=3, stride=1, padding=1)
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        B, C, H, W = x.shape
        disp_coarse = self.conv(x)
        disp = F.interpolate(disp_coarse, size=(H, W), mode='bilinear', align_corners=True)
        disp = self.tanh(disp) * 0.1
        grid = self._create_grid(B, H, W, x.device)
        grid = grid + disp.permute(0, 2, 3, 1)
        warped = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return warped

    def _create_grid(self, B, H, W, device):
        xs = torch.linspace(-1, 1, W, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        return grid

# class EnhancedFisheyeRectificationModel(nn.Module):
#     def __init__(self, config):
#         super(EnhancedFisheyeRectificationModel, self).__init__()
#         self.image_size = config.image_size
#         channels = [config.initial_channels * (config.growth_rate ** i) for i in range(config.num_blocks)]
#         self.encoder_blocks = nn.ModuleList([
#             self._make_encoder_block(3 if i == 0 else channels[i - 1], channels[i])
#             for i in range(config.num_blocks)
#         ])
#         self.num_skip_connections = config.num_skip_connections
#         self.gated_skips = nn.ModuleList(
#             [GatedSkipConnection(ch) for ch in reversed(channels[-self.num_skip_connections:])]
#         )
#         self.decoder_blocks = nn.ModuleList([
#             self._make_decoder_block(
#                 encoder_channel=channels[-(i + 1)],
#                 out_channels=channels[-(i + 2)] if (i + 1) < self.num_skip_connections else 3
#             )
#             for i in range(self.num_skip_connections)
#         ])
#         self.border_attention = self._make_attention_block()
#         self.refinement = self._make_refinement_block()

#     def _make_encoder_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             DeformableConv2d(in_channels, out_channels),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             DeformableConv2d(out_channels, out_channels),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         )

#     def _make_decoder_block(self, encoder_channel, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(2 * encoder_channel, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         )

#     def _make_attention_block(self):
#         return nn.Sequential(
#             nn.Conv2d(3, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 1, 3, padding=1),
#             nn.Sigmoid()
#         )

#     def _make_refinement_block(self):
#         return nn.Sequential(
#             nn.Conv2d(3, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 3, 3, padding=1)
#         )

#     def forward(self, x):
#         attention = self.border_attention(x)
#         x = x * attention
#         features = []
#         for block in self.encoder_blocks:
#             x = block(x)
#             features.append(x)
#             x = F.max_pool2d(x, 2)
#         skip_features = features[-self.num_skip_connections:]
#         for i in range(self.num_skip_connections):
#             x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
#             skip_feature = skip_features[-(i + 1)]
#             skip_feature = self.gated_skips[i](skip_feature)
#             x = torch.cat([x, skip_feature], dim=1)
#             x = self.decoder_blocks[i](x)
#         x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
#         return self.refinement(x)

class CascadedRectificationModel(nn.Module):
    def __init__(self, config):
        super(CascadedRectificationModel, self).__init__()
        self.coarse_net = CoarseRectificationNet(in_channels=3, feature_channels=config.initial_channels, downsample_factor=16)
        self.refinement_net = EnhancedFisheyeRectificationModel(config)

    def forward(self, x):
        coarse = self.coarse_net(x)
        refined = self.refinement_net(coarse)
        return refined

def ssim_loss(output, target):
    output_scaled = (output + 1) / 2
    target_scaled = (target + 1) / 2
    return 1 - ssim(output_scaled, target_scaled, data_range=1.0, size_average=True)

def gradient_loss(output, target):
    sobel_x = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], dtype=output.dtype, device=output.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]], dtype=output.dtype, device=output.device).view(1, 1, 3, 3)
    channels = output.shape[1]
    sobel_x = sobel_x.repeat(channels, 1, 1, 1)
    sobel_y = sobel_y.repeat(channels, 1, 1, 1)
    grad_out_x = F.conv2d(output, sobel_x, padding=1, groups=channels)
    grad_out_y = F.conv2d(output, sobel_y, padding=1, groups=channels)
    grad_tar_x = F.conv2d(target, sobel_x, padding=1, groups=target.shape[1])
    grad_tar_y = F.conv2d(target, sobel_y, padding=1, groups=target.shape[1])
    loss = F.l1_loss(grad_out_x, grad_tar_x) + F.l1_loss(grad_out_y, grad_tar_y)
    return loss

# class TotalLoss(nn.Module):
#     def __init__(self, perceptual_weight=0.5, content_weight=1.0, ssim_weight=0.2, grad_weight=0.4):
#         super(TotalLoss, self).__init__()
#         self.perceptual_loss = PerceptualLoss()
#         self.content_loss = nn.L1Loss()
#         self.perceptual_weight = perceptual_weight
#         self.content_weight = content_weight
#         self.ssim_weight = ssim_weight
#         self.grad_weight = grad_weight

#     def forward(self, output, target):
#         p_loss = self.perceptual_loss(output, target)
#         c_loss = self.content_loss(output, target)
#         s_loss = ssim_loss(output, target)
#         g_loss = gradient_loss(output, target)
#         return (self.content_weight * c_loss +
#                 self.perceptual_weight * p_loss +
#                 self.ssim_weight * s_loss +
#                 self.grad_weight * g_loss)

# class BatchImagePairLoader(Dataset):
#     def __init__(self, fisheye_dir, rectified_dir, height, width, batch_size=16, shuffle=True):
#         self.fisheye_dir = fisheye_dir
#         self.rectified_dir = rectified_dir
#         self.height = height
#         self.width = width
#         self.batch_size = batch_size
#         self.shuffle = shuffle
        
#         # Get sorted file lists
#         self.fisheye_files = sorted([f for f in os.listdir(self.fisheye_dir) 
#                                    if f.endswith(('.jpg', '.png', '.jpeg'))])
#         self.rectified_files = sorted([f for f in os.listdir(self.rectified_dir) 
#                                      if f.endswith(('.jpg', '.png', '.jpeg'))])
        
#         assert len(self.fisheye_files) == len(self.rectified_files), "Mismatch in number of images"
        
#         self.transform = transforms.Compose([
#             transforms.Resize((height, width)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         ])
        
#         self.indices = list(range(len(self.fisheye_files)))
#         if shuffle:
#             np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.fisheye_files) // self.batch_size

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        fisheye_batch = []
        rectified_batch = []
        
        for i in batch_indices:
            fisheye_path = os.path.join(self.fisheye_dir, self.fisheye_files[i])
            rectified_path = os.path.join(self.rectified_dir, self.rectified_files[i])
            
            fisheye_img = Image.open(fisheye_path).convert('RGB')
            rectified_img = Image.open(rectified_path).convert('RGB')
            
            fisheye_tensor = self.transform(fisheye_img)
            rectified_tensor = self.transform(rectified_img)
            
            fisheye_batch.append(fisheye_tensor)
            rectified_batch.append(rectified_tensor)
        
        return torch.stack(fisheye_batch), torch.stack(rectified_batch)

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.criterion = TotalLoss(config.perceptual_weight,
                                   config.content_weight,
                                   config.ssim_weight,
                                   config.grad_weight).cuda()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config.scheduler_step_size, 
            gamma=config.scheduler_gamma
        )
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        
        # Initialize from checkpoint if specified
        if config.load_checkpoint and config.checkpoint_path:
            self.load_for_training(config.checkpoint_path, config.reset_optimizer)

    def load_for_training(self, checkpoint_path, reset_optimizer=False):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if not reset_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            print(f"Resuming from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
        else:
            print("Loaded model weights only, resetting optimizer and training history")

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.savefig(os.path.join(self.config.results_dir, 'loss_plot.png'))
        plt.close()

    def save_metrics(self):
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        with open(os.path.join(self.config.results_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)

    def plot_results(self, fisheye, target, output, iteration, loss_value):
        batch_size = fisheye.shape[0]
        fig, axes = plt.subplots(3, batch_size, figsize=(3 * batch_size, 9))
        if batch_size == 1:
            axes = axes.reshape(3, 1)
        row_titles = ['Fisheye', 'Ground Truth', 'Output']
    
        def normalize_for_display(img):
            # Convert to numpy and transpose
            img = img.detach().cpu().numpy().transpose(1, 2, 0)
            # Denormalize from [-1, 1] to [0, 1]
            #img = (img + 1) / 2
            # Clip values to ensure they're in [0, 1]
            #img = np.clip(img, 0, 1)
            return img
    
        for i in range(batch_size):
            # Process fisheye image
            img_fisheye = normalize_for_display(fisheye[i])
            axes[0, i].imshow(img_fisheye)
            axes[0, i].axis('off')
            
            # Process target image
            img_target = normalize_for_display(target[i])
            axes[1, i].imshow(img_target)
            axes[1, i].axis('off')
            
            # Process output image
            img_output = normalize_for_display(output[i])
            axes[2, i].imshow(img_output)
            axes[2, i].axis('off')
            
            for row in range(3):
                axes[row, 0].set_ylabel(row_titles[row], fontsize=12)
        
        plt.suptitle(f"Iteration: {iteration}, Loss: {loss_value:.4f}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        plt.close()

    def save_checkpoint(self, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        path = os.path.join(
            self.config.checkpoint_dir,
            'checkpoint_epoch_model2.pth'
        )
        torch.save(checkpoint, path)
        
        if loss == self.best_val_loss:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for fisheye, target in val_loader:
                fisheye, target = fisheye.cuda(), target.cuda()
                output = self.model(fisheye)
                loss, _ = self.criterion(output, target)
                val_loss+= loss.item()
        avg_val_loss = val_loss / len(val_loader)
        return avg_val_loss

    def train(self, train_loader, val_loader):
        for epoch in range(self.config.start_epoch, self.config.num_epochs):
            self.model.train()
            epoch_loss = 0
            
            for i, (fisheye, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")):
                fisheye, target = fisheye.cuda(), target.cuda()
                
                self.optimizer.zero_grad()
                output = self.model(fisheye)
                
                # Get both total loss and individual components
                loss, loss_components = self.criterion(output, target)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                if i % self.config.plot_interval == 0:
                    self.plot_results(fisheye, target, output, i, loss.item())
            
            avg_train_loss = epoch_loss / len(train_loader)
            avg_val_loss = self.validate(val_loader)
            
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"Training Loss: {avg_train_loss:.4f}")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, avg_val_loss)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    print("Early stopping triggered")
                    break
            
            self.plot_losses()
            self.save_metrics()

def main():
    config = Config()
    
    print("Configuration:")
    print(f"Loading checkpoint: {config.load_checkpoint}")
    if config.load_checkpoint:
        print(f"Checkpoint path: {config.checkpoint_path}")
        print(f"Reset optimizer: {config.reset_optimizer}")
        print(f"Starting from epoch: {config.start_epoch}")
    
    print("\nData directories:")
    print(f"Train fisheye: {config.train_fisheye_dir}")
    print(f"Train rectified: {config.train_rectified_dir}")
    print(f"Val fisheye: {config.val_fisheye_dir}")
    print(f"Val rectified: {config.val_rectified_dir}")
    from datasets import BatchImagePairLoader
    try:
        dataset_train = BatchImagePairLoader(
            config.train_fisheye_dir, 
            config.train_rectified_dir, 
            config.image_size, 
            config.image_size, 
            batch_size=config.batch_size, 
            shuffle=True
        )
        
        dataset_validation = BatchImagePairLoader(
            config.val_fisheye_dir, 
            config.val_rectified_dir, 
            config.image_size, 
            config.image_size, 
            batch_size=config.batch_size, 
            shuffle=False
        )
        
        train_loader = DataLoader(dataset_train, batch_size=None)
        val_loader = DataLoader(dataset_validation, batch_size=None)
        
        model = CascadedRectificationModel(config).cuda()
        trainer = Trainer(model, config)
        
        trainer.train(train_loader, val_loader)
        
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        raise

if __name__ == "__main__":
    main()
