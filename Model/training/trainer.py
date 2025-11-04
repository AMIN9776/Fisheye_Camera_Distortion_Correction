import os
import json
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from ..config import Config
from ..losses import TotalLoss
from ..models import CascadedRectificationModel, EnhancedFisheyeRectificationModel


class Trainer:
    """
    Trainer class for Fisheye Rectification Model.
    
    Handles the complete training pipeline including optimization,
    scheduling, checkpointing, and validation.
    
    Args:
        model: Neural network model to train
        config: Configuration object with training parameters
        device: Device to train on (default: auto-detect)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: Optional[str] = None
    ):
        """Initialize the trainer with model and configuration."""
        self.model = model
        self.config = config
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize loss function
        self.criterion = TotalLoss(
            perceptual_weight=config.perceptual_weight,
            content_weight=config.content_weight,
            ssim_weight=config.ssim_weight,
            grad_weight=config.grad_weight,
            color_weight=config.color_weight
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-5
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Loss history
        self.train_losses = []
        self.val_losses = []
        self.loss_components_history = {
            'train': [],
            'val': []
        }
        
        # Load checkpoint if specified
        if config.load_checkpoint and config.checkpoint_path:
            self.load_checkpoint(config.checkpoint_path, config.reset_optimizer)
    
    def train_epoch(
        self, 
        train_loader: DataLoader,
        epoch: int
    ) -> Tuple[float, Dict[str, float]]:

        self.model.train()
        epoch_loss = 0.0
        epoch_components = {
            'perceptual': 0.0,
            'content': 0.0,
            'ssim': 0.0,
            'gradient': 0.0,
            'color': 0.0
        }
        
        # Progress bar
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
            leave=False
        )
        
        for batch_idx, (fisheye, target) in enumerate(pbar):
            # Move data to device
            fisheye = fisheye.to(self.device)
            target = target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(fisheye)
            
            # Compute loss
            loss, loss_components = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            for key in epoch_components:
                if key in loss_components:
                    epoch_components[key] += loss_components[key]
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.get_current_lr():.6f}'
            })
            
            # Visualization at intervals
            if batch_idx % self.config.plot_interval == 0 and hasattr(self, 'visualize_callback'):
                self.visualize_callback(fisheye, target, output, epoch, batch_idx, loss.item())
        
        # Calculate averages
        num_batches = len(train_loader)
        avg_loss = epoch_loss / num_batches
        avg_components = {k: v / num_batches for k, v in epoch_components.items()}
        
        return avg_loss, avg_components
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        val_loss = 0.0
        val_components = {
            'perceptual': 0.0,
            'content': 0.0,
            'ssim': 0.0,
            'gradient': 0.0,
            'color': 0.0
        }
        
        with torch.no_grad():
            for fisheye, target in tqdm(val_loader, desc="Validation", leave=False):
                # Move data to device
                fisheye = fisheye.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                output = self.model(fisheye)
                
                # Compute loss
                loss, loss_components = self.criterion(output, target)
                
                # Update statistics
                val_loss += loss.item()
                for key in val_components:
                    if key in loss_components:
                        val_components[key] += loss_components[key]
        
        # Calculate averages
        num_batches = len(val_loader)
        avg_val_loss = val_loss / num_batches
        avg_components = {k: v / num_batches for k, v in val_components.items()}
        
        return avg_val_loss, avg_components
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None,
        visualization_callback: Optional[Any] = None
    ) -> Dict[str, List[float]]:
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        if visualization_callback is not None:
            self.visualize_callback = visualization_callback
        
        print(f"Starting training on {self.device}")
        print(f"Model type: {self.config.model_type}")
        print(f"Total parameters: {self.count_parameters():,}")
        
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss, train_components = self.train_epoch(train_loader, epoch)
            
            # Validation phase
            val_loss, val_components = self.validate(val_loader)
            
            # Update history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.loss_components_history['train'].append(train_components)
            self.loss_components_history['val'].append(val_components)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Print epoch summary
            self.print_epoch_summary(epoch, train_loss, val_loss, train_components, val_components)
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                print(f"  → Validation loss improved from {self.best_val_loss:.4f} to {val_loss:.4f}")
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self.patience_counter += 1
                
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_loss, is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
            
            # Save plots
            if (epoch + 1) % 5 == 0:
                self.save_loss_plots()
                self.save_metrics()
        
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'loss_components': self.loss_components_history
        }
    
    def save_checkpoint(
        self,
        epoch: int,
        loss: float,
        is_best: bool = False
    ) -> None:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'loss_components_history': self.loss_components_history,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f'checkpoint_epoch_{epoch:04d}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"  → Saved best model to {best_path}")
        
        # Save latest checkpoint
        latest_path = os.path.join(self.config.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        reset_optimizer: bool = False
    ) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            reset_optimizer: Whether to reset optimizer state
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if not reset_optimizer:
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training history
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.loss_components_history = checkpoint.get('loss_components_history', {'train': [], 'val': []})
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.current_epoch = checkpoint.get('epoch', 0) + 1
            
            print(f"  → Resumed from epoch {self.current_epoch} with loss {checkpoint['loss']:.4f}")
        else:
            print("  → Loaded model weights only, reset optimizer and training state")
    
    def save_loss_plots(self) -> None:
        """Save loss plots to results directory."""
        if len(self.train_losses) == 0:
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Main loss plot
        axes[0, 0].plot(self.train_losses, label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.val_losses, label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Component losses
        components = ['perceptual', 'content', 'ssim', 'gradient', 'color']
        positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        
        for comp, pos in zip(components, positions):
            train_comp = [h.get(comp, 0) for h in self.loss_components_history['train']]
            val_comp = [h.get(comp, 0) for h in self.loss_components_history['val']]
            
            if train_comp:
                axes[pos].plot(train_comp, label=f'Train {comp.capitalize()}', linewidth=1.5)
                axes[pos].plot(val_comp, label=f'Val {comp.capitalize()}', linewidth=1.5)
                axes[pos].set_xlabel('Epoch')
                axes[pos].set_ylabel('Loss')
                axes[pos].set_title(f'{comp.capitalize()} Loss')
                axes[pos].legend()
                axes[pos].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        plot_path = os.path.join(self.config.results_dir, 'training_losses.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_metrics(self) -> None:
        """Save training metrics to JSON file."""
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'loss_components': self.loss_components_history,
            'best_val_loss': float(self.best_val_loss),
            'current_epoch': self.current_epoch,
            'config': self.config.__dict__
        }
        
        metrics_path = os.path.join(self.config.results_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
    
    def print_epoch_summary(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_components: Dict[str, float],
        val_components: Dict[str, float]
    ) -> None:
        """Print formatted epoch summary."""
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{self.config.num_epochs} Summary")
        print(f"{'='*60}")
        print(f"Overall:  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"LR: {self.get_current_lr():.6f} | Best Val: {self.best_val_loss:.4f}")
        print(f"{'─'*60}")
        print("Component Losses:")
        print(f"{'Component':<12} {'Train':>10} {'Val':>10} {'Diff':>10}")
        print(f"{'─'*60}")
        
        for comp in ['perceptual', 'content', 'ssim', 'gradient', 'color']:
            train_val = train_components.get(comp, 0)
            val_val = val_components.get(comp, 0)
            diff = val_val - train_val
            print(f"{comp.capitalize():<12} {train_val:>10.4f} {val_val:>10.4f} {diff:>+10.4f}")
        
        print(f"{'='*60}")
    
    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def set_learning_rate(self, lr: float) -> None:
        """Manually set learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print(f"Learning rate set to {lr:.6f}")