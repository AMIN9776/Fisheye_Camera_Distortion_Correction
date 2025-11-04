import os
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime
import json

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class TrainingCallback:
    """
    Base class for training callbacks.
    
    Callbacks allow for modular extensions to the training process,
    such as logging, visualization, or custom model saving logic.
    """
    
    def on_epoch_start(self, epoch: int, trainer: Any) -> None:
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, trainer: Any, metrics: Dict[str, float]) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_start(self, batch_idx: int, trainer: Any) -> None:
        """Called at the start of each batch."""
        pass
    
    def on_batch_end(self, batch_idx: int, trainer: Any, loss: float) -> None:
        """Called at the end of each batch."""
        pass
    
    def on_validation_start(self, trainer: Any) -> None:
        """Called at the start of validation."""
        pass
    
    def on_validation_end(self, trainer: Any, metrics: Dict[str, float]) -> None:
        """Called at the end of validation."""
        pass
    
    def on_training_start(self, trainer: Any) -> None:
        """Called at the start of training."""
        pass
    
    def on_training_end(self, trainer: Any) -> None:
        """Called at the end of training."""
        pass


class VisualizationCallback(TrainingCallback):
    
    def __init__(
        self,
        save_dir: str,
        frequency: int = 100,
        num_samples: int = 4
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.frequency = frequency
        self.num_samples = num_samples
        self.batch_counter = 0
    
    def on_batch_end(
        self,
        batch_idx: int,
        trainer: Any,
        loss: float,
        fisheye: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None
    ) -> None:
        """Visualize results at specified frequency."""
        self.batch_counter += 1
        
        if self.batch_counter % self.frequency == 0 and all(x is not None for x in [fisheye, target, output]):
            self.visualize_batch(
                fisheye[:self.num_samples],
                target[:self.num_samples],
                output[:self.num_samples],
                trainer.current_epoch,
                batch_idx,
                loss
            )
    
    def visualize_batch(
        self,
        fisheye: torch.Tensor,
        target: torch.Tensor,
        output: torch.Tensor,
        epoch: int,
        batch_idx: int,
        loss: float
    ) -> None:
        """Create and save visualization of batch results."""
        batch_size = min(fisheye.shape[0], self.num_samples)
        
        fig, axes = plt.subplots(3, batch_size, figsize=(3 * batch_size, 9))
        if batch_size == 1:
            axes = axes.reshape(3, 1)
        
        row_titles = ['Input (Fisheye)', 'Ground Truth', 'Output']
        
        for i in range(batch_size):
            # Convert tensors to displayable format
            img_fisheye = self.tensor_to_image(fisheye[i])
            img_target = self.tensor_to_image(target[i])
            img_output = self.tensor_to_image(output[i])
            
            # Display images
            axes[0, i].imshow(img_fisheye)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel(row_titles[0], fontsize=12)
            
            axes[1, i].imshow(img_target)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel(row_titles[1], fontsize=12)
            
            axes[2, i].imshow(img_output)
            axes[2, i].axis('off')
            if i == 0:
                axes[2, i].set_ylabel(row_titles[2], fontsize=12)
        
        plt.suptitle(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss:.4f}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure
        filename = f"epoch_{epoch:04d}_batch_{batch_idx:06d}.png"
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy image for visualization."""
        # Move to CPU and convert to numpy
        img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize if needed (assuming normalization to [-1, 1] or [0, 1])
        if img.min() < 0:
            img = (img + 1) / 2  # From [-1, 1] to [0, 1]
        
        # Clip to valid range
        img = np.clip(img, 0, 1)
        
        return img


class ModelCheckpointCallback(TrainingCallback):
    def __init__(
        self,
        checkpoint_dir: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_top_k: int = 3,
        save_last: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        
        self.best_scores = []
        self.best_paths = []
        
        # Determine comparison operator
        if mode == 'min':
            self.is_better = lambda new, old: new < old
            self.best_score = float('inf')
        else:
            self.is_better = lambda new, old: new > old
            self.best_score = float('-inf')
    
    def on_epoch_end(self, epoch: int, trainer: Any, metrics: Dict[str, float]) -> None:
        """Save checkpoint based on monitored metric."""
        if self.monitor in metrics:
            current_score = metrics[self.monitor]
            
            # Check if this is the best score
            if self.is_better(current_score, self.best_score):
                self.best_score = current_score
                self.save_checkpoint(trainer, epoch, current_score, is_best=True)
            
            # Save top-k models
            if self.save_top_k > 0:
                self.update_top_k(trainer, epoch, current_score)
        
        # Save last model
        if self.save_last:
            self.save_checkpoint(trainer, epoch, metrics.get('loss', 0), filename='last.pth')
    
    def update_top_k(self, trainer: Any, epoch: int, score: float) -> None:
        """Maintain top-k best models."""
        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch:04d}_score_{score:.4f}.pth"
        
        # Add to list
        self.best_scores.append(score)
        self.best_paths.append(checkpoint_path)
        
        # Sort based on mode
        if self.mode == 'min':
            sorted_indices = np.argsort(self.best_scores)
        else:
            sorted_indices = np.argsort(self.best_scores)[::-1]
        
        # Keep only top-k
        if len(self.best_scores) > self.save_top_k:
            # Remove worst model
            worst_idx = sorted_indices[-1]
            worst_path = self.best_paths[worst_idx]
            
            if worst_path.exists():
                worst_path.unlink()
            
            del self.best_scores[worst_idx]
            del self.best_paths[worst_idx]
        
        # Save current model if it's in top-k
        if str(checkpoint_path) in [str(p) for p in self.best_paths]:
            self.save_checkpoint(trainer, epoch, score, filename=checkpoint_path.name)
    
    def save_checkpoint(
        self,
        trainer: Any,
        epoch: int,
        score: float,
        is_best: bool = False,
        filename: Optional[str] = None
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict(),
            'score': score,
            'monitor': self.monitor,
            'config': trainer.config.__dict__
        }
        
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:04d}.pth"
        
        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"  → Saved best model (score: {score:.4f})")


class LearningRateSchedulerCallback(TrainingCallback):
    """
    Custom learning rate scheduling callback.
    
    Args:
        scheduler_type: Type of scheduler ('step', 'cosine', 'plateau')
        **kwargs: Additional arguments for the scheduler
    """
    
    def __init__(self, scheduler_type: str = 'step', **kwargs):
        self.scheduler_type = scheduler_type
        self.scheduler_kwargs = kwargs
        self.scheduler = None
    
    def on_training_start(self, trainer: Any) -> None:
        """Initialize the scheduler."""
        if self.scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                trainer.optimizer,
                step_size=self.scheduler_kwargs.get('step_size', 30),
                gamma=self.scheduler_kwargs.get('gamma', 0.1)
            )
        elif self.scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                trainer.optimizer,
                T_max=self.scheduler_kwargs.get('T_max', 100),
                eta_min=self.scheduler_kwargs.get('eta_min', 1e-6)
            )
        elif self.scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                trainer.optimizer,
                mode='min',
                factor=self.scheduler_kwargs.get('factor', 0.5),
                patience=self.scheduler_kwargs.get('patience', 10)
            )
    
    def on_epoch_end(self, epoch: int, trainer: Any, metrics: Dict[str, float]) -> None:
        """Step the scheduler."""
        if self.scheduler is not None:
            if self.scheduler_type == 'plateau':
                self.scheduler.step(metrics.get('val_loss', 0))
            else:
                self.scheduler.step()


class TensorBoardCallback(TrainingCallback):
    """
    TensorBoard logging callback.
    
    Args:
        log_dir: Directory for TensorBoard logs
        comment: Comment to add to the run name
    """
    
    def __init__(self, log_dir: str = './runs', comment: str = ''):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir, comment=comment)
            self.enabled = True
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
            self.enabled = False
    
    def on_epoch_end(self, epoch: int, trainer: Any, metrics: Dict[str, float]) -> None:
        """Log metrics to TensorBoard."""
        if not self.enabled:
            return
        
        # Log scalar metrics
        for key, value in metrics.items():
            self.writer.add_scalar(f'metrics/{key}', value, epoch)
        
        # Log learning rate
        lr = trainer.get_current_lr()
        self.writer.add_scalar('learning_rate', lr, epoch)
    
    def on_training_end(self, trainer: Any) -> None:
        """Close TensorBoard writer."""
        if self.enabled:
            self.writer.close()


class EarlyStoppingCallback(TrainingCallback):
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 20,
        mode: str = 'min',
        min_delta: float = 0.0001
    ):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.early_stop = False
    
    def on_epoch_end(self, epoch: int, trainer: Any, metrics: Dict[str, float]) -> None:
        """Check for early stopping condition."""
        if self.monitor not in metrics:
            return
        
        current = metrics[self.monitor]
        
        if self.mode == 'min':
            improved = current < (self.best_score - self.min_delta)
        else:
            improved = current > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = current
            self.counter = 0
        else:
            self.counter += 1
            
            if self.counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best {self.monitor}: {self.best_score:.4f}")
                self.early_stop = True
                trainer.should_stop = True


class MetricsLogger(TrainingCallback):
    
    def __init__(self, log_file: str, separator: str = ','):
        self.log_file = Path(log_file)
        self.separator = separator
        self.header_written = False
    
    def on_epoch_end(self, epoch: int, trainer: Any, metrics: Dict[str, float]) -> None:
        """Log metrics to file."""
        # Add epoch to metrics
        metrics = {'epoch': epoch + 1, **metrics}
        
        # Write header if first time
        if not self.header_written:
            with open(self.log_file, 'w') as f:
                header = self.separator.join(metrics.keys())
                f.write(header + '\n')
            self.header_written = True
        
        # Append metrics
        with open(self.log_file, 'a') as f:
            values = self.separator.join([str(v) for v in metrics.values()])
            f.write(values + '\n')


class CallbackManager:
    def __init__(self, callbacks: Optional[List[TrainingCallback]] = None):
        self.callbacks = callbacks or []
    
    def add_callback(self, callback: TrainingCallback) -> None:
        """Add a callback to the manager."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: TrainingCallback) -> None:
        """Remove a callback from the manager."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def on_epoch_start(self, epoch: int, trainer: Any) -> None:
        """Call on_epoch_start for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_start(epoch, trainer)
    
    def on_epoch_end(self, epoch: int, trainer: Any, metrics: Dict[str, float]) -> None:
        """Call on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, trainer, metrics)
    
    def on_batch_start(self, batch_idx: int, trainer: Any) -> None:
        """Call on_batch_start for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_start(batch_idx, trainer)
    
    def on_batch_end(self, batch_idx: int, trainer: Any, loss: float) -> None:
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch_idx, trainer, loss)
    
    def on_validation_start(self, trainer: Any) -> None:
        """Call on_validation_start for all callbacks."""
        for callback in self.callbacks:
            callback.on_validation_start(trainer)
    
    def on_validation_end(self, trainer: Any, metrics: Dict[str, float]) -> None:
        """Call on_validation_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_validation_end(trainer, metrics)
    
    def on_training_start(self, trainer: Any) -> None:
        """Call on_training_start for all callbacks."""
        for callback in self.callbacks:
            callback.on_training_start(trainer)
    
    def on_training_end(self, trainer: Any) -> None:
        """Call on_training_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_training_end(trainer)