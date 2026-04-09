"""
Configuration management for Fisheye Rectification Model.
Loads configuration from YAML file and provides easy access to all parameters.
"""

import os
import yaml
from typing import Optional, Dict, Any


class Config:
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from YAML file or use defaults.
        
        Args:
            config_path: Path to YAML configuration file. If None, looks for 
                        'config.yaml' in the same directory as this file.
        """
        # Determine config file path
        if config_path is None:
            # Look for config.yaml in the same directory as this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, 'config.yaml')
        
        # Load configuration from YAML file
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Model parameters
        self._load_model_params(config['model'])
        
        # Training parameters
        self._load_training_params(config['training'])
        
        # Loss weights
        self._load_loss_params(config['loss'])
        
        # Paths
        self._load_paths(config['paths'])
        
        # Create necessary directories
        self._create_directories()
    
    def _load_model_params(self, model_config: Dict[str, Any]) -> None:
        """Load model-related parameters."""
        self.initial_channels = model_config['initial_channels']
        self.num_blocks = model_config['num_blocks']
        self.growth_rate = model_config['growth_rate']
        self.image_size = model_config['image_size']
        self.num_skip_connections = model_config.get('num_skip_connections', self.num_blocks)
        self.model_type = model_config.get('model_type', 'cascaded')
        
        # Checkpoint loading parameters
        self.load_checkpoint = model_config.get('load_checkpoint', False)
        self.checkpoint_path = model_config.get('checkpoint_path', None)
        self.reset_optimizer = model_config.get('reset_optimizer', False)
    
    def _load_training_params(self, training_config: Dict[str, Any]) -> None:
        """Load training-related parameters."""
        self.batch_size = training_config['batch_size']
        self.learning_rate = training_config['learning_rate']
        self.num_epochs = training_config['num_epochs']
        self.start_epoch = training_config.get('start_epoch', 0)
        self.plot_interval = training_config['plot_interval']
        self.validation_split = training_config.get('validation_split', 0.02)
        self.scheduler_step_size = training_config['scheduler_step_size']
        self.scheduler_gamma = training_config['scheduler_gamma']
        self.early_stopping_patience = training_config['early_stopping_patience']
    
    def _load_loss_params(self, loss_config: Dict[str, Any]) -> None:
        """Load loss function weights."""
        self.perceptual_weight = loss_config['perceptual_weight']
        self.content_weight = loss_config['content_weight']
        self.ssim_weight = loss_config.get('ssim_weight', 0.1)
        self.grad_weight = loss_config.get('grad_weight', 0.1)
        self.color_weight = loss_config.get('color_weight', 0.5)
    
    def _load_paths(self, paths_config: Dict[str, Any]) -> None:
        """Load file and directory paths."""
        self.checkpoint_dir = paths_config['checkpoint_dir']
        self.results_dir = paths_config['results_dir']
        self.data_dir = paths_config.get('data_dir', './data')
        self.train_fisheye_dir = paths_config['train_fisheye_dir']
        self.train_rectified_dir = paths_config['train_rectified_dir']
        self.val_fisheye_dir = paths_config['val_fisheye_dir']
        self.val_rectified_dir = paths_config['val_rectified_dir']
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def update_from_args(self, args) -> None:
        # Update paths if provided
        if hasattr(args, 'checkpoint') and args.checkpoint:
            self.checkpoint_path = args.checkpoint
            self.load_checkpoint = True
        
        if hasattr(args, 'batch_size') and args.batch_size:
            self.batch_size = args.batch_size
        
        if hasattr(args, 'learning_rate') and args.learning_rate:
            self.learning_rate = args.learning_rate
        
        if hasattr(args, 'epochs') and args.epochs:
            self.num_epochs = args.epochs
        
        if hasattr(args, 'model_type') and args.model_type:
            self.model_type = args.model_type
    
    def save_to_file(self, filepath: str) -> None:
        config_dict = {
            'model': {
                'initial_channels': self.initial_channels,
                'num_blocks': self.num_blocks,
                'growth_rate': self.growth_rate,
                'image_size': self.image_size,
                'num_skip_connections': self.num_skip_connections,
                'model_type': self.model_type,
                'load_checkpoint': self.load_checkpoint,
                'checkpoint_path': self.checkpoint_path,
                'reset_optimizer': self.reset_optimizer
            },
            'training': {
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs,
                'start_epoch': self.start_epoch,
                'plot_interval': self.plot_interval,
                'validation_split': self.validation_split,
                'scheduler_step_size': self.scheduler_step_size,
                'scheduler_gamma': self.scheduler_gamma,
                'early_stopping_patience': self.early_stopping_patience
            },
            'loss': {
                'perceptual_weight': self.perceptual_weight,
                'content_weight': self.content_weight,
                'ssim_weight': self.ssim_weight,
                'grad_weight': self.grad_weight,
                'color_weight': self.color_weight
            },
            'paths': {
                'checkpoint_dir': self.checkpoint_dir,
                'results_dir': self.results_dir,
                'data_dir': self.data_dir,
                'train_fisheye_dir': self.train_fisheye_dir,
                'train_rectified_dir': self.train_rectified_dir,
                'val_fisheye_dir': self.val_fisheye_dir,
                'val_rectified_dir': self.val_rectified_dir
            }
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"""
Fisheye Rectification Configuration:
*********************************************
Model:
  Type: {self.model_type}
  Initial Channels: {self.initial_channels}
  Blocks: {self.num_blocks}
  Growth Rate: {self.growth_rate}
  Image Size: {self.image_size}
  Skip Connections: {self.num_skip_connections}

Training:
  Batch Size: {self.batch_size}
  Learning Rate: {self.learning_rate}
  Epochs: {self.num_epochs}
  Early Stopping Patience: {self.early_stopping_patience}

Loss Weights:
  Perceptual: {self.perceptual_weight}
  Content: {self.content_weight}
  SSIM: {self.ssim_weight}
  Gradient: {self.grad_weight}
  Color: {self.color_weight}
"""