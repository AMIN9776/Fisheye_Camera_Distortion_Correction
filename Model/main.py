import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from torch.utils.data import DataLoader

# Import all modules
from config import Config
from data import BatchImagePairLoader
from models import CascadedRectificationModel, EnhancedFisheyeRectificationModel
from losses import TotalLoss
from training import (
    Trainer,
    VisualizationCallback,
    ModelCheckpointCallback,
    EarlyStoppingCallback,
    TensorBoardCallback,
    MetricsLogger,
    CallbackManager
)
from utils import (
    set_random_seed,
    get_device,
    create_experiment_dir,
    setup_logging,
    get_image_files,
    plot_training_history,
    visualize_batch_results,
    evaluate_batch,
    MetricsTracker,
    Timer,
    get_latest_checkpoint,
    backup_code,
    save_config
)


def train(args: argparse.Namespace) -> None:
    """
    Train the fisheye rectification model.
    
    Args:
        args: Command-line arguments
    """
    # Load configuration
    config = Config(args.config)
    
    # Override with command-line arguments
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.num_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.model_type:
        config.model_type = args.model_type
    if args.checkpoint:
        config.checkpoint_path = args.checkpoint
        config.load_checkpoint = True
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    
    # Create experiment directory
    if args.experiment_name:
        exp_dir = create_experiment_dir(config.results_dir, args.experiment_name)
        config.checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
        config.results_dir = os.path.join(exp_dir, 'results')
    
    # Setup logging
    logger = setup_logging(
        log_file=os.path.join(config.results_dir, 'training.log'),
        level=args.log_level
    )
    
    logger.info("="*60)
    logger.info("Fisheye Rectification Model Training")
    logger.info("="*60)
    
    # Backup code if requested
    if args.backup_code:
        backup_code('.', os.path.join(config.results_dir, 'code_backup'))
    
    # Save configuration
    save_config(config, os.path.join(config.results_dir, 'config.yaml'))
    
    # Get device
    device = get_device(args.device, args.gpu_id)
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Loading datasets...")
    
    with Timer("Dataset loading"):
        train_dataset = BatchImagePairLoader(
            data_dir=config.train_fisheye_dir,
            label_dir=config.train_rectified_dir,
            height=config.image_size,
            width=config.image_size,
            batch_size=config.batch_size,
            shuffle=True
        )
        
        val_dataset = BatchImagePairLoader(
            data_dir=config.val_fisheye_dir,
            label_dir=config.val_rectified_dir,
            height=config.image_size,
            width=config.image_size,
            batch_size=config.batch_size,
            shuffle=False
        )
    
    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=args.num_workers)
    
    logger.info(f"Training samples: {len(train_loader) * config.batch_size}")
    logger.info(f"Validation samples: {len(val_loader) * config.batch_size}")
    
    # Create model
    logger.info(f"Creating {config.model_type} model...")
    
    with Timer("Model creation"):
        if config.model_type == 'cascaded':
            model = CascadedRectificationModel(config)
        else:
            model = EnhancedFisheyeRectificationModel(config)
    
    # Print model summary
    from utils import count_parameters
    param_counts = count_parameters(model)
    logger.info(f"Model parameters: {param_counts['total']:,} total, {param_counts['trainable']:,} trainable")
    
    # Create trainer
    trainer = Trainer(model, config, device=str(device))
    
    # Setup callbacks
    callbacks = []
    
    # Visualization callback
    if not args.no_visualization:
        callbacks.append(
            VisualizationCallback(
                save_dir=os.path.join(config.results_dir, 'visualizations'),
                frequency=config.plot_interval,
                num_samples=4
            )
        )
    
    # Model checkpoint callback
    callbacks.append(
        ModelCheckpointCallback(
            checkpoint_dir=config.checkpoint_dir,
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        )
    )
    
    # Early stopping callback
    if not args.no_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                monitor='val_loss',
                patience=config.early_stopping_patience,
                mode='min',
                min_delta=0.0001
            )
        )
    
    # TensorBoard callback
    if args.tensorboard:
        callbacks.append(
            TensorBoardCallback(
                log_dir=os.path.join(config.results_dir, 'tensorboard'),
                comment=args.experiment_name or ''
            )
        )
    
    # Metrics logger
    callbacks.append(
        MetricsLogger(
            log_file=os.path.join(config.results_dir, 'metrics.csv')
        )
    )
    
    callback_manager = CallbackManager(callbacks)
    
    # Train the model
    logger.info("Starting training...")
    
    try:
        # Call training start callbacks
        callback_manager.on_training_start(trainer)
        
        # Training loop
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.num_epochs
        )
        
        # Call training end callbacks
        callback_manager.on_training_end(trainer)
        
        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        
        # Plot final training history
        plot_training_history(
            history['train_losses'],
            history['val_losses'],
            history.get('loss_components'),
            save_path=os.path.join(config.results_dir, 'final_training_history.png'),
            show=False
        )
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint(
            trainer.current_epoch,
            trainer.val_losses[-1] if trainer.val_losses else 0,
            is_best=False
        )
        logger.info("Checkpoint saved")
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


def evaluate(args: argparse.Namespace) -> None:
    # Setup
    set_random_seed(args.seed)
    device = get_device(args.device, args.gpu_id)
    
    # Setup logging
    logger = setup_logging(level=args.log_level)
    
    logger.info("="*60)
    logger.info("Model Evaluation")
    logger.info("="*60)
    
    # Find checkpoint
    checkpoint_path = args.checkpoint
    if not checkpoint_path and args.checkpoint_dir:
        checkpoint_path = get_latest_checkpoint(args.checkpoint_dir)
    
    if not checkpoint_path:
        logger.error("No checkpoint found. Please specify --checkpoint or --checkpoint_dir")
        sys.exit(1)
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load configuration from checkpoint
    if 'config' in checkpoint:
        config = Config()
        for key, value in checkpoint['config'].items():
            setattr(config, key, value)
    else:
        config = Config(args.config)
    
    # Override paths if provided
    if args.val_fisheye_dir:
        config.val_fisheye_dir = args.val_fisheye_dir
    if args.val_rectified_dir:
        config.val_rectified_dir = args.val_rectified_dir
    
    # Create model
    logger.info(f"Creating {config.model_type} model...")
    if config.model_type == 'cascaded':
        model = CascadedRectificationModel(config)
    else:
        model = EnhancedFisheyeRectificationModel(config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load validation data
    logger.info("Loading validation dataset...")
    val_dataset = BatchImagePairLoader(
        data_dir=config.val_fisheye_dir,
        label_dir=config.val_rectified_dir,
        height=config.image_size,
        width=config.image_size,
        batch_size=args.batch_size or config.batch_size,
        shuffle=False
    )
    
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=args.num_workers)
    
    # Metrics to compute
    metrics_to_compute = ['psnr', 'ssim', 'mae', 'rmse']
    if args.compute_lpips:
        metrics_to_compute.append('lpips')
    if args.compute_fid:
        metrics_to_compute.append('fid')
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(metrics_to_compute)
    
    # Evaluation loop
    logger.info("Running evaluation...")
    
    all_metrics = []
    
    with Timer("Evaluation"):
        with torch.no_grad():
            for batch_idx, (fisheye, target) in enumerate(val_loader):
                fisheye = fisheye.to(device)
                target = target.to(device)
                
                # Forward pass
                output = model(fisheye)
                
                # Compute metrics
                batch_metrics = evaluate_batch(output, target, metrics_to_compute)
                all_metrics.append(batch_metrics)
                metrics_tracker.update(batch_metrics, batch_idx)
                
                # Save sample visualizations
                if args.save_visualizations and batch_idx < args.num_visualization_batches:
                    vis_path = os.path.join(
                        args.output_dir,
                        f'batch_{batch_idx:04d}_visualization.png'
                    )
                    visualize_batch_results(
                        fisheye, target, output,
                        num_samples=min(4, fisheye.shape[0]),
                        save_path=vis_path,
                        show=False
                    )
                
                # Progress
                if batch_idx % 10 == 0:
                    logger.info(f"Processed {batch_idx}/{len(val_loader)} batches")
    
    # Print results
    summary = metrics_tracker.get_summary()
    
    logger.info("\n" + "="*60)
    logger.info("Evaluation Results")
    logger.info("="*60)
    
    for metric, stats in summary.items():
        logger.info(f"\n{metric.upper()}:")
        logger.info(f"  Mean:  {stats['mean']:.4f}")
        logger.info(f"  Std:   {stats['std']:.4f}")
        logger.info(f"  Min:   {stats['min']:.4f}")
        logger.info(f"  Max:   {stats['max']:.4f}")
    
    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        import json
        with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"\nResults saved to {args.output_dir}")


def infer(args: argparse.Namespace) -> None:
    """
    Run inference on new images.
    
    Args:
        args: Command-line arguments
    """
    from inference import FisheyeRectifier
    
    # Setup
    device = get_device(args.device, args.gpu_id)
    logger = setup_logging(level=args.log_level)
    
    logger.info("="*60)
    logger.info("Inference Mode")
    logger.info("="*60)
    
    # Find checkpoint
    checkpoint_path = args.checkpoint
    if not checkpoint_path and args.checkpoint_dir:
        checkpoint_path = get_latest_checkpoint(args.checkpoint_dir)
    
    if not checkpoint_path:
        logger.error("No checkpoint found. Please specify --checkpoint or --checkpoint_dir")
        sys.exit(1)
    
    # Create rectifier
    rectifier = FisheyeRectifier(
        checkpoint_path=checkpoint_path,
        device=str(device),
        config_path=args.config
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        logger.info(f"Processing single image: {input_path}")
        
        with Timer("Inference"):
            if args.visualize:
                vis_path = os.path.join(args.output_dir, f"{input_path.stem}_comparison.png")
                rectifier.visualize_rectification(str(input_path), vis_path)
            else:
                rectified, _ = rectifier.rectify_image(str(input_path))
                output_path = os.path.join(args.output_dir, f"{input_path.stem}_rectified.png")
                
                from PIL import Image
                rectified_img = Image.fromarray((rectified * 255).astype('uint8'))
                rectified_img.save(output_path)
                logger.info(f"Saved result to {output_path}")
    
    elif input_path.is_dir():
        # Directory of images
        image_files = get_image_files(str(input_path))
        logger.info(f"Found {len(image_files)} images to process")
        
        if len(image_files) == 0:
            logger.error("No valid images found")
            sys.exit(1)
        
        with Timer("Batch inference"):
            # Process in batches
            rectified_images = rectifier.rectify_batch(
                [str(f) for f in image_files],
                batch_size=args.batch_size or 4
            )
            
            # Save results
            for img_file, rectified in zip(image_files, rectified_images):
                output_path = os.path.join(args.output_dir, f"{img_file.stem}_rectified.png")
                
                from PIL import Image
                rectified_img = Image.fromarray((rectified * 255).astype('uint8'))
                rectified_img.save(output_path)
            
            logger.info(f"Processed {len(image_files)} images")
    
    else:
        logger.error(f"Invalid input path: {input_path}")
        sys.exit(1)
    
    logger.info(f"Results saved to {args.output_dir}")


def main():
    """Main entry point."""
    
    # Create main parser
    parser = argparse.ArgumentParser(
        description='Fisheye Image Rectification Model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Mode to run')
    subparsers.required = True
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps'],
                              help='Device to use')
    common_parser.add_argument('--gpu_id', type=int, help='Specific GPU ID')
    common_parser.add_argument('--seed', type=int, default=42,
                              help='Random seed for reproducibility')
    common_parser.add_argument('--log_level', type=str, default='INFO',
                              choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                              help='Logging level')
    common_parser.add_argument('--num_workers', type=int, default=4,
                              help='Number of data loading workers')
    
    # Training parser
    train_parser = subparsers.add_parser('train', parents=[common_parser],
                                        help='Train the model')
    train_parser.add_argument('--config', type=str, required=True,
                             help='Path to configuration file')
    train_parser.add_argument('--checkpoint', type=str,
                             help='Resume from checkpoint')
    train_parser.add_argument('--experiment_name', type=str,
                             help='Name for the experiment')
    train_parser.add_argument('--batch_size', type=int,
                             help='Override batch size')
    train_parser.add_argument('--epochs', type=int,
                             help='Override number of epochs')
    train_parser.add_argument('--lr', type=float,
                             help='Override learning rate')
    train_parser.add_argument('--model_type', type=str,
                             choices=['cascaded', 'enhanced'],
                             help='Override model type')
    train_parser.add_argument('--tensorboard', action='store_true',
                             help='Enable TensorBoard logging')
    train_parser.add_argument('--no_visualization', action='store_true',
                             help='Disable visualization during training')
    train_parser.add_argument('--no_early_stopping', action='store_true',
                             help='Disable early stopping')
    train_parser.add_argument('--backup_code', action='store_true',
                             help='Backup source code')
    
    # Evaluation parser
    eval_parser = subparsers.add_parser('evaluate', parents=[common_parser],
                                       help='Evaluate a trained model')
    eval_parser.add_argument('--checkpoint', type=str,
                            help='Path to model checkpoint')
    eval_parser.add_argument('--checkpoint_dir', type=str,
                            help='Directory to find latest checkpoint')
    eval_parser.add_argument('--config', type=str,
                            help='Path to configuration file')
    eval_parser.add_argument('--val_fisheye_dir', type=str,
                            help='Validation fisheye directory')
    eval_parser.add_argument('--val_rectified_dir', type=str,
                            help='Validation rectified directory')
    eval_parser.add_argument('--batch_size', type=int,
                            help='Batch size for evaluation')
    eval_parser.add_argument('--output_dir', type=str, default='./evaluation_output',
                            help='Output directory for results')
    eval_parser.add_argument('--compute_lpips', action='store_true',
                            help='Compute LPIPS metric')
    eval_parser.add_argument('--compute_fid', action='store_true',
                            help='Compute FID metric')
    eval_parser.add_argument('--save_visualizations', action='store_true',
                            help='Save visualization images')
    eval_parser.add_argument('--num_visualization_batches', type=int, default=5,
                            help='Number of batches to visualize')
    
    # Inference parser
    infer_parser = subparsers.add_parser('infer', parents=[common_parser],
                                        help='Run inference on new images')
    infer_parser.add_argument('--checkpoint', type=str,
                             help='Path to model checkpoint')
    infer_parser.add_argument('--checkpoint_dir', type=str,
                             help='Directory to find latest checkpoint')
    infer_parser.add_argument('--config', type=str,
                             help='Path to configuration file')
    infer_parser.add_argument('--input', type=str, required=True,
                             help='Input image or directory')
    infer_parser.add_argument('--output_dir', type=str, default='./inference_output',
                             help='Output directory')
    infer_parser.add_argument('--batch_size', type=int,
                             help='Batch size for processing')
    infer_parser.add_argument('--visualize', action='store_true',
                             help='Create before/after visualization')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute appropriate mode
    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    elif args.mode == 'infer':
        infer(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()