#!/usr/bin/env python3
"""
Training script for nutrient analysis model.

Usage:
    python scripts/train_model.py --config config/config.yaml
"""

import argparse
import yaml
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.nutrient_model import NutrientAnalysisModel, TeacherModel
from src.data.dataset import create_data_loaders
from src.training.train import Trainer
from src.utils.helpers import set_seed, setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train nutrient analysis model')

    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to dataset directory (overrides config)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume training')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, overrides config)')

    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command line arguments
    if args.data_dir:
        config['dataset']['data_dir'] = args.data_dir
    if args.device:
        config['hardware']['device'] = args.device

    # Setup
    set_seed(config.get('seed', 42))
    logger = setup_logging(config.get('logging', {}))

    logger.info("="*80)
    logger.info("Nutrient Analysis Model Training")
    logger.info("="*80)

    # Determine device
    device = config['hardware']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'

    logger.info(f"Using device: {device}")

    # Create data loaders
    logger.info("\nCreating data loaders...")
    data_loaders = create_data_loaders(
        data_dir=config['dataset']['data_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        image_size=config['dataset']['image_size']
    )

    logger.info(f"Train samples: {len(data_loaders['train_dataset'])}")
    logger.info(f"Val samples: {len(data_loaders['val_dataset'])}")
    logger.info(f"Test samples: {len(data_loaders['test_dataset'])}")

    # Create model
    logger.info("\nCreating model...")
    model = NutrientAnalysisModel(
        num_classes=config['dataset']['num_classes'],
        width_multiplier=config['model']['width_multiplier'],
        use_se=config['model']['squeeze_excitation'],
        use_sa=config['model']['shuffle_attention'],
        dropout=config['model']['dropout']
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Estimated size: {total_params * 4 / (1024**2):.2f} MB")

    # Knowledge distillation setup
    teacher_model = None
    use_distillation = config.get('distillation', {}).get('enabled', False)

    if use_distillation:
        logger.info("\nSetting up knowledge distillation...")
        teacher_checkpoint = config['distillation'].get('teacher_checkpoint')

        if teacher_checkpoint and Path(teacher_checkpoint).exists():
            logger.info(f"Loading teacher model from: {teacher_checkpoint}")
            teacher_model = TeacherModel(num_classes=config['dataset']['num_classes'])
            teacher_model.load_state_dict(torch.load(teacher_checkpoint))
        else:
            logger.warning("Teacher checkpoint not found, distillation disabled")
            use_distillation = False

    # Create trainer
    logger.info("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        config=config,
        device=device,
        use_distillation=use_distillation,
        teacher_model=teacher_model
    )

    # Resume from checkpoint if specified
    if args.checkpoint:
        logger.info(f"\nResuming from checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)

    # Train
    logger.info("\nStarting training...")
    logger.info(f"Number of epochs: {config['training']['num_epochs']}")
    logger.info(f"Batch size: {config['training']['batch_size']}")
    logger.info(f"Learning rate: {config['training']['learning_rate']}")

    trainer.train(num_epochs=config['training']['num_epochs'])

    logger.info("\nTraining completed!")
    logger.info(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
    logger.info(f"Checkpoints saved to: {trainer.checkpoint_dir}")


if __name__ == '__main__':
    main()
