#!/usr/bin/env python3
"""
Training script for SIDRM model.

Usage:
    python scripts/train_sidrm.py --config configs/sidrm_config.yaml --device cuda
    python scripts/train_sidrm.py --mobile --epochs 100
    python scripts/train_sidrm.py --resume checkpoints/checkpoint_epoch_50.pth
"""

import argparse
import yaml
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.sidrm import SIDRM
from src.models.sidrm_mobile import SIDRMMobileOptimized, SIDRMEdgeOnly
from src.data.nhanes_dataset import NHANESPreprocessor, create_nhanes_dataloaders
from src.training.sidrm_trainer import SIDRMTrainer
from src.training.sidrm_losses import SIDRMMultiObjectiveLoss


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict, input_dim: int, num_nutrients: int, mobile: bool = False, edge: bool = False):
    """Create SIDRM model based on configuration."""
    model_config = config['model']

    if edge:
        print("Creating Edge-Only SIDRM model...")
        model = SIDRMEdgeOnly(
            input_dim=input_dim,
            hidden_dim=48,
            num_layers=2,
            num_nutrients=num_nutrients,
            dropout_rate=0.1
        )
    elif mobile:
        print("Creating Mobile-Optimized SIDRM model...")
        model = SIDRMMobileOptimized(
            input_dim=input_dim,
            num_populations=model_config['num_populations'],
            hidden_dim=64,
            num_layers=3,
            num_attention_heads=4,
            num_nutrients=num_nutrients,
            dropout_rate=0.2
        )
    else:
        print("Creating Full SIDRM model...")
        model = SIDRM(
            input_dim=input_dim,
            num_populations=model_config['num_populations'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            num_attention_heads=model_config['num_attention_heads'],
            num_nutrients=num_nutrients,
            dropout_rate=model_config['dropout_rate'],
            use_population_specific=model_config['use_population_specific']
        )

    print(model.get_model_summary())
    return model


def main(args):
    """Main training function."""
    # Load configuration
    config = load_config(args.config)

    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Prepare data
    print("\n" + "="*70)
    print("DATA PREPARATION")
    print("="*70)

    preprocessor = NHANESPreprocessor(
        random_state=config['training'].get('random_state', 42)
    )

    # Generate/load NHANES data
    data_splits = preprocessor.generate_synthetic_nhanes_data(
        n_samples=config['data']['n_samples'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio']
    )

    # Create dataloaders
    dataloaders = create_nhanes_dataloaders(
        data_splits,
        preprocessor,
        batch_size=config['training']['batch_size'],
        num_workers=config['data'].get('num_workers', 4)
    )

    # Get data dimensions
    batch = next(iter(dataloaders['train']))
    input_dim = batch['features'].shape[1]
    num_nutrients = batch['labels'].shape[1]

    print(f"\nData dimensions:")
    print(f"  Input features: {input_dim}")
    print(f"  Number of nutrients: {num_nutrients}")

    # Create model
    print("\n" + "="*70)
    print("MODEL CREATION")
    print("="*70)

    model = create_model(
        config,
        input_dim,
        num_nutrients,
        mobile=args.mobile,
        edge=args.edge
    )

    # Create loss function
    loss_config = config['loss']
    criterion = SIDRMMultiObjectiveLoss(
        num_nutrients=num_nutrients,
        alpha=loss_config['alpha'],
        beta=loss_config['beta'],
        gamma=loss_config['gamma'],
        delta=loss_config['delta']
    )

    print(f"\nLoss function weights:")
    print(f"  Accuracy (α): {loss_config['alpha']}")
    print(f"  Interpretability (β): {loss_config['beta']}")
    print(f"  Clinical (γ): {loss_config['gamma']}")
    print(f"  Safety (δ): {loss_config['delta']}")

    # Create trainer
    print("\n" + "="*70)
    print("TRAINING SETUP")
    print("="*70)

    train_config = config['training']
    trainer = SIDRMTrainer(
        model=model,
        criterion=criterion,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        learning_rate=train_config['learning_rate'],
        weight_decay=train_config['weight_decay'],
        device=device,
        patience=train_config['patience'],
        decay_factor=train_config['decay_factor'],
        gradient_clip=train_config.get('gradient_clip', 1.0),
        save_dir=args.checkpoint_dir
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    num_epochs = args.epochs if args.epochs else train_config['num_epochs']

    history = trainer.train(
        num_epochs=num_epochs,
        early_stopping=True,
        early_stopping_patience=train_config.get('early_stopping_patience', 30),
        save_best=True,
        verbose=True
    )

    # Save final history
    history_path = Path(args.checkpoint_dir) / 'training_history.json'
    trainer.save_history(str(history_path))

    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Best epoch: {trainer.best_epoch}")
    print(f"Model saved to: {args.checkpoint_dir}/best_model.pth")
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SIDRM model")

    parser.add_argument(
        '--config',
        type=str,
        default='configs/sidrm_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for training'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--mobile',
        action='store_true',
        help='Train mobile-optimized model'
    )
    parser.add_argument(
        '--edge',
        action='store_true',
        help='Train edge-only model'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints/sidrm',
        help='Directory to save checkpoints'
    )

    args = parser.parse_args()
    main(args)
