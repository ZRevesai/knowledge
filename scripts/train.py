#!/usr/bin/env python3
"""
Training script for KGNN model.

Usage:
    python scripts/train.py --config configs/config.yaml
"""

import argparse
import yaml
import torch
import torch.optim as optim
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessing import NutritionalDataPreprocessor
from src.data.dataset import create_dataloaders
from src.knowledge import (
    NutritionalOntology,
    NutrientInteractionNetwork,
    MaskMatrixBuilder
)
from src.models import KGNN
from src.training import KGNNTrainer, MultiObjectiveLoss
from src.visualization import plot_training_history


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config: dict):
    """Create necessary directories."""
    paths = config['paths']
    for path_key, path_value in paths.items():
        Path(path_value).mkdir(parents=True, exist_ok=True)


def prepare_data(config: dict):
    """Prepare data for training."""
    print("\n" + "=" * 70)
    print("PREPARING DATA")
    print("=" * 70)

    # Initialize preprocessor
    preprocessor = NutritionalDataPreprocessor(
        scaling_method='standard',
        imputation_method='knn',
        random_state=config['reproducibility']['seed']
    )

    # Generate synthetic data
    print("\nGenerating synthetic nutritional data...")
    n_features = sum(config['features'].values())
    features_df, labels_df = preprocessor.generate_synthetic_data(
        n_samples=5000,
        n_features=n_features,
        n_micronutrients=len(config['micronutrients'])
    )

    print(f"Generated {len(features_df)} samples with {features_df.shape[1]} features")
    print(f"Deficiency labels for {labels_df.shape[1]} micronutrients")

    # Preprocess
    print("\nPreprocessing data...")
    X, y = preprocessor.fit_transform(features_df, labels_df)

    # Split data
    print("\nSplitting data...")
    data_splits = preprocessor.split_data(
        X, y,
        train_size=config['data']['train_split'],
        val_size=config['data']['val_split'],
        test_size=config['data']['test_split'],
        stratify=config['data']['stratify']
    )

    print(f"  Train: {data_splits['X_train'].shape}")
    print(f"  Val:   {data_splits['X_val'].shape}")
    print(f"  Test:  {data_splits['X_test'].shape}")

    return preprocessor, data_splits


def build_knowledge_components(config: dict, feature_names: list):
    """Build knowledge integration components."""
    print("\n" + "=" * 70)
    print("BUILDING KNOWLEDGE COMPONENTS")
    print("=" * 70)

    # Create ontology
    print("\nCreating nutritional ontology...")
    ontology = NutritionalOntology()
    print(f"  Nutrients in ontology: {len(ontology.get_all_nutrients())}")

    # Create interaction network
    print("\nCreating nutrient interaction network...")
    interaction_network = NutrientInteractionNetwork()
    interactions = interaction_network.get_all_interactions()
    print(f"  Total interactions: {len(interactions)}")
    print(f"  Synergistic: {sum(1 for i in interactions if i[2] == 'synergistic')}")
    print(f"  Antagonistic: {sum(1 for i in interactions if i[2] == 'antagonistic')}")

    # Build mask matrix builder
    print("\nBuilding mask matrix builder...")
    mask_builder = MaskMatrixBuilder(
        ontology=ontology,
        interaction_network=interaction_network,
        similarity_threshold=config['knowledge']['similarity_threshold']
    )

    return ontology, interaction_network, mask_builder


def create_model(config: dict, input_dim: int, mask_builder, feature_names: list):
    """Create KGNN model."""
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)

    # Initialize embedding weights
    embedding_dim = config['model']['embedding_dim']
    embedding_weights = mask_builder.initialize_embedding_weights(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        feature_names=feature_names
    )

    # Create model
    model = KGNN(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        hidden_dims=config['model']['hidden_dims'],
        num_micronutrients=config['model']['num_micronutrients'],
        micronutrient_names=config['micronutrients'],
        embedding_weights=embedding_weights,
        dropout_rate=config['model']['dropout_rate'],
        activation=config['model']['activation'],
        use_batch_norm=config['model']['use_batch_norm']
    )

    print(model.get_model_summary())

    return model


def train_model(model, config: dict, dataloaders: dict, mask_builder, feature_names: list):
    """Train the KGNN model."""
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Create loss function
    criterion = MultiObjectiveLoss(
        lambda_k=config['knowledge']['knowledge_weight'],
        lambda_e=config['knowledge']['explanation_weight'],
        mask_builder=mask_builder,
        feature_names=feature_names
    )

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Create scheduler
    if config['training']['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs']
        )
    elif config['training']['scheduler'] == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=5,
            factor=0.5
        )
    else:
        scheduler = None

    # Create trainer
    trainer = KGNNTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        early_stopping_patience=config['training']['early_stopping_patience'],
        gradient_clip=config['training']['gradient_clip'],
        checkpoint_dir=config['paths']['models_dir']
    )

    # Train
    history = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        num_epochs=config['training']['num_epochs']
    )

    return trainer, history


def main():
    parser = argparse.ArgumentParser(description='Train KGNN model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    args = parser.parse_args()

    # Load config
    print("Loading configuration...")
    config = load_config(args.config)

    # Set random seeds
    import random
    import numpy as np
    seed = config['reproducibility']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = config['reproducibility']['deterministic']
        torch.backends.cudnn.benchmark = config['reproducibility']['benchmark']

    # Setup directories
    setup_directories(config)

    # Prepare data
    preprocessor, data_splits = prepare_data(config)

    # Create dataloaders
    dataloaders = create_dataloaders(
        data_splits,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        feature_names=preprocessor.feature_names,
        micronutrient_names=preprocessor.micronutrient_names
    )

    # Build knowledge components
    ontology, interaction_network, mask_builder = build_knowledge_components(
        config,
        preprocessor.feature_names
    )

    # Create model
    model = create_model(
        config,
        data_splits['X_train'].shape[1],
        mask_builder,
        preprocessor.feature_names
    )

    # Train model
    trainer, history = train_model(
        model,
        config,
        dataloaders,
        mask_builder,
        preprocessor.feature_names
    )

    # Save visualizations
    print("\n" + "=" * 70)
    print("SAVING VISUALIZATIONS")
    print("=" * 70)

    figures_dir = Path(config['paths']['figures_dir'])
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_training_history(
        history,
        save_path=str(figures_dir / 'training_history.png'),
        show=False
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nBest model saved to: {config['paths']['models_dir']}/best_model.pt")
    print(f"Training history saved to: {config['paths']['models_dir']}/training_history.json")
    print(f"Figures saved to: {config['paths']['figures_dir']}/")


if __name__ == "__main__":
    main()
