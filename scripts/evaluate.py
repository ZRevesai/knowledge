#!/usr/bin/env python3
"""
Evaluation script for KGNN model.

Usage:
    python scripts/evaluate.py --checkpoint models/best_model.pt --config configs/config.yaml
"""

import argparse
import yaml
import torch
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
from src.evaluation import evaluate_model, print_evaluation_results
from src.visualization import (
    plot_attention_heatmap,
    plot_per_micronutrient_performance,
    visualize_explanation
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path: str, model: torch.nn.Module, device: str):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description='Evaluate KGNN model')
    parser.add_argument('--checkpoint', type=str, default='models/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    args = parser.parse_args()

    # Load config
    print("Loading configuration...")
    config = load_config(args.config)

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    print("=" * 70)
    print("PREPARING DATA")
    print("=" * 70)

    # Initialize preprocessor
    preprocessor = NutritionalDataPreprocessor(
        random_state=config['reproducibility']['seed']
    )

    # Generate data (in practice, load your test data here)
    n_features = sum(config['features'].values())
    features_df, labels_df = preprocessor.generate_synthetic_data(
        n_samples=5000,
        n_features=n_features,
        n_micronutrients=len(config['micronutrients'])
    )

    # Preprocess
    X, y = preprocessor.fit_transform(features_df, labels_df)

    # Split data
    data_splits = preprocessor.split_data(X, y)

    # Create test dataloader
    dataloaders = create_dataloaders(
        data_splits,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        feature_names=preprocessor.feature_names,
        micronutrient_names=preprocessor.micronutrient_names
    )

    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    # Build knowledge components
    ontology = NutritionalOntology()
    interaction_network = NutrientInteractionNetwork()
    mask_builder = MaskMatrixBuilder(
        ontology,
        interaction_network,
        similarity_threshold=config['knowledge']['similarity_threshold']
    )

    # Initialize model
    embedding_weights = mask_builder.initialize_embedding_weights(
        input_dim=data_splits['X_train'].shape[1],
        embedding_dim=config['model']['embedding_dim'],
        feature_names=preprocessor.feature_names
    )

    model = KGNN(
        input_dim=data_splits['X_train'].shape[1],
        embedding_dim=config['model']['embedding_dim'],
        hidden_dims=config['model']['hidden_dims'],
        num_micronutrients=config['model']['num_micronutrients'],
        micronutrient_names=config['micronutrients'],
        embedding_weights=embedding_weights
    )

    # Load checkpoint
    model = load_model(args.checkpoint, model, device)

    print("\n" + "=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70)

    # Evaluate
    results = evaluate_model(
        model=model,
        data_loader=dataloaders['test'],
        device=device,
        return_predictions=True
    )

    # Print results
    print_evaluation_results(
        results,
        model_name="KGNN",
        micronutrient_names=config['micronutrients']
    )

    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    figures_dir = Path(config['paths']['figures_dir'])
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Plot attention heatmap
    if 'attention_weights' in results['predictions']:
        print("\nGenerating attention heatmap...")
        plot_attention_heatmap(
            results['predictions']['attention_weights'][:50],  # First 50 samples
            feature_names=preprocessor.feature_names,
            save_path=str(figures_dir / 'attention_heatmap.png'),
            show=False
        )

    # Plot per-micronutrient performance
    print("Generating per-micronutrient performance plot...")
    plot_per_micronutrient_performance(
        results,
        config['micronutrients'],
        metric='f1_score',
        save_path=str(figures_dir / 'per_micronutrient_performance.png'),
        show=False
    )

    # Generate explanation for a sample
    print("Generating sample explanation...")
    sample = torch.FloatTensor(data_splits['X_test'][0:1]).to(device)
    explanation = model.explain_prediction(sample, preprocessor.feature_names)
    visualize_explanation(
        explanation,
        save_path=str(figures_dir / 'sample_explanation.png'),
        show=False
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"\nResults:")
    print(f"  Accuracy:  {results['performance']['accuracy'] * 100:.2f}%")
    print(f"  F1-Score:  {results['performance']['f1_score'] * 100:.2f}%")
    print(f"  AUC-ROC:   {results['performance'].get('auc_roc', 0.0) * 100:.2f}%")
    print(f"\nInterpretability:")
    print(f"  Overall Score: {results['interpretability']['overall_score']:.3f}")
    print(f"\nFigures saved to: {config['paths']['figures_dir']}/")


if __name__ == "__main__":
    main()
