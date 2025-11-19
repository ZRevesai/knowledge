#!/usr/bin/env python3
"""
Evaluation script for SIDRM model.

Usage:
    python scripts/evaluate_sidrm.py --checkpoint checkpoints/best_model.pth --test
    python scripts/evaluate_sidrm.py --checkpoint checkpoints/best_model.pth --interpretability
    python scripts/evaluate_sidrm.py --checkpoint checkpoints/best_model.pth --per-population
"""

import argparse
import torch
import yaml
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.sidrm import SIDRM
from src.models.sidrm_mobile import SIDRMMobileOptimized, SIDRMEdgeOnly
from src.data.nhanes_dataset import NHANESPreprocessor, create_nhanes_dataloaders
from src.evaluation.sidrm_metrics import SIDRMEvaluator
from src.evaluation.sidrm_interpretability import SIDRMInterpretabilityEvaluator


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path: str, config: dict, input_dim: int, num_nutrients: int, device: str):
    """Load model from checkpoint."""
    # Determine model type from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model (try to infer type from checkpoint)
    model_config = config['model']

    try:
        # Try full SIDRM first
        model = SIDRM(
            input_dim=input_dim,
            num_populations=model_config['num_populations'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            num_attention_heads=model_config['num_attention_heads'],
            num_nutrients=num_nutrients,
            dropout_rate=model_config['dropout_rate']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded full SIDRM model")
    except:
        try:
            # Try mobile model
            model = SIDRMMobileOptimized(
                input_dim=input_dim,
                num_populations=model_config['num_populations'],
                hidden_dim=64,
                num_layers=3,
                num_attention_heads=4,
                num_nutrients=num_nutrients
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded mobile-optimized SIDRM model")
        except:
            # Try edge model
            model = SIDRMEdgeOnly(
                input_dim=input_dim,
                hidden_dim=48,
                num_layers=2,
                num_nutrients=num_nutrients
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded edge-only SIDRM model")

    model = model.to(device)
    model.eval()

    print(f"Loaded from checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'Unknown')}")

    return model


def evaluate_performance(args, model, dataloaders, preprocessor, device):
    """Evaluate model performance."""
    print("\n" + "="*70)
    print("PERFORMANCE EVALUATION")
    print("="*70)

    evaluator = SIDRMEvaluator(
        model=model,
        device=device,
        population_names=['Pregnant', 'Elderly', 'Children', 'Chronic_Disease'],
        nutrient_names=preprocessor.nutrient_names
    )

    # Evaluate on test set
    results = evaluator.evaluate(
        dataloaders['test'],
        return_predictions=True,
        compute_per_nutrient=args.per_nutrient,
        compute_per_population=args.per_population
    )

    # Print results
    evaluator.print_evaluation_results(results, detailed=True)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics = {
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1_score': float(results['f1_score']),
            'auc_roc': float(results.get('auc_roc', 0.0))
        }

        if 'per_population' in results:
            metrics['per_population'] = {
                name: {k: float(v) if isinstance(v, (int, float)) else v
                       for k, v in pop_metrics.items()}
                for name, pop_metrics in results['per_population'].items()
            }

        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    return results


def evaluate_interpretability(args, model, dataloaders, preprocessor, device):
    """Evaluate model interpretability."""
    print("\n" + "="*70)
    print("INTERPRETABILITY EVALUATION")
    print("="*70)

    # Get background data for SHAP
    train_batch = next(iter(dataloaders['train']))
    background_data = train_batch['features'][:100]  # Use 100 samples

    # Create interpretability evaluator
    interp_evaluator = SIDRMInterpretabilityEvaluator(
        model=model,
        background_data=background_data,
        feature_names=preprocessor.feature_names,
        num_bootstrap_samples=args.shap_samples,
        device=device
    )

    # Get test data
    test_batch = next(iter(dataloaders['test']))
    test_features = test_batch['features'][:args.eval_samples]
    test_populations = test_batch['population'][:args.eval_samples]

    # Evaluate interpretability
    print(f"Evaluating interpretability on {args.eval_samples} samples...")
    print(f"SHAP bootstrap samples: {args.shap_samples}")

    interp_results = interp_evaluator.evaluate_interpretability(
        test_features,
        test_populations,
        compute_shap=args.compute_shap,
        shap_nsamples=100
    )

    # Print results
    print("\nInterpretability Metrics:")
    print("-" * 70)

    if 'shap_stability' in interp_results:
        print(f"SHAP Stability (Equation 10):        {interp_results['shap_stability']:.4f}")

    if 'attention_consistency' in interp_results:
        print(f"Attention Consistency (Equation 11): {interp_results['attention_consistency']:.4f}")

    if 'feature_ranking_correlation' in interp_results:
        print(f"Feature Ranking Correlation (Eq 12): {interp_results['feature_ranking_correlation']:.4f}")

    print(f"\nOverall Interpretability Score:      {interp_results['overall_interpretability']:.4f}")

    # Save results
    if args.output:
        output_path = Path(args.output).parent / 'interpretability_results.json'

        interp_metrics = {
            'shap_stability': float(interp_results.get('shap_stability', 0.0)),
            'attention_consistency': float(interp_results.get('attention_consistency', 0.0)),
            'feature_ranking_correlation': float(interp_results.get('feature_ranking_correlation', 0.0)),
            'overall_interpretability': float(interp_results['overall_interpretability'])
        }

        with open(output_path, 'w') as f:
            json.dump(interp_metrics, f, indent=2)

        print(f"Interpretability results saved to: {output_path}")

    return interp_results


def main(args):
    """Main evaluation function."""
    # Load configuration
    config = load_config(args.config)

    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Prepare data
    print("\n" + "="*70)
    print("DATA PREPARATION")
    print("="*70)

    preprocessor = NHANESPreprocessor(random_state=42)

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
        batch_size=args.batch_size,
        num_workers=0  # Set to 0 for evaluation
    )

    # Get dimensions
    batch = next(iter(dataloaders['train']))
    input_dim = batch['features'].shape[1]
    num_nutrients = batch['labels'].shape[1]

    # Load model
    print("\n" + "="*70)
    print("MODEL LOADING")
    print("="*70)

    model = load_model(
        args.checkpoint,
        config,
        input_dim,
        num_nutrients,
        device
    )

    # Evaluate performance
    if args.test or args.per_population or args.per_nutrient:
        performance_results = evaluate_performance(
            args,
            model,
            dataloaders,
            preprocessor,
            device
        )

    # Evaluate interpretability
    if args.interpretability:
        interpretability_results = evaluate_interpretability(
            args,
            model,
            dataloaders,
            preprocessor,
            device
        )

    print("\n" + "="*70)
    print("EVALUATION COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SIDRM model")

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
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
        help='Device to use for evaluation'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Evaluate on test set'
    )
    parser.add_argument(
        '--per-population',
        action='store_true',
        help='Compute per-population metrics'
    )
    parser.add_argument(
        '--per-nutrient',
        action='store_true',
        help='Compute per-nutrient metrics'
    )
    parser.add_argument(
        '--interpretability',
        action='store_true',
        help='Evaluate interpretability'
    )
    parser.add_argument(
        '--compute-shap',
        action='store_true',
        help='Compute SHAP values (slow)'
    )
    parser.add_argument(
        '--shap-samples',
        type=int,
        default=10,
        help='Number of bootstrap samples for SHAP stability'
    )
    parser.add_argument(
        '--eval-samples',
        type=int,
        default=100,
        help='Number of samples to evaluate for interpretability'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/sidrm_evaluation.json',
        help='Output file for results'
    )

    args = parser.parse_args()

    # Set defaults if no specific evaluation requested
    if not (args.test or args.per_population or args.per_nutrient or args.interpretability):
        args.test = True
        args.per_population = True

    main(args)
