#!/usr/bin/env python3
"""
Model evaluation script.

Usage:
    python scripts/evaluate_model.py --model checkpoints/best_model.pth --data_dir ./data/food101
"""

import argparse
import torch
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.nutrient_model import NutrientAnalysisModel
from src.data.dataset import create_data_loaders
from src.evaluation.metrics import (
    compute_accuracy,
    compute_top_k_accuracy,
    compute_mae,
    compute_mape,
    compute_food_security_metrics,
    MetricsCalculator
)
from src.utils.helpers import setup_logging, save_json


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate nutrient analysis model')

    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--output', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')

    return parser.parse_args()


def evaluate_model(model, data_loader, device='cuda'):
    """
    Evaluate model on dataset.

    Args:
        model: Trained model
        data_loader: Data loader
        device: Device to use

    Returns:
        Dictionary with evaluation results
    """
    model.eval()

    all_predictions = {
        'food_logits': [],
        'portion': [],
        'macronutrients': [],
        'micronutrients': []
    }

    all_targets = {
        'label': [],
        'portion': [],
        'macronutrients': [],
        'micronutrients': [],
        'food_security_category': []
    }

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            images = batch['image'].to(device)

            # Forward pass
            predictions = model(images)

            # Store predictions
            for key in all_predictions:
                if key in predictions:
                    all_predictions[key].append(predictions[key].cpu())

            # Store targets
            for key in all_targets:
                if key in batch:
                    all_targets[key].append(batch[key])

    # Concatenate results
    for key in all_predictions:
        if all_predictions[key]:
            all_predictions[key] = torch.cat(all_predictions[key], dim=0)

    for key in all_targets:
        if all_targets[key]:
            all_targets[key] = torch.cat(all_targets[key], dim=0)

    # Compute metrics
    metrics_calc = MetricsCalculator()
    metrics = metrics_calc.compute_all_metrics(
        all_predictions,
        all_targets,
        food_security_category=all_targets.get('food_security_category')
    )

    return metrics


def main():
    """Main evaluation function."""
    args = parse_args()

    # Setup
    logger = setup_logging()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("Model Evaluation")
    logger.info("="*80)

    # Setup device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'

    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"\nLoading model from: {args.model}")
    model = NutrientAnalysisModel.from_pretrained(args.model)
    model.to(device)
    model.eval()

    # Load data
    logger.info(f"\nLoading {args.split} dataset...")
    data_loaders = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4
    )

    data_loader = data_loaders[args.split]
    logger.info(f"Number of samples: {len(data_loaders[f'{args.split}_dataset'])}")

    # Evaluate
    logger.info("\nEvaluating model...")
    metrics = evaluate_model(model, data_loader, device=device)

    # Print results
    logger.info("\n" + "="*80)
    logger.info("Evaluation Results")
    logger.info("="*80)

    if 'top1_accuracy' in metrics:
        logger.info(f"\nFood Recognition:")
        logger.info(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.4f} ({metrics['top1_accuracy']*100:.2f}%)")

    if 'top5_accuracy' in metrics:
        logger.info(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.4f} ({metrics['top5_accuracy']*100:.2f}%)")

    if 'macro_mae' in metrics:
        logger.info(f"\nNutrient Estimation:")
        logger.info(f"  Macronutrient MAE: {metrics['macro_mae']:.4f}")

    if 'macro_mape' in metrics:
        logger.info(f"  Macronutrient MAPE: {metrics['macro_mape']:.2f}%")

    if 'micro_mae' in metrics:
        logger.info(f"  Micronutrient MAE: {metrics['micro_mae']:.4f}")

    if 'portion_mae' in metrics:
        logger.info(f"  Portion MAE: {metrics['portion_mae']:.4f}g")

    if 'food_security' in metrics:
        logger.info(f"\nFood Security Categories:")
        for category, accuracy in metrics['food_security'].items():
            logger.info(f"  {category}: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Save results
    results_file = output_dir / f'evaluation_results_{args.split}.json'

    # Convert tensors to float for JSON serialization
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            metrics_serializable[key] = value.item()
        elif isinstance(value, dict):
            metrics_serializable[key] = {k: v if not isinstance(v, torch.Tensor) else v.item()
                                        for k, v in value.items()}
        else:
            metrics_serializable[key] = value

    save_json(metrics_serializable, results_file)
    logger.info(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
