"""
Performance evaluation metrics for KGNN and baseline models.

Implements metrics from Table I in the paper:
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


def calculate_performance_metrics(y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  y_prob: Optional[np.ndarray] = None,
                                  threshold: float = 0.5,
                                  per_class: bool = True) -> Dict:
    """
    Calculate comprehensive performance metrics.

    Args:
        y_true: True labels (n_samples, n_classes)
        y_pred: Predicted labels (n_samples, n_classes) - binary
        y_prob: Predicted probabilities (n_samples, n_classes) - for AUC
        threshold: Threshold for converting probabilities to binary predictions
        per_class: Whether to calculate metrics per class

    Returns:
        Dictionary with performance metrics
    """
    metrics = {}

    # Overall metrics (micro-averaged across all classes)
    metrics['accuracy'] = accuracy_score(
        y_true.flatten(),
        y_pred.flatten()
    )

    metrics['precision'] = precision_score(
        y_true.flatten(),
        y_pred.flatten(),
        zero_division=0
    )

    metrics['recall'] = recall_score(
        y_true.flatten(),
        y_pred.flatten(),
        zero_division=0
    )

    metrics['f1_score'] = f1_score(
        y_true.flatten(),
        y_pred.flatten(),
        zero_division=0
    )

    # AUC-ROC (requires probabilities)
    if y_prob is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(
                y_true.flatten(),
                y_prob.flatten()
            )
        except ValueError:
            metrics['auc_roc'] = 0.0

    # Per-class metrics
    if per_class:
        n_classes = y_true.shape[1]
        metrics['per_class'] = {}

        for i in range(n_classes):
            class_metrics = {}

            class_metrics['accuracy'] = accuracy_score(
                y_true[:, i],
                y_pred[:, i]
            )

            class_metrics['precision'] = precision_score(
                y_true[:, i],
                y_pred[:, i],
                zero_division=0
            )

            class_metrics['recall'] = recall_score(
                y_true[:, i],
                y_pred[:, i],
                zero_division=0
            )

            class_metrics['f1_score'] = f1_score(
                y_true[:, i],
                y_pred[:, i],
                zero_division=0
            )

            if y_prob is not None:
                try:
                    class_metrics['auc_roc'] = roc_auc_score(
                        y_true[:, i],
                        y_prob[:, i]
                    )
                except ValueError:
                    class_metrics['auc_roc'] = 0.0

            # Confusion matrix
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            class_metrics['confusion_matrix'] = cm.tolist()

            metrics['per_class'][f'class_{i}'] = class_metrics

    return metrics


def calculate_interpretability_metrics(model: nn.Module,
                                      data_loader: DataLoader,
                                      device: str = 'cpu') -> Dict:
    """
    Calculate interpretability metrics from Table II in the paper.

    Metrics:
    - Explanation Completeness
    - Explanation Consistency
    - Feature Concentration

    Args:
        model: Trained model with interpretability features
        data_loader: DataLoader for evaluation
        device: Device for computation

    Returns:
        Dictionary with interpretability scores
    """
    from .interpretability import InterpretabilityEvaluator

    evaluator = InterpretabilityEvaluator(model, device)

    metrics = evaluator.evaluate(data_loader)

    return metrics


def evaluate_model(model: nn.Module,
                  data_loader: DataLoader,
                  device: str = 'cpu',
                  threshold: float = 0.5,
                  return_predictions: bool = False) -> Dict:
    """
    Comprehensive model evaluation.

    Args:
        model: Model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device for computation
        threshold: Threshold for binary predictions
        return_predictions: Whether to return predictions

    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    model.to(device)

    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_attention_weights = []

    with torch.no_grad():
        for features, labels in tqdm(data_loader, desc="Evaluating"):
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(features, return_attention=True)
            predictions = outputs['predictions']

            # Store results
            all_probabilities.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Binary predictions
            binary_preds = (predictions > threshold).float()
            all_predictions.append(binary_preds.cpu().numpy())

            # Store attention if available
            if 'attention_weights' in outputs:
                all_attention_weights.append(
                    outputs['attention_weights'].cpu().numpy()
                )

    # Concatenate all batches
    y_prob = np.concatenate(all_probabilities, axis=0)
    y_pred = np.concatenate(all_predictions, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    # Calculate performance metrics
    performance_metrics = calculate_performance_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        threshold=threshold,
        per_class=True
    )

    # Calculate interpretability metrics
    interpretability_metrics = calculate_interpretability_metrics(
        model=model,
        data_loader=data_loader,
        device=device
    )

    results = {
        'performance': performance_metrics,
        'interpretability': interpretability_metrics
    }

    if return_predictions:
        results['predictions'] = {
            'probabilities': y_prob,
            'binary': y_pred,
            'true_labels': y_true
        }

        if all_attention_weights:
            results['predictions']['attention_weights'] = np.concatenate(
                all_attention_weights, axis=0
            )

    return results


def compare_models(models: Dict[str, nn.Module],
                  data_loader: DataLoader,
                  device: str = 'cpu',
                  micronutrient_names: Optional[List[str]] = None) -> Dict:
    """
    Compare multiple models on the same dataset.

    Args:
        models: Dictionary mapping model names to model instances
        data_loader: DataLoader for evaluation
        device: Device for computation
        micronutrient_names: Names of micronutrients

    Returns:
        Dictionary with comparison results
    """
    comparison = {}

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")

        results = evaluate_model(
            model=model,
            data_loader=data_loader,
            device=device
        )

        comparison[model_name] = results

    # Create summary table
    summary = create_comparison_summary(comparison, micronutrient_names)
    comparison['summary'] = summary

    return comparison


def create_comparison_summary(comparison: Dict,
                             micronutrient_names: Optional[List[str]] = None) -> Dict:
    """
    Create a summary comparison table.

    Args:
        comparison: Comparison results dictionary
        micronutrient_names: Names of micronutrients

    Returns:
        Summary dictionary
    """
    summary = {
        'performance': {},
        'interpretability': {}
    }

    # Extract performance metrics for each model
    for model_name, results in comparison.items():
        if model_name == 'summary':
            continue

        perf = results['performance']
        interp = results.get('interpretability', {})

        summary['performance'][model_name] = {
            'accuracy': perf.get('accuracy', 0.0) * 100,
            'precision': perf.get('precision', 0.0) * 100,
            'recall': perf.get('recall', 0.0) * 100,
            'f1_score': perf.get('f1_score', 0.0) * 100,
            'auc_roc': perf.get('auc_roc', 0.0) * 100
        }

        summary['interpretability'][model_name] = {
            'explanation_completeness': interp.get('explanation_completeness', 0.0),
            'explanation_consistency': interp.get('explanation_consistency', 0.0),
            'feature_concentration': interp.get('feature_concentration', 0.0),
            'overall_score': interp.get('overall_score', 0.0)
        }

    return summary


def print_evaluation_results(results: Dict,
                            model_name: str = "Model",
                            micronutrient_names: Optional[List[str]] = None):
    """
    Print evaluation results in a formatted way.

    Args:
        results: Results dictionary from evaluate_model
        model_name: Name of the model
        micronutrient_names: Names of micronutrients
    """
    print(f"\n{'=' * 70}")
    print(f"{model_name} Evaluation Results")
    print(f"{'=' * 70}")

    # Performance metrics
    print("\nPerformance Metrics:")
    print("-" * 70)
    perf = results['performance']
    print(f"  Accuracy:  {perf['accuracy'] * 100:6.2f}%")
    print(f"  Precision: {perf['precision'] * 100:6.2f}%")
    print(f"  Recall:    {perf['recall'] * 100:6.2f}%")
    print(f"  F1-Score:  {perf['f1_score'] * 100:6.2f}%")
    if 'auc_roc' in perf:
        print(f"  AUC-ROC:   {perf['auc_roc'] * 100:6.2f}%")

    # Per-class metrics
    if 'per_class' in perf and micronutrient_names:
        print("\nPer-Micronutrient Performance:")
        print("-" * 70)
        print(f"{'Micronutrient':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
        print("-" * 70)

        for i, (class_key, class_metrics) in enumerate(perf['per_class'].items()):
            name = micronutrient_names[i] if i < len(micronutrient_names) else f"Class {i}"
            print(f"{name:<20} "
                  f"{class_metrics['accuracy'] * 100:9.2f}% "
                  f"{class_metrics['precision'] * 100:9.2f}% "
                  f"{class_metrics['recall'] * 100:9.2f}% "
                  f"{class_metrics['f1_score'] * 100:9.2f}%")

    # Interpretability metrics
    if 'interpretability' in results:
        print("\nInterpretability Metrics:")
        print("-" * 70)
        interp = results['interpretability']
        print(f"  Explanation Completeness: {interp.get('explanation_completeness', 0.0):.3f}")
        print(f"  Explanation Consistency:  {interp.get('explanation_consistency', 0.0):.3f}")
        print(f"  Feature Concentration:    {interp.get('feature_concentration', 0.0):.3f}")
        print(f"  Overall Score:            {interp.get('overall_score', 0.0):.3f}")

    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    print("Testing Evaluation Metrics...")

    # Create dummy data
    n_samples = 100
    n_classes = 12

    y_true = np.random.randint(0, 2, (n_samples, n_classes))
    y_prob = np.random.rand(n_samples, n_classes)
    y_pred = (y_prob > 0.5).astype(int)

    # Test performance metrics
    print("\nTesting performance metrics...")
    metrics = calculate_performance_metrics(y_true, y_pred, y_prob)

    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall: {metrics['recall'] * 100:.2f}%")
    print(f"F1-Score: {metrics['f1_score'] * 100:.2f}%")
    print(f"AUC-ROC: {metrics['auc_roc'] * 100:.2f}%")
    print(f"Per-class metrics calculated for {len(metrics['per_class'])} classes")

    # Test pretty printing
    micronutrient_names = [
        "Vitamin_B12", "Vitamin_D", "Vitamin_B6", "Folate",
        "Iron", "Calcium", "Magnesium", "Zinc",
        "Vitamin_C", "Vitamin_E", "Vitamin_A", "Selenium"
    ]

    dummy_results = {
        'performance': metrics,
        'interpretability': {
            'explanation_completeness': 0.87,
            'explanation_consistency': 0.92,
            'feature_concentration': 0.91,
            'overall_score': 0.90
        }
    }

    print_evaluation_results(dummy_results, "Test Model", micronutrient_names)

    print("\nEvaluation metrics test completed!")
