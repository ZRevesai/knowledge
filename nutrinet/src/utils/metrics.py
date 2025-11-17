"""
Evaluation metrics for NUTRINET.

Implements metrics from Section 4 of the paper:
- Mean Absolute Error (MAE) for nutrient prediction
- Area Under ROC Curve (AUC) for deficiency risk assessment
- F1-score for personalized recommendation quality
- Interpretability metrics (fidelity, conciseness, clinical relevance)
"""

import torch
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from typing import Dict, List, Optional, Tuple


def compute_nutrient_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute metrics for nutrient composition prediction task.

    Primary metric: Mean Absolute Error (MAE)

    Args:
        y_true: True nutrient values [n_samples, n_nutrients]
        y_pred: Predicted nutrient values [n_samples, n_nutrients]

    Returns:
        Dictionary of metrics
    """
    # Ensure numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # MAE (primary metric from paper)
    mae = mean_absolute_error(y_true, y_pred)

    # Additional metrics
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'mape': float(mape),
        'r2': float(r2),
    }


def compute_deficiency_risk_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute metrics for deficiency risk assessment task.

    Primary metric: Area Under ROC Curve (AUC)

    Args:
        y_true: True deficiency labels [n_samples] (binary: 0 or 1)
        y_pred: Predicted deficiency labels [n_samples] (binary: 0 or 1)
        y_pred_proba: Predicted probabilities [n_samples] (optional, for AUC)

    Returns:
        Dictionary of metrics
    """
    # Ensure numpy arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Convert to binary if needed
    if y_pred.max() > 1:
        y_pred = (y_pred > 0.5).astype(int)

    # AUC (primary metric from paper)
    if y_pred_proba is not None:
        auc = roc_auc_score(y_true, y_pred_proba)
    else:
        # Use predictions as probabilities
        auc = roc_auc_score(y_true, y_pred)

    # Additional classification metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        'auc': float(auc),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'specificity': float(specificity),
    }


def compute_recommendation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_k: int = 5,
) -> Dict[str, float]:
    """
    Compute metrics for personalized recommendation quality.

    Primary metric: F1-score

    Args:
        y_true: True recommendations [n_samples, n_items] (binary)
        y_pred: Predicted recommendations [n_samples, n_items] (binary or scores)
        top_k: Number of top recommendations to consider

    Returns:
        Dictionary of metrics
    """
    # Ensure numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # If predictions are scores, convert to top-k binary
    if y_pred.max() > 1 or y_pred.dtype == np.float32 or y_pred.dtype == np.float64:
        y_pred_binary = np.zeros_like(y_pred)
        for i in range(y_pred.shape[0]):
            top_indices = np.argsort(y_pred[i])[-top_k:]
            y_pred_binary[i, top_indices] = 1
        y_pred = y_pred_binary

    # Flatten for metric computation
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # F1-score (primary metric from paper)
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)

    # Precision and Recall
    precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)

    # Precision@k and Recall@k
    precision_at_k = []
    recall_at_k = []

    for i in range(y_true.shape[0]):
        true_items = set(np.where(y_true[i] == 1)[0])
        pred_items = set(np.where(y_pred[i] == 1)[0])

        if len(pred_items) > 0:
            prec_k = len(true_items & pred_items) / len(pred_items)
        else:
            prec_k = 0.0

        if len(true_items) > 0:
            rec_k = len(true_items & pred_items) / len(true_items)
        else:
            rec_k = 0.0

        precision_at_k.append(prec_k)
        recall_at_k.append(rec_k)

    return {
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        f'precision@{top_k}': float(np.mean(precision_at_k)),
        f'recall@{top_k}': float(np.mean(recall_at_k)),
    }


def compute_interpretability_metrics(
    explanations: List[Dict],
    ground_truth: Optional[List[Dict]] = None,
) -> Dict[str, float]:
    """
    Compute interpretability metrics.

    Metrics from Table 3 in the paper:
    - Explanation Fidelity: How well explanation matches model's actual decision
    - Explanation Conciseness: Average number of words in explanation
    - Clinical Relevance Score: Domain expert rating

    Args:
        explanations: List of explanation dictionaries from model
        ground_truth: Optional ground truth explanations for comparison

    Returns:
        Dictionary of interpretability metrics
    """
    if not explanations:
        return {
            'explanation_fidelity': 0.0,
            'explanation_conciseness': 0.0,
            'clinical_relevance': 0.0,
        }

    # Explanation Fidelity
    # Measure: Overlap between important features in explanation and actual gradients
    fidelities = []
    for exp in explanations:
        if 'node_importance' in exp and 'significant_nodes' in exp:
            # Compute fidelity as correlation between importance and gradient
            node_importance = np.array(exp['node_importance'])
            significant_nodes = exp['significant_nodes']

            if len(significant_nodes) > 0:
                # Fidelity = fraction of top-k important nodes that are significant
                k = min(10, len(node_importance))
                top_k_nodes = np.argsort(node_importance)[-k:]
                overlap = len(set(top_k_nodes) & set(significant_nodes))
                fidelity = overlap / k
                fidelities.append(fidelity)

    explanation_fidelity = np.mean(fidelities) if fidelities else 0.0

    # Explanation Conciseness
    # Measure: Average word count in summary
    conciseness = []
    for exp in explanations:
        if 'summary' in exp:
            word_count = len(exp['summary'].split())
            conciseness.append(word_count)

    explanation_conciseness = np.mean(conciseness) if conciseness else 0.0

    # Clinical Relevance Score
    # Measure: Percentage of explanations that identify clinically known pathways
    relevance_scores = []
    for exp in explanations:
        if 'critical_paths' in exp:
            # Check if paths contain known nutrient interactions
            # For now, use number of paths as proxy
            num_paths = len(exp['critical_paths'])
            relevance = min(1.0, num_paths / 5)  # Normalize to [0, 1]
            relevance_scores.append(relevance)

    clinical_relevance = np.mean(relevance_scores) if relevance_scores else 0.0

    return {
        'explanation_fidelity': float(explanation_fidelity),
        'explanation_conciseness': float(explanation_conciseness),
        'clinical_relevance': float(clinical_relevance),
    }


def compute_all_metrics(
    task_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    explanations: Optional[List[Dict]] = None,
) -> Dict[str, float]:
    """
    Compute all relevant metrics for a task.

    Args:
        task_type: Type of task ('nutrient_prediction', 'deficiency_risk', or 'recommendation')
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        explanations: Explanation dictionaries (optional)

    Returns:
        Dictionary of all computed metrics
    """
    metrics = {}

    if task_type == 'nutrient_prediction':
        metrics.update(compute_nutrient_prediction_metrics(y_true, y_pred))

    elif task_type == 'deficiency_risk':
        metrics.update(compute_deficiency_risk_metrics(y_true, y_pred, y_pred_proba))

    elif task_type == 'recommendation':
        metrics.update(compute_recommendation_metrics(y_true, y_pred))

    else:
        raise ValueError(f"Unknown task type: {task_type}")

    # Add interpretability metrics if explanations provided
    if explanations is not None:
        interp_metrics = compute_interpretability_metrics(explanations)
        metrics.update(interp_metrics)

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Print metrics in a formatted table.

    Args:
        metrics: Dictionary of metrics
        title: Title for the table
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")

    for metric_name, value in metrics.items():
        # Format metric name
        name_formatted = metric_name.replace('_', ' ').title()

        # Format value
        if isinstance(value, float):
            if value < 0.01:
                value_str = f"{value:.4f}"
            elif value < 1:
                value_str = f"{value:.3f}"
            elif value < 100:
                value_str = f"{value:.2f}"
            else:
                value_str = f"{value:.1f}"
        else:
            value_str = str(value)

        print(f"{name_formatted:.<40} {value_str:>8}")

    print(f"{'='*50}\n")
