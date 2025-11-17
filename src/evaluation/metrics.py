"""
Evaluation metrics for nutrient analysis model.

Implements metrics from Table 2 in the paper:
- Top-1 and Top-5 Accuracy (Equation 22)
- Mean Absolute Error (Equation 23)
- Mean Absolute Percentage Error (Equation 24)
- Interpretability metrics (Equations 25-32)
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error
from scipy.stats import pearsonr


def compute_top_k_accuracy(predictions, targets, k=1):
    """
    Compute Top-K accuracy.

    Reference: Equation (22)
    A_k = N_correct_k / N_total

    Args:
        predictions: Model predictions (B, num_classes)
        targets: Ground truth labels (B,)
        k: K value for top-k accuracy

    Returns:
        Top-k accuracy
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # Get top k predictions
    top_k_preds = np.argsort(predictions, axis=1)[:, -k:]

    # Check if true label is in top k
    correct = np.sum([targets[i] in top_k_preds[i] for i in range(len(targets))])

    accuracy = correct / len(targets)
    return accuracy


def compute_accuracy(predictions, targets):
    """Compute top-1 accuracy."""
    return compute_top_k_accuracy(predictions, targets, k=1)


def compute_mae(predictions, targets):
    """
    Compute Mean Absolute Error.

    Reference: Equation (23)
    MAE = (1/n) Σ |y_i - ŷ_i|

    Args:
        predictions: Predicted values
        targets: Ground truth values

    Returns:
        MAE value
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    return mean_absolute_error(targets.flatten(), predictions.flatten())


def compute_mape(predictions, targets, epsilon=1e-8):
    """
    Compute Mean Absolute Percentage Error.

    Reference: Equation (24)
    MAPE = (100/n) Σ |y_i - ŷ_i| / y_i

    Args:
        predictions: Predicted values
        targets: Ground truth values
        epsilon: Small value to avoid division by zero

    Returns:
        MAPE value (percentage)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    targets = targets + epsilon
    mape = np.mean(np.abs((targets - predictions) / targets)) * 100

    return mape


def compute_food_security_metrics(predictions, targets, food_categories):
    """
    Compute accuracy metrics per food security category.

    Args:
        predictions: Model predictions
        targets: Ground truth labels
        food_categories: Food security category labels

    Returns:
        Dictionary with per-category accuracies
    """
    category_accuracies = {}

    # Food security categories
    categories = {
        0: 'staple_foods',
        1: 'affordable_proteins',
        2: 'accessible_produce',
        3: 'processed_foods',
        4: 'specialty_foods'
    }

    for cat_id, cat_name in categories.items():
        # Get indices for this category
        cat_mask = food_categories == cat_id

        if cat_mask.sum() > 0:
            cat_preds = predictions[cat_mask]
            cat_targets = targets[cat_mask]

            accuracy = compute_accuracy(cat_preds, cat_targets)
            category_accuracies[cat_name] = accuracy

    return category_accuracies


# Interpretability Metrics

def compute_explanation_quality(model_explanations, expert_annotations):
    """
    Compute explanation quality metric.

    Reference: Equation (25)
    Q_exp = (1/n) Σ corr(E_model,i, E_expert,i)

    Args:
        model_explanations: Model-generated explanations
        expert_annotations: Expert annotations

    Returns:
        Explanation quality score
    """
    correlations = []

    for model_exp, expert_exp in zip(model_explanations, expert_annotations):
        corr, _ = pearsonr(model_exp.flatten(), expert_exp.flatten())
        correlations.append(corr)

    return np.mean(correlations)


def compute_prediction_confidence(confidences, accuracies, num_bins=10):
    """
    Compute prediction confidence calibration.

    Reference: Equation (26)
    P_conf = 1 - (1/M) Σ |conf_i - acc_i| × n_i/N

    Args:
        confidences: Model confidence scores
        accuracies: Actual accuracies (binary correct/incorrect)
        num_bins: Number of confidence bins

    Returns:
        Confidence calibration score
    """
    bins = np.linspace(0, 1, num_bins + 1)
    bin_errors = []

    for i in range(num_bins):
        # Get samples in this confidence bin
        bin_mask = (confidences >= bins[i]) & (confidences < bins[i + 1])

        if bin_mask.sum() > 0:
            bin_conf = confidences[bin_mask].mean()
            bin_acc = accuracies[bin_mask].mean()
            bin_n = bin_mask.sum()

            bin_error = np.abs(bin_conf - bin_acc) * (bin_n / len(confidences))
            bin_errors.append(bin_error)

    calibration = 1 - np.sum(bin_errors)
    return max(0, min(1, calibration))


def compute_feature_attribution(model_rankings, expert_rankings, k=10):
    """
    Compute feature attribution accuracy.

    Reference: Equation (27)
    F_attr = (1/k) Σ exp(-|rank_model(fj) - rank_expert(fj)|/k)

    Args:
        model_rankings: Model's feature importance rankings
        expert_rankings: Expert's feature rankings
        k: Number of features to evaluate

    Returns:
        Feature attribution score
    """
    scores = []

    for model_rank, expert_rank in zip(model_rankings, expert_rankings):
        rank_diff = np.abs(model_rank - expert_rank)
        score = np.exp(-rank_diff / k)
        scores.append(score)

    return np.mean(scores)


def compute_cultural_adaptation(comprehension_scores, cultural_groups):
    """
    Compute cultural adaptation metric.

    Reference: Equation (28)
    A_cultural-exp = (1/G) Σ (correct_responses_g / total_responses_g)

    Args:
        comprehension_scores: Comprehension scores per group
        cultural_groups: Number of cultural groups

    Returns:
        Cultural adaptation score
    """
    group_scores = []

    for group_id in range(cultural_groups):
        group_mask = comprehension_scores['group'] == group_id
        if group_mask.sum() > 0:
            correct = comprehension_scores['correct'][group_mask].sum()
            total = group_mask.sum()
            group_scores.append(correct / total)

    return np.mean(group_scores) if group_scores else 0.0


def compute_localisation_score(attention_regions, expert_regions):
    """
    Compute localisation score (IoU).

    Reference: Equation (29)
    L_score = |A_model ∩ A_expert| / |A_model ∪ A_expert|

    Args:
        attention_regions: Model attention regions
        expert_regions: Expert-annotated regions

    Returns:
        Localisation score (IoU)
    """
    intersection = np.logical_and(attention_regions, expert_regions).sum()
    union = np.logical_or(attention_regions, expert_regions).sum()

    if union == 0:
        return 0.0

    return intersection / union


def compute_feature_consistency(feature_importances, num_categories):
    """
    Compute feature consistency across categories.

    Reference: Equation (30)
    F_consistency = (1/C) Σ (1 - σ_c/μ_c)

    Args:
        feature_importances: Feature importance scores per category
        num_categories: Number of categories

    Returns:
        Feature consistency score
    """
    consistency_scores = []

    for cat in range(num_categories):
        cat_importances = feature_importances[cat]

        if len(cat_importances) > 0:
            mean = np.mean(cat_importances)
            std = np.std(cat_importances)

            if mean > 0:
                consistency = 1 - (std / mean)
                consistency_scores.append(max(0, consistency))

    return np.mean(consistency_scores) if consistency_scores else 0.0


class MetricsCalculator:
    """Comprehensive metrics calculator for model evaluation."""

    def __init__(self):
        self.metrics = {}

    def compute_all_metrics(self, predictions, targets, **kwargs):
        """
        Compute all evaluation metrics.

        Args:
            predictions: Dictionary with model predictions
            targets: Dictionary with ground truth
            **kwargs: Additional data for metric computation

        Returns:
            Dictionary with all metrics
        """
        metrics = {}

        # Food recognition metrics
        if 'food_logits' in predictions and 'label' in targets:
            metrics['top1_accuracy'] = compute_accuracy(
                predictions['food_logits'],
                targets['label']
            )
            metrics['top5_accuracy'] = compute_top_k_accuracy(
                predictions['food_logits'],
                targets['label'],
                k=5
            )

        # Nutrient estimation metrics
        if 'macronutrients' in predictions and 'macronutrients' in targets:
            metrics['macro_mae'] = compute_mae(
                predictions['macronutrients'],
                targets['macronutrients']
            )
            metrics['macro_mape'] = compute_mape(
                predictions['macronutrients'],
                targets['macronutrients']
            )

        if 'micronutrients' in predictions and 'micronutrients' in targets:
            metrics['micro_mae'] = compute_mae(
                predictions['micronutrients'],
                targets['micronutrients']
            )

        if 'portion' in predictions and 'portion' in targets:
            metrics['portion_mae'] = compute_mae(
                predictions['portion'],
                targets['portion']
            )

        # Food security category metrics
        if 'food_security_category' in kwargs:
            metrics['food_security'] = compute_food_security_metrics(
                predictions['food_logits'],
                targets['label'],
                kwargs['food_security_category']
            )

        return metrics


if __name__ == "__main__":
    print("Testing metrics...")

    # Test Top-K Accuracy
    preds = np.random.rand(100, 500)
    targets = np.random.randint(0, 500, 100)

    top1 = compute_top_k_accuracy(preds, targets, k=1)
    top5 = compute_top_k_accuracy(preds, targets, k=5)

    print(f"Top-1 Accuracy: {top1:.4f}")
    print(f"Top-5 Accuracy: {top5:.4f}")

    # Test MAE and MAPE
    pred_nutrients = np.random.rand(100, 3) * 50
    true_nutrients = np.random.rand(100, 3) * 50

    mae = compute_mae(pred_nutrients, true_nutrients)
    mape = compute_mape(pred_nutrients, true_nutrients)

    print(f"\nMAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")

    print("\nMetrics test completed successfully!")
