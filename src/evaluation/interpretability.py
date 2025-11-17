"""
Interpretability evaluation metrics for KGNN.

Implements metrics from Table II in the paper:
- Explanation Completeness
- Explanation Consistency
- Feature Concentration
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from tqdm import tqdm


def explanation_completeness(predictions: np.ndarray,
                             feature_importance: np.ndarray,
                             features: np.ndarray) -> float:
    """
    Calculate explanation completeness.

    Measures the proportion of prediction variance explained by
    the provided feature importance scores.

    Args:
        predictions: Model predictions (n_samples, n_outputs)
        feature_importance: Feature importance scores (n_samples, n_features)
        features: Input features (n_samples, n_features)

    Returns:
        Completeness score (0 to 1)
    """
    # Compute weighted feature contributions
    weighted_contributions = features * feature_importance

    # Calculate correlation between weighted contributions and predictions
    completeness_scores = []

    for i in range(predictions.shape[1]):
        # Sum of weighted contributions per sample
        total_contribution = np.sum(weighted_contributions, axis=1)

        # Correlation with predictions
        if np.std(total_contribution) > 0 and np.std(predictions[:, i]) > 0:
            corr = np.corrcoef(total_contribution, predictions[:, i])[0, 1]
            completeness_scores.append(abs(corr))
        else:
            completeness_scores.append(0.0)

    return np.mean(completeness_scores)


def explanation_consistency(feature_importance_list: List[np.ndarray],
                           similarity_threshold: float = 0.8) -> float:
    """
    Calculate explanation consistency.

    Measures similarity of explanations for similar input samples.

    Args:
        feature_importance_list: List of feature importance arrays for similar samples
        similarity_threshold: Threshold for considering samples similar

    Returns:
        Consistency score (0 to 1)
    """
    if len(feature_importance_list) < 2:
        return 1.0

    consistency_scores = []

    # Compare pairs of similar samples
    for i in range(len(feature_importance_list) - 1):
        for j in range(i + 1, min(i + 10, len(feature_importance_list))):  # Compare with next 10 samples
            imp1 = feature_importance_list[i]
            imp2 = feature_importance_list[j]

            # Calculate similarity using cosine similarity
            if np.linalg.norm(imp1) > 0 and np.linalg.norm(imp2) > 0:
                similarity = 1 - cosine(imp1, imp2)
                consistency_scores.append(max(0, similarity))

    if not consistency_scores:
        return 1.0

    return np.mean(consistency_scores)


def feature_concentration(feature_importance: np.ndarray,
                         top_k: int = 10) -> float:
    """
    Calculate feature concentration.

    Measures sparsity of feature importance distribution (Gini coefficient).
    Higher concentration means more focused, interpretable explanations.

    Args:
        feature_importance: Feature importance scores (n_samples, n_features)
        top_k: Number of top features to consider

    Returns:
        Concentration score (0 to 1)
    """
    concentration_scores = []

    for importance in feature_importance:
        # Sort importance scores
        sorted_importance = np.sort(np.abs(importance))[::-1]

        # Calculate concentration using top-k approach
        top_k_sum = np.sum(sorted_importance[:top_k])
        total_sum = np.sum(sorted_importance)

        if total_sum > 0:
            concentration = top_k_sum / total_sum
            concentration_scores.append(concentration)

    return np.mean(concentration_scores)


class InterpretabilityEvaluator:
    """
    Comprehensive interpretability evaluator for KGNN and baseline models.

    Calculates all interpretability metrics from Table II.
    """

    def __init__(self,
                 model: nn.Module,
                 device: str = 'cpu'):
        """
        Initialize evaluator.

        Args:
            model: Model to evaluate
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def extract_feature_importance(self,
                                   features: torch.Tensor) -> np.ndarray:
        """
        Extract feature importance from model.

        Args:
            features: Input features

        Returns:
            Feature importance array
        """
        with torch.no_grad():
            # Get model outputs with attention
            outputs = self.model(features, return_attention=True)

            # Extract attention weights as feature importance proxy
            if 'attention_weights' in outputs:
                importance = outputs['attention_weights'].cpu().numpy()
            else:
                # Fallback: use gradient-based importance
                features.requires_grad = True
                outputs = self.model(features)
                predictions = outputs['predictions']

                # Compute gradients
                importance_list = []
                for i in range(predictions.shape[1]):
                    if features.grad is not None:
                        features.grad.zero_()

                    predictions[:, i].sum().backward(retain_graph=True)
                    importance_list.append(features.grad.abs().cpu().numpy())

                importance = np.mean(importance_list, axis=0)

        return importance

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate interpretability metrics on a dataset.

        Args:
            data_loader: DataLoader for evaluation

        Returns:
            Dictionary with interpretability metrics
        """
        all_predictions = []
        all_features = []
        all_importance = []

        # Collect data
        for features, _ in tqdm(data_loader, desc="Computing interpretability"):
            features = features.to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.model(features, return_attention=True)
                predictions = outputs['predictions']

            # Get feature importance
            importance = self.extract_feature_importance(features)

            all_predictions.append(predictions.cpu().numpy())
            all_features.append(features.cpu().numpy())
            all_importance.append(importance)

        # Concatenate
        predictions = np.concatenate(all_predictions, axis=0)
        features = np.concatenate(all_features, axis=0)
        importance = np.concatenate(all_importance, axis=0)

        # Calculate metrics
        metrics = {}

        # 1. Explanation Completeness
        metrics['explanation_completeness'] = explanation_completeness(
            predictions, importance, features
        )

        # 2. Explanation Consistency
        metrics['explanation_consistency'] = explanation_consistency(
            [importance[i] for i in range(len(importance))]
        )

        # 3. Feature Concentration
        metrics['feature_concentration'] = feature_concentration(importance)

        # 4. Overall Score (average of three metrics)
        metrics['overall_score'] = np.mean([
            metrics['explanation_completeness'],
            metrics['explanation_consistency'],
            metrics['feature_concentration']
        ])

        return metrics

    def explain_single_prediction(self,
                                  features: torch.Tensor,
                                  feature_names: Optional[List[str]] = None,
                                  top_k: int = 10) -> Dict:
        """
        Generate detailed explanation for a single prediction.

        Args:
            features: Input features (1, n_features)
            feature_names: Names of features
            top_k: Number of top features to include

        Returns:
            Explanation dictionary
        """
        with torch.no_grad():
            # Get model outputs
            outputs = self.model(features, return_attention=True, return_hidden=True)

            # Extract importance
            importance = self.extract_feature_importance(features)

            # Get top features
            top_indices = np.argsort(np.abs(importance[0]))[-top_k:][::-1]

            explanation = {
                'predictions': outputs['predictions'][0].cpu().numpy(),
                'top_features': []
            }

            for idx in top_indices:
                feature_info = {
                    'index': int(idx),
                    'importance': float(importance[0, idx]),
                    'value': float(features[0, idx].cpu().numpy())
                }

                if feature_names and idx < len(feature_names):
                    feature_info['name'] = feature_names[idx]

                explanation['top_features'].append(feature_info)

            # Add attention weights if available
            if 'attention_weights' in outputs:
                explanation['attention_weights'] = outputs['attention_weights'][0].cpu().numpy()

            return explanation

    def evaluate_attention_quality(self,
                                   data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate quality of attention mechanisms.

        Args:
            data_loader: DataLoader for evaluation

        Returns:
            Dictionary with attention quality metrics
        """
        all_attention = []
        all_predictions = []

        # Collect attention weights
        for features, _ in data_loader:
            features = features.to(self.device)

            with torch.no_grad():
                outputs = self.model(features, return_attention=True)

                if 'attention_weights' in outputs:
                    attention = outputs['attention_weights'].cpu().numpy()
                    predictions = outputs['predictions'].cpu().numpy()

                    all_attention.append(attention)
                    all_predictions.append(predictions)

        if not all_attention:
            return {}

        attention = np.concatenate(all_attention, axis=0)
        predictions = np.concatenate(all_predictions, axis=0)

        # Calculate attention quality metrics
        metrics = {}

        # 1. Attention Sparsity (entropy-based)
        entropies = []
        for attn in attention:
            # Normalize
            attn_norm = attn / (attn.sum() + 1e-8)
            # Calculate entropy
            entropy = -np.sum(attn_norm * np.log(attn_norm + 1e-8))
            entropies.append(entropy)

        metrics['attention_sparsity'] = 1.0 - (np.mean(entropies) / np.log(attention.shape[1]))

        # 2. Attention-Prediction Correlation
        correlations = []
        for i in range(predictions.shape[1]):
            pred = predictions[:, i]
            attn_sum = attention.sum(axis=1)

            if np.std(pred) > 0 and np.std(attn_sum) > 0:
                corr = np.corrcoef(pred, attn_sum)[0, 1]
                correlations.append(abs(corr))

        if correlations:
            metrics['attention_prediction_correlation'] = np.mean(correlations)
        else:
            metrics['attention_prediction_correlation'] = 0.0

        return metrics


def compare_interpretability(models: Dict[str, nn.Module],
                            data_loader: DataLoader,
                            device: str = 'cpu') -> Dict:
    """
    Compare interpretability of multiple models.

    Args:
        models: Dictionary of model name to model instance
        data_loader: DataLoader for evaluation
        device: Device for computation

    Returns:
        Comparison results
    """
    results = {}

    for model_name, model in models.items():
        print(f"\nEvaluating interpretability of {model_name}...")

        evaluator = InterpretabilityEvaluator(model, device)
        metrics = evaluator.evaluate(data_loader)

        results[model_name] = metrics

    return results


if __name__ == "__main__":
    print("Testing Interpretability Evaluator...")

    # Create dummy data
    n_samples = 100
    n_features = 105
    n_outputs = 12

    predictions = np.random.rand(n_samples, n_outputs)
    features = np.random.randn(n_samples, n_features)
    importance = np.random.rand(n_samples, n_features)

    # Test explanation completeness
    print("\n1. Testing Explanation Completeness...")
    completeness = explanation_completeness(predictions, importance, features)
    print(f"   Completeness: {completeness:.3f}")

    # Test explanation consistency
    print("\n2. Testing Explanation Consistency...")
    importance_list = [importance[i] for i in range(min(50, n_samples))]
    consistency = explanation_consistency(importance_list)
    print(f"   Consistency: {consistency:.3f}")

    # Test feature concentration
    print("\n3. Testing Feature Concentration...")
    concentration = feature_concentration(importance)
    print(f"   Concentration: {concentration:.3f}")

    # Test with dummy model
    print("\n4. Testing InterpretabilityEvaluator...")

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.ReLU(),
                nn.Linear(64, n_outputs),
                nn.Sigmoid()
            )

        def forward(self, x, return_attention=False, return_hidden=False):
            pred = self.fc(x)
            output = {'predictions': pred}
            if return_attention:
                output['attention_weights'] = torch.rand(x.size(0), 64)
            return output

    model = DummyModel()
    evaluator = InterpretabilityEvaluator(model)

    # Create dummy dataloader
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(torch.FloatTensor(features), torch.FloatTensor(predictions))
    loader = DataLoader(dataset, batch_size=32)

    metrics = evaluator.evaluate(loader)
    print(f"   Explanation Completeness: {metrics['explanation_completeness']:.3f}")
    print(f"   Explanation Consistency: {metrics['explanation_consistency']:.3f}")
    print(f"   Feature Concentration: {metrics['feature_concentration']:.3f}")
    print(f"   Overall Score: {metrics['overall_score']:.3f}")

    print("\nInterpretability evaluator test completed!")
