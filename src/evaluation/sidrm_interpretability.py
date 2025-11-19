"""
SHAP-based Interpretability Framework for SIDRM.

Implements the interpretability metrics from the paper:
- SHAP Stability Score (Equation 10)
- Attention Consistency (Equation 11)
- Feature Ranking Correlation (Equation 12)

Reference: "Smart Interpretable Dietary Recommender Model for Vulnerable Populations"
by Zvinodashe Revesai and Okuthe P. Kogeda (2025)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import shap
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon
import warnings


class SHAPInterpreter:
    """
    SHAP-based interpretability framework for SIDRM.

    Implements SHAP value computation (Equation 9) and SHAP Stability Score (Equation 10):

    φ_i = Σ_{S⊆N\{i}} |S|!(|N|-|S|-1)!/|N|! [f(S∪{i}) - f(S)]

    S_SHAP = 1 - (1/r)Σ ||φ_i - φ̄||_2 / ||φ̄||_2
    """

    def __init__(self,
                 model: nn.Module,
                 background_data: torch.Tensor,
                 num_bootstrap_samples: int = 100,
                 device: str = 'cpu'):
        """
        Initialize SHAP interpreter.

        Args:
            model: SIDRM model to interpret
            background_data: Background dataset for SHAP (batch_size, input_dim)
            num_bootstrap_samples: Number of bootstrap samples for stability (r in Equation 10)
            device: Device to run computations on
        """
        self.model = model
        self.background_data = background_data.cpu().numpy()
        self.num_bootstrap_samples = num_bootstrap_samples
        self.device = device

        # Create SHAP explainer
        self._create_explainer()

    def _create_explainer(self):
        """Create SHAP DeepExplainer for the model."""
        # Wrap model forward pass for SHAP
        def model_forward(x):
            """Model wrapper for SHAP."""
            self.model.eval()
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x).to(self.device)
                outputs = self.model(x_tensor, return_attention=False)
                # Return nutrient recommendations for explanation
                return outputs['nutrient_recommendations'].cpu().numpy()

        # Use KernelExplainer for model-agnostic explanations
        self.explainer = shap.KernelExplainer(
            model_forward,
            self.background_data,
            link="identity"
        )

    def compute_shap_values(self,
                           x: torch.Tensor,
                           population_indices: Optional[torch.Tensor] = None,
                           nsamples: int = 100) -> np.ndarray:
        """
        Compute SHAP values for input samples.

        Implements Equation (9) from the paper:
        φ_i = Σ_{S⊆N\{i}} |S|!(|N|-|S|-1)!/|N|! [f(S∪{i}) - f(S)]

        Args:
            x: Input tensor (batch_size, input_dim)
            population_indices: Population category for each sample
            nsamples: Number of samples for SHAP computation

        Returns:
            SHAP values array of shape (batch_size, input_dim, num_nutrients)
        """
        x_np = x.cpu().numpy()

        # Compute SHAP values
        shap_values = self.explainer.shap_values(
            x_np,
            nsamples=nsamples,
            silent=True
        )

        # Handle both single and multi-output cases
        if isinstance(shap_values, list):
            # Multi-output: stack along last dimension
            shap_values = np.stack(shap_values, axis=-1)
        elif len(shap_values.shape) == 2:
            # Single output: add dimension
            shap_values = shap_values[..., np.newaxis]

        return shap_values

    def compute_shap_stability(self,
                              x: torch.Tensor,
                              population_indices: Optional[torch.Tensor] = None,
                              nsamples: int = 100) -> float:
        """
        Compute SHAP Stability Score.

        Implements Equation (10) from the paper:
        S_SHAP = 1 - (1/r)Σ ||φ_i - φ̄||_2 / ||φ̄||_2

        where:
        - r: number of bootstrap samples
        - φ_i: SHAP values for bootstrap sample i
        - φ̄: mean SHAP values across bootstrap samples

        Args:
            x: Input tensor (batch_size, input_dim)
            population_indices: Population category for each sample
            nsamples: Number of samples for each SHAP computation

        Returns:
            SHAP stability score (higher is better, max=1.0)
        """
        batch_size = x.size(0)

        # Collect SHAP values from bootstrap samples
        bootstrap_shap_values = []

        for i in range(self.num_bootstrap_samples):
            # Bootstrap sample (with replacement)
            indices = torch.randint(0, batch_size, (batch_size,))
            x_bootstrap = x[indices]

            # Compute SHAP values
            shap_vals = self.compute_shap_values(
                x_bootstrap,
                population_indices[indices] if population_indices is not None else None,
                nsamples=nsamples
            )

            bootstrap_shap_values.append(shap_vals)

        # Convert to array: (num_bootstrap, batch_size, input_dim, num_nutrients)
        bootstrap_shap_values = np.stack(bootstrap_shap_values, axis=0)

        # Compute mean SHAP values: φ̄
        mean_shap = bootstrap_shap_values.mean(axis=0)

        # Compute stability score
        deviations = []
        for i in range(self.num_bootstrap_samples):
            # ||φ_i - φ̄||_2 / ||φ̄||_2
            diff = bootstrap_shap_values[i] - mean_shap
            norm_diff = np.linalg.norm(diff)
            norm_mean = np.linalg.norm(mean_shap)

            if norm_mean > 0:
                deviations.append(norm_diff / norm_mean)

        # S_SHAP = 1 - (1/r)Σ deviations
        stability_score = 1.0 - np.mean(deviations)

        return float(stability_score)

    def get_top_features(self,
                        shap_values: np.ndarray,
                        feature_names: Optional[List[str]] = None,
                        top_k: int = 10,
                        nutrient_idx: int = 0) -> Dict:
        """
        Get top-k most important features for a specific nutrient.

        Args:
            shap_values: SHAP values (batch_size, input_dim, num_nutrients)
            feature_names: Names of input features
            top_k: Number of top features to return
            nutrient_idx: Index of nutrient to analyze

        Returns:
            Dictionary with top feature information
        """
        # Average SHAP values across batch for specific nutrient
        nutrient_shap = shap_values[:, :, nutrient_idx].mean(axis=0)

        # Get top-k feature indices by absolute SHAP value
        top_indices = np.argsort(np.abs(nutrient_shap))[-top_k:][::-1]

        # Prepare results
        top_features = {
            'feature_indices': top_indices.tolist(),
            'shap_values': nutrient_shap[top_indices].tolist(),
            'importance_scores': np.abs(nutrient_shap[top_indices]).tolist()
        }

        if feature_names is not None:
            top_features['feature_names'] = [feature_names[i] for i in top_indices]

        return top_features


class AttentionConsistencyMetric:
    """
    Attention Consistency metric for interpretability.

    Implements Equation (11) from the paper:
    C_attention = (1/h)Σ(1 - JS(A_j1, A_j2))

    where:
    - h: number of attention heads
    - JS: Jensen-Shannon divergence between attention maps
    """

    @staticmethod
    def compute_consistency(attention_weights_list: List[torch.Tensor]) -> float:
        """
        Compute attention consistency across heads and layers.

        Args:
            attention_weights_list: List of attention weight tensors from each layer
                                   Each tensor: (batch, num_heads, seq_len, seq_len)

        Returns:
            Attention consistency score (higher is better, range [0, 1])
        """
        if not attention_weights_list:
            return 0.0

        consistency_scores = []

        for attn_weights in attention_weights_list:
            # attn_weights: (batch, num_heads, seq, seq)
            batch_size, num_heads, seq_len, _ = attn_weights.shape

            # Compute consistency across heads
            for i in range(num_heads - 1):
                for j in range(i + 1, num_heads):
                    # Get attention distributions for heads i and j
                    attn_i = attn_weights[:, i, :, :].flatten(1).cpu().numpy()
                    attn_j = attn_weights[:, j, :, :].flatten(1).cpu().numpy()

                    # Compute Jensen-Shannon divergence for each sample
                    for b in range(batch_size):
                        # Normalize to probability distributions
                        p = attn_i[b] / (attn_i[b].sum() + 1e-10)
                        q = attn_j[b] / (attn_j[b].sum() + 1e-10)

                        # Jensen-Shannon divergence
                        js_div = jensenshannon(p, q)

                        # Consistency = 1 - JS divergence
                        consistency_scores.append(1.0 - js_div)

        # Average consistency across all head pairs and layers
        if consistency_scores:
            return float(np.mean(consistency_scores))
        else:
            return 0.0

    @staticmethod
    def compute_layer_consistency(attention_weights_list: List[torch.Tensor]) -> List[float]:
        """
        Compute attention consistency for each layer separately.

        Args:
            attention_weights_list: List of attention weight tensors from each layer

        Returns:
            List of consistency scores, one per layer
        """
        layer_consistencies = []

        for attn_weights in attention_weights_list:
            consistency = AttentionConsistencyMetric.compute_consistency([attn_weights])
            layer_consistencies.append(consistency)

        return layer_consistencies


class FeatureRankingCorrelation:
    """
    Feature Ranking Correlation metric for interpretability.

    Implements Equation (12) from the paper:
    C_ranking = 1 - (6Σd_i²) / (m(m²-1))

    where:
    - d_i: rank difference for feature i between two rankings
    - m: total number of features
    """

    @staticmethod
    def compute_correlation(ranking1: np.ndarray,
                          ranking2: np.ndarray,
                          method: str = 'spearman') -> float:
        """
        Compute correlation between two feature rankings.

        Implements Equation (12) using Spearman's rank correlation.

        Args:
            ranking1: First feature ranking (importance scores)
            ranking2: Second feature ranking (importance scores)
            method: Correlation method ('spearman' or 'kendall')

        Returns:
            Correlation coefficient (range [-1, 1], higher is better)
        """
        if method == 'spearman':
            correlation, _ = spearmanr(ranking1, ranking2)
        else:
            # Manual implementation of Equation (12)
            m = len(ranking1)

            # Get ranks
            ranks1 = np.argsort(np.argsort(ranking1))
            ranks2 = np.argsort(np.argsort(ranking2))

            # Compute rank differences
            d = ranks1 - ranks2
            d_squared = np.sum(d ** 2)

            # Spearman correlation: ρ = 1 - (6Σd²)/(m(m²-1))
            correlation = 1.0 - (6 * d_squared) / (m * (m**2 - 1))

        return float(correlation)


class SIDRMInterpretabilityEvaluator:
    """
    Comprehensive interpretability evaluator for SIDRM.

    Combines all three interpretability metrics:
    1. SHAP Stability Score (Equation 10)
    2. Attention Consistency (Equation 11)
    3. Feature Ranking Correlation (Equation 12)
    """

    def __init__(self,
                 model: nn.Module,
                 background_data: torch.Tensor,
                 feature_names: Optional[List[str]] = None,
                 num_bootstrap_samples: int = 100,
                 device: str = 'cpu'):
        """
        Initialize interpretability evaluator.

        Args:
            model: SIDRM model
            background_data: Background dataset for SHAP
            feature_names: Names of input features
            num_bootstrap_samples: Number of bootstrap samples for SHAP stability
            device: Device to run computations on
        """
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names
        self.device = device

        # Initialize SHAP interpreter
        self.shap_interpreter = SHAPInterpreter(
            model=model,
            background_data=background_data,
            num_bootstrap_samples=num_bootstrap_samples,
            device=device
        )

    def evaluate_interpretability(self,
                                  x: torch.Tensor,
                                  population_indices: Optional[torch.Tensor] = None,
                                  compute_shap: bool = True,
                                  shap_nsamples: int = 100) -> Dict:
        """
        Compute all interpretability metrics.

        Args:
            x: Input tensor (batch_size, input_dim)
            population_indices: Population category for each sample
            compute_shap: Whether to compute SHAP-based metrics (can be slow)
            shap_nsamples: Number of samples for SHAP computation

        Returns:
            Dictionary with all interpretability metrics
        """
        results = {}

        # 1. SHAP Stability Score (Equation 10)
        if compute_shap:
            print("Computing SHAP stability score...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shap_stability = self.shap_interpreter.compute_shap_stability(
                    x,
                    population_indices,
                    nsamples=shap_nsamples
                )
            results['shap_stability'] = shap_stability

            # Compute SHAP values for feature importance
            shap_values = self.shap_interpreter.compute_shap_values(
                x,
                population_indices,
                nsamples=shap_nsamples
            )
            results['shap_values'] = shap_values

        # 2. Attention Consistency (Equation 11)
        print("Computing attention consistency...")
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                x.to(self.device),
                population_indices.to(self.device) if population_indices is not None else None,
                return_attention=True
            )

        if 'attention_weights' in outputs and outputs['attention_weights']:
            attention_consistency = AttentionConsistencyMetric.compute_consistency(
                outputs['attention_weights']
            )
            results['attention_consistency'] = attention_consistency

            # Per-layer consistency
            layer_consistencies = AttentionConsistencyMetric.compute_layer_consistency(
                outputs['attention_weights']
            )
            results['layer_attention_consistencies'] = layer_consistencies
        else:
            results['attention_consistency'] = 0.0

        # 3. Feature Ranking Correlation (Equation 12)
        if compute_shap and 'shap_values' in results:
            print("Computing feature ranking correlation...")

            # Compare SHAP-based ranking with attention-based ranking
            # Get attention-based feature importance
            attn_importance = self.model._compute_feature_importance(
                outputs['attention_weights']
            )

            # Get SHAP-based feature importance (average across samples and nutrients)
            shap_importance = np.abs(shap_values).mean(axis=(0, 2))

            # Ensure same length
            min_len = min(len(attn_importance), len(shap_importance))
            attn_importance = attn_importance[:min_len]
            shap_importance = shap_importance[:min_len]

            # Compute correlation
            ranking_correlation = FeatureRankingCorrelation.compute_correlation(
                attn_importance,
                shap_importance
            )
            results['feature_ranking_correlation'] = ranking_correlation

        # Overall interpretability score (average of available metrics)
        interpretability_scores = []
        if 'shap_stability' in results:
            interpretability_scores.append(results['shap_stability'])
        if 'attention_consistency' in results:
            interpretability_scores.append(results['attention_consistency'])
        if 'feature_ranking_correlation' in results:
            # Normalize correlation from [-1, 1] to [0, 1]
            normalized_corr = (results['feature_ranking_correlation'] + 1) / 2
            interpretability_scores.append(normalized_corr)

        if interpretability_scores:
            results['overall_interpretability'] = np.mean(interpretability_scores)
        else:
            results['overall_interpretability'] = 0.0

        return results

    def visualize_explanations(self,
                              x: torch.Tensor,
                              sample_idx: int = 0,
                              nutrient_idx: int = 0,
                              shap_nsamples: int = 100) -> Dict:
        """
        Generate visualization data for model explanations.

        Args:
            x: Input tensor
            sample_idx: Index of sample to explain
            nutrient_idx: Index of nutrient to explain
            shap_nsamples: Number of samples for SHAP

        Returns:
            Dictionary with visualization data
        """
        # Compute SHAP values
        shap_values = self.shap_interpreter.compute_shap_values(
            x,
            nsamples=shap_nsamples
        )

        # Get top features
        top_features = self.shap_interpreter.get_top_features(
            shap_values,
            feature_names=self.feature_names,
            top_k=10,
            nutrient_idx=nutrient_idx
        )

        # Get attention weights
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x.to(self.device), return_attention=True)

        visualization_data = {
            'top_features': top_features,
            'attention_weights': outputs['attention_weights'],
            'predictions': outputs['nutrient_recommendations'][sample_idx].cpu().numpy(),
            'deficiency_probabilities': outputs['deficiency_probabilities'][sample_idx].cpu().numpy()
        }

        return visualization_data


if __name__ == "__main__":
    print("Testing SIDRM Interpretability Framework...")
    print("=" * 70)

    from ..models.sidrm import SIDRM

    # Create test model
    input_dim = 105
    batch_size = 32
    num_nutrients = 50

    model = SIDRM(
        input_dim=input_dim,
        num_populations=4,
        hidden_dim=128,
        num_layers=5,
        num_nutrients=num_nutrients,
        dropout_rate=0.3
    )

    # Create test data
    x = torch.randn(batch_size, input_dim)
    background_data = torch.randn(100, input_dim)
    population_indices = torch.randint(0, 4, (batch_size,))

    # Initialize evaluator
    evaluator = SIDRMInterpretabilityEvaluator(
        model=model,
        background_data=background_data,
        num_bootstrap_samples=10,  # Reduced for testing
        device='cpu'
    )

    # Evaluate interpretability (without SHAP for faster testing)
    print("\nEvaluating interpretability (attention-based only)...")
    results = evaluator.evaluate_interpretability(
        x,
        population_indices,
        compute_shap=False  # Set to True for full evaluation
    )

    print(f"Attention consistency: {results['attention_consistency']:.4f}")
    print(f"Overall interpretability: {results['overall_interpretability']:.4f}")

    if 'layer_attention_consistencies' in results:
        print("\nPer-layer attention consistencies:")
        for i, consistency in enumerate(results['layer_attention_consistencies']):
            print(f"  Layer {i+1}: {consistency:.4f}")

    print("\nInterpretability framework test completed!")
