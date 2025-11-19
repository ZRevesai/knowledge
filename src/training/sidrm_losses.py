"""
Multi-Objective Loss Function for SIDRM Training.

Implements Equation (13) from the paper:
L_total = α·L_accuracy + β·L_interpret + γ·L_clinical + δ·L_safety

where:
- α = 0.4: Accuracy loss weight
- β = 0.3: Interpretability loss weight
- γ = 0.2: Clinical loss weight
- δ = 0.1: Safety loss weight

Reference: "Smart Interpretable Dietary Recommender Model for Vulnerable Populations"
by Zvinodashe Revesai and Okuthe P. Kogeda (2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import numpy as np


class AccuracyLoss(nn.Module):
    """
    Accuracy loss component (L_accuracy).

    Uses cross-entropy for nutrient deficiency classification.
    """

    def __init__(self, num_nutrients: int = 50, weight: Optional[torch.Tensor] = None):
        """
        Initialize accuracy loss.

        Args:
            num_nutrients: Number of nutrients to predict
            weight: Optional class weights for imbalanced data
        """
        super(AccuracyLoss, self).__init__()
        self.num_nutrients = num_nutrients
        self.bce_loss = nn.BCEWithLogitsLoss(weight=weight, reduction='mean')

    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                nutrient_recommendations: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute accuracy loss.

        Args:
            predictions: Predicted deficiency probabilities (batch_size, num_nutrients)
            targets: Ground truth deficiency labels (batch_size, num_nutrients)
            nutrient_recommendations: Nutrient adequacy scores (optional)

        Returns:
            Accuracy loss value
        """
        # Binary cross-entropy for multi-label classification
        loss = self.bce_loss(predictions, targets)

        # If nutrient recommendations provided, add MSE for regression
        if nutrient_recommendations is not None:
            mse_loss = F.mse_loss(nutrient_recommendations, targets)
            loss = 0.7 * loss + 0.3 * mse_loss

        return loss


class InterpretabilityLoss(nn.Module):
    """
    Interpretability loss component (L_interpret).

    Uses attention entropy regularization to encourage focused, interpretable attention patterns.
    """

    def __init__(self, entropy_weight: float = 1.0, sparsity_weight: float = 0.5):
        """
        Initialize interpretability loss.

        Args:
            entropy_weight: Weight for attention entropy term
            sparsity_weight: Weight for attention sparsity term
        """
        super(InterpretabilityLoss, self).__init__()
        self.entropy_weight = entropy_weight
        self.sparsity_weight = sparsity_weight

    def forward(self, attention_weights: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute interpretability loss from attention weights.

        Encourages:
        1. Low entropy (focused attention)
        2. Sparsity (few important features)

        Args:
            attention_weights: List of attention weight tensors from each layer
                              Each tensor: (batch, num_heads, seq_len, seq_len)

        Returns:
            Interpretability loss value
        """
        if not attention_weights:
            return torch.tensor(0.0)

        total_entropy_loss = 0.0
        total_sparsity_loss = 0.0
        num_layers = len(attention_weights)

        for attn_weights in attention_weights:
            # attn_weights: (batch, num_heads, seq, seq)
            batch_size, num_heads, seq_len, _ = attn_weights.shape

            # Average attention across sequence dimension for each head
            # (batch, num_heads, seq)
            avg_attn = attn_weights.mean(dim=-1)

            # 1. Entropy loss (encourage low entropy = focused attention)
            # H = -Σ p(i) log p(i)
            entropy = -torch.sum(avg_attn * torch.log(avg_attn + 1e-10), dim=-1)
            entropy_loss = entropy.mean()

            # 2. Sparsity loss (encourage sparse attention)
            # L1 regularization on attention weights
            sparsity_loss = torch.mean(torch.abs(avg_attn))

            total_entropy_loss += entropy_loss
            total_sparsity_loss += sparsity_loss

        # Average across layers
        avg_entropy_loss = total_entropy_loss / num_layers
        avg_sparsity_loss = total_sparsity_loss / num_layers

        # Combined interpretability loss
        loss = self.entropy_weight * avg_entropy_loss + self.sparsity_weight * avg_sparsity_loss

        return loss


class ClinicalLoss(nn.Module):
    """
    Clinical loss component (L_clinical).

    Penalizes violations of established dietary guidelines and clinical knowledge.
    """

    def __init__(self,
                 guideline_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                 population_specific: bool = True):
        """
        Initialize clinical loss.

        Args:
            guideline_bounds: Dictionary mapping nutrient names to (min, max) acceptable ranges
            population_specific: Whether to use population-specific guidelines
        """
        super(ClinicalLoss, self).__init__()
        self.guideline_bounds = guideline_bounds or self._default_guidelines()
        self.population_specific = population_specific

    def _default_guidelines(self) -> Dict[str, Tuple[float, float]]:
        """
        Default dietary guidelines based on RDA/DRI standards.

        Returns:
            Dictionary of nutrient bounds
        """
        return {
            # Format: nutrient_name: (lower_bound, upper_bound)
            'vitamin_b12': (2.4, 100.0),  # mcg/day
            'vitamin_d': (600.0, 4000.0),  # IU/day
            'calcium': (1000.0, 2500.0),  # mg/day
            'iron': (8.0, 45.0),  # mg/day
            'folate': (400.0, 1000.0),  # mcg/day
            'vitamin_c': (75.0, 2000.0),  # mg/day
            # Add more nutrients as needed
        }

    def forward(self,
                nutrient_recommendations: torch.Tensor,
                population_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute clinical loss based on guideline violations.

        Args:
            nutrient_recommendations: Predicted nutrient adequacy scores (batch_size, num_nutrients)
            population_indices: Population category for each sample (batch_size,)

        Returns:
            Clinical loss value
        """
        batch_size, num_nutrients = nutrient_recommendations.shape

        # Normalize recommendations to [0, 1] range
        normalized_recs = torch.sigmoid(nutrient_recommendations)

        # Penalize extreme values (too low or too high)
        # Encourage recommendations in middle range [0.3, 0.7]
        target_lower = 0.3
        target_upper = 0.7

        # Violations for values outside acceptable range
        lower_violations = F.relu(target_lower - normalized_recs)
        upper_violations = F.relu(normalized_recs - target_upper)

        # Combined violation loss
        violation_loss = torch.mean(lower_violations + upper_violations)

        # Population-specific adjustments
        if population_indices is not None and self.population_specific:
            # Pregnant women (population 0): higher requirements for folate, iron
            # Elderly (population 1): higher requirements for B12, D, calcium
            # Children (population 2): balanced requirements
            # Chronic disease (population 3): varied based on condition

            population_adjustments = torch.zeros_like(normalized_recs)

            # Example adjustments (would be more sophisticated in practice)
            pregnant_mask = (population_indices == 0).unsqueeze(1)
            elderly_mask = (population_indices == 1).unsqueeze(1)

            # Penalize low folate/iron for pregnant women (assuming indices 3, 4)
            if num_nutrients > 4:
                population_adjustments[:, 3:5] = torch.where(
                    pregnant_mask.expand(-1, 2),
                    F.relu(0.5 - normalized_recs[:, 3:5]),
                    population_adjustments[:, 3:5]
                )

            # Penalize low B12/D/calcium for elderly (assuming indices 0, 1, 5)
            if num_nutrients > 5:
                elderly_nutrients = torch.stack([
                    normalized_recs[:, 0],
                    normalized_recs[:, 1],
                    normalized_recs[:, 5]
                ], dim=1)

                population_adjustments[:, [0, 1, 5]] = torch.where(
                    elderly_mask.expand(-1, 3),
                    F.relu(0.5 - elderly_nutrients),
                    population_adjustments[:, [0, 1, 5]]
                )

            population_loss = torch.mean(population_adjustments)
            violation_loss = 0.7 * violation_loss + 0.3 * population_loss

        return violation_loss


class SafetyLoss(nn.Module):
    """
    Safety loss component (L_safety).

    Adds penalty terms for unsafe nutrient recommendations that could lead to toxicity.
    """

    def __init__(self,
                 upper_limits: Optional[Dict[str, float]] = None,
                 toxicity_weight: float = 2.0):
        """
        Initialize safety loss.

        Args:
            upper_limits: Dictionary mapping nutrient names to upper tolerable limits (UL)
            toxicity_weight: Weight for toxicity penalty (higher = more conservative)
        """
        super(SafetyLoss, self).__init__()
        self.upper_limits = upper_limits or self._default_upper_limits()
        self.toxicity_weight = toxicity_weight

    def _default_upper_limits(self) -> Dict[str, float]:
        """
        Default upper tolerable limits based on dietary guidelines.

        Returns:
            Dictionary of upper limits
        """
        return {
            # Format: nutrient_name: upper_tolerable_limit
            'vitamin_a': 3000.0,  # mcg/day (can cause toxicity)
            'vitamin_d': 4000.0,  # IU/day
            'calcium': 2500.0,  # mg/day
            'iron': 45.0,  # mg/day (particularly important for elderly)
            'zinc': 40.0,  # mg/day
            'vitamin_e': 1000.0,  # mg/day
            'folate': 1000.0,  # mcg/day
            # Add more nutrients as needed
        }

    def forward(self,
                nutrient_recommendations: torch.Tensor,
                deficiency_probabilities: torch.Tensor) -> torch.Tensor:
        """
        Compute safety loss for unsafe recommendations.

        Args:
            nutrient_recommendations: Nutrient adequacy scores (batch_size, num_nutrients)
            deficiency_probabilities: Deficiency probabilities (batch_size, num_nutrients)

        Returns:
            Safety loss value
        """
        # Penalize very high recommendations (potential toxicity)
        # Recommendations should generally not exceed 1.5 (150% of adequate intake)
        toxicity_threshold = 1.5
        toxicity_violations = F.relu(torch.sigmoid(nutrient_recommendations) - toxicity_threshold)
        toxicity_loss = torch.mean(toxicity_violations) * self.toxicity_weight

        # Penalize contradictory predictions (high deficiency prob with high recommendation)
        # This encourages consistency
        contradiction_loss = torch.mean(
            deficiency_probabilities * torch.sigmoid(nutrient_recommendations)
        )

        # Combined safety loss
        safety_loss = toxicity_loss + 0.5 * contradiction_loss

        return safety_loss


class SIDRMMultiObjectiveLoss(nn.Module):
    """
    Multi-objective loss function for SIDRM training.

    Implements Equation (13) from the paper:
    L_total = α·L_accuracy + β·L_interpret + γ·L_clinical + δ·L_safety

    where α=0.4, β=0.3, γ=0.2, δ=0.1
    """

    def __init__(self,
                 num_nutrients: int = 50,
                 alpha: float = 0.4,
                 beta: float = 0.3,
                 gamma: float = 0.2,
                 delta: float = 0.1,
                 class_weights: Optional[torch.Tensor] = None,
                 guideline_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                 upper_limits: Optional[Dict[str, float]] = None):
        """
        Initialize multi-objective loss.

        Args:
            num_nutrients: Number of nutrients to predict
            alpha: Weight for accuracy loss (default: 0.4)
            beta: Weight for interpretability loss (default: 0.3)
            gamma: Weight for clinical loss (default: 0.2)
            delta: Weight for safety loss (default: 0.1)
            class_weights: Optional weights for imbalanced classes
            guideline_bounds: Clinical guideline bounds
            upper_limits: Upper tolerable limits for safety
        """
        super(SIDRMMultiObjectiveLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        # Initialize component losses
        self.accuracy_loss = AccuracyLoss(num_nutrients, weight=class_weights)
        self.interpretability_loss = InterpretabilityLoss()
        self.clinical_loss = ClinicalLoss(guideline_bounds)
        self.safety_loss = SafetyLoss(upper_limits)

    def forward(self,
                outputs: Dict[str, torch.Tensor],
                targets: torch.Tensor,
                population_indices: Optional[torch.Tensor] = None,
                return_components: bool = False) -> Dict[str, torch.Tensor]:
        """
        Compute total multi-objective loss.

        Args:
            outputs: Dictionary from model forward pass containing:
                    - nutrient_recommendations: (batch_size, num_nutrients)
                    - deficiency_probabilities: (batch_size, num_nutrients)
                    - attention_weights: List of attention tensors
            targets: Ground truth deficiency labels (batch_size, num_nutrients)
            population_indices: Population category for each sample
            return_components: Whether to return individual loss components

        Returns:
            Dictionary containing total loss and optionally component losses
        """
        # Extract outputs
        nutrient_recs = outputs['nutrient_recommendations']
        deficiency_probs = outputs['deficiency_probabilities']
        attention_weights = outputs.get('attention_weights', [])

        # Compute component losses
        # 1. Accuracy loss
        L_accuracy = self.accuracy_loss(
            deficiency_probs,
            targets,
            nutrient_recommendations=nutrient_recs
        )

        # 2. Interpretability loss
        L_interpret = self.interpretability_loss(attention_weights)

        # 3. Clinical loss
        L_clinical = self.clinical_loss(nutrient_recs, population_indices)

        # 4. Safety loss
        L_safety = self.safety_loss(nutrient_recs, deficiency_probs)

        # Total loss (Equation 13)
        L_total = (
            self.alpha * L_accuracy +
            self.beta * L_interpret +
            self.gamma * L_clinical +
            self.delta * L_safety
        )

        # Prepare output
        loss_dict = {'total_loss': L_total}

        if return_components:
            loss_dict.update({
                'accuracy_loss': L_accuracy,
                'interpretability_loss': L_interpret,
                'clinical_loss': L_clinical,
                'safety_loss': L_safety
            })

        return loss_dict

    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss component weights."""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'delta': self.delta
        }

    def set_loss_weights(self,
                        alpha: Optional[float] = None,
                        beta: Optional[float] = None,
                        gamma: Optional[float] = None,
                        delta: Optional[float] = None):
        """
        Update loss component weights.

        Args:
            alpha: New accuracy weight
            beta: New interpretability weight
            gamma: New clinical weight
            delta: New safety weight
        """
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        if delta is not None:
            self.delta = delta

        # Optionally normalize to sum to 1
        total = self.alpha + self.beta + self.gamma + self.delta
        if abs(total - 1.0) > 0.01:
            print(f"Warning: Loss weights sum to {total:.3f}, not 1.0")


if __name__ == "__main__":
    print("Testing SIDRM Multi-Objective Loss Function...")
    print("=" * 70)

    # Test parameters
    batch_size = 32
    num_nutrients = 50
    num_layers = 5
    num_heads = 8
    seq_len = 1

    # Create dummy model outputs
    outputs = {
        'nutrient_recommendations': torch.randn(batch_size, num_nutrients),
        'deficiency_probabilities': torch.sigmoid(torch.randn(batch_size, num_nutrients)),
        'attention_weights': [
            torch.softmax(torch.randn(batch_size, num_heads, seq_len, seq_len), dim=-1)
            for _ in range(num_layers)
        ]
    }

    # Create dummy targets
    targets = torch.randint(0, 2, (batch_size, num_nutrients)).float()
    population_indices = torch.randint(0, 4, (batch_size,))

    # Initialize loss function
    criterion = SIDRMMultiObjectiveLoss(
        num_nutrients=num_nutrients,
        alpha=0.4,
        beta=0.3,
        gamma=0.2,
        delta=0.1
    )

    # Compute loss
    loss_dict = criterion(
        outputs,
        targets,
        population_indices,
        return_components=True
    )

    # Print results
    print("Loss components:")
    print(f"  Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"  Accuracy loss: {loss_dict['accuracy_loss'].item():.4f}")
    print(f"  Interpretability loss: {loss_dict['interpretability_loss'].item():.4f}")
    print(f"  Clinical loss: {loss_dict['clinical_loss'].item():.4f}")
    print(f"  Safety loss: {loss_dict['safety_loss'].item():.4f}")

    # Test loss weights
    print("\nLoss weights:")
    weights = criterion.get_loss_weights()
    for name, value in weights.items():
        print(f"  {name}: {value:.2f}")

    print("\nLoss function test completed successfully!")
