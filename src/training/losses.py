"""
Loss functions for KGNN training.

Implements Equation 6 from the paper:
L = L_pred + λ_k × L_know + λ_e × L_expl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PredictionLoss(nn.Module):
    """
    Prediction loss L_pred for micronutrient deficiency detection.

    Uses binary cross-entropy for multi-label classification.
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        """
        Initialize prediction loss.

        Args:
            pos_weight: Positive class weights for handling class imbalance
        """
        super(PredictionLoss, self).__init__()
        self.pos_weight = pos_weight
        if pos_weight is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.BCELoss()

    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction loss.

        Args:
            predictions: Predicted probabilities (batch_size, num_micronutrients)
            targets: True labels (batch_size, num_micronutrients)

        Returns:
            Loss value (scalar)
        """
        # Binary cross-entropy loss
        loss = self.criterion(predictions, targets)
        return loss


class KnowledgeLoss(nn.Module):
    """
    Knowledge consistency loss L_know.

    Penalizes violations of nutritional knowledge constraints.
    """

    def __init__(self,
                 mask_builder,
                 feature_names: list,
                 lambda_similarity: float = 1.0,
                 lambda_interaction: float = 1.0):
        """
        Initialize knowledge loss.

        Args:
            mask_builder: MaskMatrixBuilder instance with ontology and interactions
            feature_names: Names of input features
            lambda_similarity: Weight for similarity constraint violations
            lambda_interaction: Weight for interaction constraint violations
        """
        super(KnowledgeLoss, self).__init__()
        self.mask_builder = mask_builder
        self.feature_names = feature_names
        self.lambda_similarity = lambda_similarity
        self.lambda_interaction = lambda_interaction

    def forward(self,
                model: nn.Module,
                embedding_layer_name: str = 'embedding_layer') -> torch.Tensor:
        """
        Compute knowledge consistency loss.

        Args:
            model: KGNN model
            embedding_layer_name: Name of the embedding layer in the model

        Returns:
            Knowledge loss value (scalar)
        """
        # Get embedding weights from the model
        if hasattr(model, embedding_layer_name):
            embedding_layer = getattr(model, embedding_layer_name)
            embedding_weights = embedding_layer.linear.weight  # (embedding_dim, input_dim)
        else:
            return torch.tensor(0.0)

        # Use mask builder to compute constraint penalties
        penalty = self.mask_builder.create_constraint_penalties(
            embedding_weights,
            self.feature_names
        )

        return penalty


class ExplanationLoss(nn.Module):
    """
    Explanation quality loss L_expl.

    Encourages meaningful and interpretable attention distributions.
    """

    def __init__(self,
                 sparsity_weight: float = 0.5,
                 smoothness_weight: float = 0.3,
                 diversity_weight: float = 0.2):
        """
        Initialize explanation loss.

        Args:
            sparsity_weight: Weight for attention sparsity
            smoothness_weight: Weight for attention smoothness
            diversity_weight: Weight for attention diversity across samples
        """
        super(ExplanationLoss, self).__init__()
        self.sparsity_weight = sparsity_weight
        self.smoothness_weight = smoothness_weight
        self.diversity_weight = diversity_weight

    def forward(self,
                attention_weights: torch.Tensor,
                predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute explanation quality loss.

        Args:
            attention_weights: Attention weights (batch_size, hidden_dim)
            predictions: Model predictions (batch_size, num_micronutrients)

        Returns:
            Explanation loss value (scalar)
        """
        loss = torch.tensor(0.0, device=attention_weights.device)

        # 1. Sparsity loss: Encourage focused attention (L1 regularization)
        sparsity_loss = torch.mean(torch.abs(attention_weights))
        loss += self.sparsity_weight * sparsity_loss

        # 2. Smoothness loss: Penalize abrupt changes in attention
        # (encourages coherent explanations)
        if attention_weights.shape[1] > 1:
            diff = attention_weights[:, 1:] - attention_weights[:, :-1]
            smoothness_loss = torch.mean(diff ** 2)
            loss += self.smoothness_weight * smoothness_loss

        # 3. Diversity loss: Encourage diverse attention across batch
        # (prevents all samples from having identical explanations)
        if attention_weights.shape[0] > 1:
            mean_attention = attention_weights.mean(dim=0)
            diversity_loss = -torch.std(attention_weights, dim=0).mean()
            loss += self.diversity_weight * diversity_loss

        return loss


class MultiObjectiveLoss(nn.Module):
    """
    Multi-objective loss function from Equation 6.

    L = L_pred + λ_k × L_know + λ_e × L_expl

    Combines prediction accuracy, knowledge consistency, and explanation quality.
    """

    def __init__(self,
                 lambda_k: float = 0.3,
                 lambda_e: float = 0.2,
                 mask_builder = None,
                 feature_names: Optional[list] = None,
                 pos_weight: Optional[torch.Tensor] = None):
        """
        Initialize multi-objective loss.

        Args:
            lambda_k: Weight for knowledge loss (λ_k from paper)
            lambda_e: Weight for explanation loss (λ_e from paper)
            mask_builder: MaskMatrixBuilder for knowledge constraints
            feature_names: Names of input features
            pos_weight: Positive class weights for prediction loss
        """
        super(MultiObjectiveLoss, self).__init__()

        self.lambda_k = lambda_k
        self.lambda_e = lambda_e

        # Initialize component losses
        self.prediction_loss = PredictionLoss(pos_weight=pos_weight)

        if mask_builder is not None and feature_names is not None:
            self.knowledge_loss = KnowledgeLoss(mask_builder, feature_names)
        else:
            self.knowledge_loss = None

        self.explanation_loss = ExplanationLoss()

    def forward(self,
                model: nn.Module,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                attention_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective loss.

        Args:
            model: KGNN model (needed for knowledge loss)
            predictions: Model predictions
            targets: True labels
            attention_weights: Attention weights (optional)

        Returns:
            Dictionary with total loss and component losses
        """
        # 1. Prediction loss L_pred
        L_pred = self.prediction_loss(predictions, targets)

        # 2. Knowledge loss L_know
        if self.knowledge_loss is not None:
            L_know = self.knowledge_loss(model)
        else:
            L_know = torch.tensor(0.0, device=predictions.device)

        # 3. Explanation loss L_expl
        if attention_weights is not None:
            L_expl = self.explanation_loss(attention_weights, predictions)
        else:
            L_expl = torch.tensor(0.0, device=predictions.device)

        # Total loss (Equation 6)
        total_loss = L_pred + self.lambda_k * L_know + self.lambda_e * L_expl

        return {
            'total_loss': total_loss,
            'prediction_loss': L_pred,
            'knowledge_loss': L_know,
            'explanation_loss': L_expl
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Optional alternative to BCELoss for highly imbalanced datasets.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize focal loss.

        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            predictions: Predicted probabilities
            targets: True labels

        Returns:
            Loss value
        """
        bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')

        # Compute pt
        pt = torch.exp(-bce_loss)

        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()


if __name__ == "__main__":
    print("Testing Loss Functions...")

    batch_size = 32
    num_micronutrients = 12
    hidden_dim = 64

    # Create dummy data
    predictions = torch.rand(batch_size, num_micronutrients)
    targets = torch.randint(0, 2, (batch_size, num_micronutrients)).float()
    attention_weights = torch.rand(batch_size, hidden_dim)

    # Test Prediction Loss
    print("\n1. Testing Prediction Loss...")
    pred_loss = PredictionLoss()
    loss_value = pred_loss(predictions, targets)
    print(f"   Loss: {loss_value.item():.4f}")

    # Test Explanation Loss
    print("\n2. Testing Explanation Loss...")
    expl_loss = ExplanationLoss()
    loss_value = expl_loss(attention_weights, predictions)
    print(f"   Loss: {loss_value.item():.4f}")

    # Test Multi-Objective Loss (without knowledge loss)
    print("\n3. Testing Multi-Objective Loss...")
    multi_loss = MultiObjectiveLoss(lambda_k=0.3, lambda_e=0.2)

    # Create a dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding_layer = nn.Linear(10, 64)

    dummy_model = DummyModel()
    loss_dict = multi_loss(dummy_model, predictions, targets, attention_weights)

    print(f"   Total Loss: {loss_dict['total_loss'].item():.4f}")
    print(f"   Prediction Loss: {loss_dict['prediction_loss'].item():.4f}")
    print(f"   Knowledge Loss: {loss_dict['knowledge_loss'].item():.4f}")
    print(f"   Explanation Loss: {loss_dict['explanation_loss'].item():.4f}")

    # Test Focal Loss
    print("\n4. Testing Focal Loss...")
    focal_loss = FocalLoss()
    loss_value = focal_loss(predictions, targets)
    print(f"   Loss: {loss_value.item():.4f}")

    print("\nLoss functions test completed!")
