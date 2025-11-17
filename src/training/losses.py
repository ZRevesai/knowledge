"""
Loss functions for multi-task learning and knowledge distillation.

Implements:
1. Multi-task loss (Equation 17)
2. Micronutrient-specific loss (Equation 18)
3. Distillation loss (Equation 21)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function for nutrient analysis.

    Reference: Equation (17) in the paper
    L_total = α L_food + β L_portion + γ L_macro + δ L_micro

    Args:
        alpha: Weight for food recognition loss
        beta: Weight for portion estimation loss
        gamma: Weight for macronutrient prediction loss
        delta: Weight for micronutrient prediction loss
        micronutrient_weights: Importance weights for different micronutrients
    """
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.5, delta=0.3,
                 micronutrient_weights=None):
        super(MultiTaskLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        # Micronutrient importance weights based on global deficiency prevalence
        if micronutrient_weights is None:
            # Default weights: [vit_a, vit_c, vit_d, iron, calcium, zinc]
            self.micronutrient_weights = torch.tensor([
                1.2,  # Vitamin A
                1.1,  # Vitamin C
                1.3,  # Vitamin D
                1.4,  # Iron
                1.3,  # Calcium
                1.2   # Zinc
            ])
        else:
            self.micronutrient_weights = torch.tensor(micronutrient_weights)

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()  # Food recognition
        self.mse_loss = nn.MSELoss()          # Portion estimation
        self.mae_loss = nn.L1Loss()           # Macronutrient prediction

    def forward(self, predictions, targets):
        """
        Compute multi-task loss.

        Args:
            predictions: Dictionary with model predictions
                - food_logits: (B, num_classes)
                - portion: (B, 1)
                - macronutrients: (B, 3)
                - micronutrients: (B, 6)
            targets: Dictionary with ground truth
                - label: (B,)
                - portion: (B, 1)
                - macronutrients: (B, 3)
                - micronutrients: (B, 6)

        Returns:
            Dictionary with total loss and individual losses
        """
        # Food recognition loss (cross-entropy)
        L_food = self.ce_loss(predictions['food_logits'], targets['label'])

        # Portion estimation loss (MSE)
        L_portion = self.mse_loss(predictions['portion'], targets['portion'])

        # Macronutrient prediction loss (MAE)
        L_macro = self.mae_loss(predictions['macronutrients'],
                               targets['macronutrients'])

        # Micronutrient prediction loss (weighted MAE)
        # Reference: Equation (18)
        # L_micro = Σ w_i · |y_i - ŷ_i|
        L_micro = self.weighted_micronutrient_loss(
            predictions['micronutrients'],
            targets['micronutrients']
        )

        # Total loss (Equation 17)
        L_total = (self.alpha * L_food +
                  self.beta * L_portion +
                  self.gamma * L_macro +
                  self.delta * L_micro)

        return {
            'total': L_total,
            'food': L_food,
            'portion': L_portion,
            'macro': L_macro,
            'micro': L_micro
        }

    def weighted_micronutrient_loss(self, predictions, targets):
        """
        Weighted MAE loss for micronutrients.

        Reference: Equation (18)
        L_micro = Σ_{i∈vitamins,minerals} w_i · |y_i - ŷ_i|

        Args:
            predictions: Predicted micronutrients (B, 6)
            targets: Target micronutrients (B, 6)

        Returns:
            Weighted loss value
        """
        # Ensure weights are on the same device
        weights = self.micronutrient_weights.to(predictions.device)

        # Compute absolute errors
        errors = torch.abs(predictions - targets)

        # Apply weights
        weighted_errors = errors * weights.unsqueeze(0)

        # Return mean weighted error
        return weighted_errors.mean()


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss.

    Reference: Equation (21) in the paper
    L_total_distill = (1 - λ)L_total + λ L_distill

    Args:
        temperature: Temperature for softening probability distributions
        lambda_distill: Weight for distillation loss
        base_loss: Base multi-task loss function
    """
    def __init__(self, temperature=2.0, lambda_distill=0.5, base_loss=None):
        super(DistillationLoss, self).__init__()

        self.temperature = temperature
        self.lambda_distill = lambda_distill

        if base_loss is None:
            self.base_loss = MultiTaskLoss()
        else:
            self.base_loss = base_loss

    def forward(self, student_outputs, teacher_outputs, targets):
        """
        Compute distillation loss.

        Args:
            student_outputs: Dictionary with student model predictions
            teacher_outputs: Dictionary with teacher model predictions
            targets: Dictionary with ground truth

        Returns:
            Dictionary with total loss and components
        """
        # Compute base multi-task loss
        base_losses = self.base_loss(student_outputs, targets)
        L_total = base_losses['total']

        # Compute distillation loss (KL divergence on food predictions)
        L_distill = self.knowledge_distillation_loss(
            student_outputs['food_logits'],
            teacher_outputs['food_logits']
        )

        # Combined loss (Equation 21)
        L_total_distill = (1 - self.lambda_distill) * L_total + \
                         self.lambda_distill * L_distill

        return {
            'total': L_total_distill,
            'base': L_total,
            'distill': L_distill,
            **base_losses
        }

    def knowledge_distillation_loss(self, student_logits, teacher_logits):
        """
        KL divergence loss for knowledge distillation.

        Args:
            student_logits: Student model logits (B, num_classes)
            teacher_logits: Teacher model logits (B, num_classes)

        Returns:
            KL divergence loss
        """
        # Soften probabilities with temperature
        student_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)

        # KL divergence
        kl_div = F.kl_div(student_probs, teacher_probs,
                         reduction='batchmean')

        # Scale by temperature squared
        return kl_div * (self.temperature ** 2)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Useful for food security categories with varying sample sizes.

    Args:
        alpha: Weighting factor
        gamma: Focusing parameter
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Compute focal loss.

        Args:
            inputs: Predictions (B, num_classes)
            targets: Ground truth labels (B,)

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class NutrientMAE(nn.Module):
    """
    Mean Absolute Error for nutrient estimation.

    Reference: Equation (23) in the paper
    MAE = (1/n) Σ |y_i - ŷ_i|
    """
    def __init__(self):
        super(NutrientMAE, self).__init__()

    def forward(self, predictions, targets):
        """
        Compute MAE.

        Args:
            predictions: Predicted nutrient values
            targets: Ground truth nutrient values

        Returns:
            MAE value
        """
        return torch.mean(torch.abs(predictions - targets))


class NutrientMAPE(nn.Module):
    """
    Mean Absolute Percentage Error for nutrient estimation.

    Reference: Equation (24) in the paper
    MAPE = (100/n) Σ |y_i - ŷ_i| / y_i
    """
    def __init__(self, epsilon=1e-8):
        super(NutrientMAPE, self).__init__()
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        """
        Compute MAPE.

        Args:
            predictions: Predicted nutrient values
            targets: Ground truth nutrient values

        Returns:
            MAPE value (percentage)
        """
        # Avoid division by zero
        targets = targets + self.epsilon

        # Compute percentage error
        percentage_error = torch.abs((targets - predictions) / targets)

        # Return mean percentage error * 100
        return 100.0 * torch.mean(percentage_error)


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")

    batch_size = 4
    num_classes = 500

    # Create dummy predictions
    predictions = {
        'food_logits': torch.randn(batch_size, num_classes),
        'portion': torch.rand(batch_size, 1) * 200,
        'macronutrients': torch.rand(batch_size, 3) * 50,
        'micronutrients': torch.rand(batch_size, 6) * 100
    }

    # Create dummy targets
    targets = {
        'label': torch.randint(0, num_classes, (batch_size,)),
        'portion': torch.rand(batch_size, 1) * 200,
        'macronutrients': torch.rand(batch_size, 3) * 50,
        'micronutrients': torch.rand(batch_size, 6) * 100
    }

    # Test MultiTaskLoss
    multi_task_loss = MultiTaskLoss(alpha=1.0, beta=0.5, gamma=0.5, delta=0.3)
    losses = multi_task_loss(predictions, targets)

    print("\nMulti-Task Loss:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")

    # Test DistillationLoss
    teacher_outputs = {
        'food_logits': torch.randn(batch_size, num_classes),
    }

    distill_loss = DistillationLoss(temperature=2.0, lambda_distill=0.5)
    distill_losses = distill_loss(predictions, teacher_outputs, targets)

    print("\nDistillation Loss:")
    for name, value in distill_losses.items():
        print(f"  {name}: {value.item():.4f}")

    # Test MAE and MAPE
    mae_loss = NutrientMAE()
    mape_loss = NutrientMAPE()

    mae = mae_loss(predictions['macronutrients'], targets['macronutrients'])
    mape = mape_loss(predictions['macronutrients'], targets['macronutrients'])

    print(f"\nMAE: {mae.item():.4f}")
    print(f"MAPE: {mape.item():.2f}%")

    print("\nLoss functions test completed successfully!")
