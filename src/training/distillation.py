"""
Knowledge Distillation for model compression.

Implements teacher-student training with feature alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillation:
    """
    Knowledge distillation trainer.

    Transfers knowledge from a larger teacher model (EfficientNet-B0)
    to a smaller student model (our lightweight architecture).

    Args:
        student_model: Student model to train
        teacher_model: Pre-trained teacher model
        temperature: Temperature for softening distributions
        lambda_distill: Weight for distillation loss
        feature_alignment: Whether to align intermediate features
        alignment_layers: Indices of layers to align
    """
    def __init__(self, student_model, teacher_model,
                 temperature=2.0, lambda_distill=0.5,
                 feature_alignment=True, alignment_layers=None):
        self.student = student_model
        self.teacher = teacher_model
        self.temperature = temperature
        self.lambda_distill = lambda_distill
        self.feature_alignment = feature_alignment

        if alignment_layers is None:
            self.alignment_layers = [3, 6, 9]
        else:
            self.alignment_layers = alignment_layers

        # Freeze teacher model
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Feature alignment modules
        if self.feature_alignment:
            self.alignment_modules = self._create_alignment_modules()

    def _create_alignment_modules(self):
        """Create projection layers for feature alignment."""
        # This is a placeholder - actual implementation depends on
        # teacher and student architectures
        modules = nn.ModuleDict()

        # Create projection layers for each alignment point
        for idx in self.alignment_layers:
            # Project student features to teacher feature space
            modules[f'align_{idx}'] = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1),  # Example dimensions
                nn.BatchNorm2d(128)
            )

        return modules

    def compute_distillation_loss(self, student_logits, teacher_logits):
        """
        Compute KL divergence for knowledge distillation.

        Args:
            student_logits: Student model predictions
            teacher_logits: Teacher model predictions

        Returns:
            Distillation loss
        """
        # Soften probabilities
        student_probs = F.log_softmax(
            student_logits / self.temperature, dim=1
        )
        teacher_probs = F.softmax(
            teacher_logits / self.temperature, dim=1
        )

        # KL divergence
        kl_div = F.kl_div(
            student_probs, teacher_probs,
            reduction='batchmean'
        )

        # Scale by temperature squared
        return kl_div * (self.temperature ** 2)

    def compute_feature_alignment_loss(self, student_features,
                                       teacher_features):
        """
        Compute feature alignment loss for intermediate layers.

        Args:
            student_features: Dictionary of student feature maps
            teacher_features: Dictionary of teacher feature maps

        Returns:
            Feature alignment loss
        """
        alignment_loss = 0.0
        num_alignments = 0

        for layer_idx in self.alignment_layers:
            student_feat = student_features.get(f'stage_{layer_idx}')
            teacher_feat = teacher_features.get(f'stage_{layer_idx}')

            if student_feat is not None and teacher_feat is not None:
                # Project student features if needed
                if f'align_{layer_idx}' in self.alignment_modules:
                    student_feat = self.alignment_modules[f'align_{layer_idx}'](
                        student_feat
                    )

                # Compute MSE between features
                alignment_loss += F.mse_loss(student_feat, teacher_feat)
                num_alignments += 1

        if num_alignments > 0:
            alignment_loss /= num_alignments

        return alignment_loss

    def train_step(self, batch, optimizer, base_loss_fn, device):
        """
        Single training step with distillation.

        Args:
            batch: Batch of data
            optimizer: Optimizer
            base_loss_fn: Base loss function
            device: Device to use

        Returns:
            Dictionary with losses
        """
        images = batch['image'].to(device)
        targets = {
            'label': batch['label'].to(device),
            'portion': batch['portion'].to(device),
            'macronutrients': batch['macronutrients'].to(device),
            'micronutrients': batch['micronutrients'].to(device)
        }

        # Student forward pass
        student_outputs, student_features = self.student(
            images, return_features=True
        )

        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher(images)
            # Assuming teacher returns dictionary with 'food_logits'

        # Compute base loss
        base_losses = base_loss_fn(student_outputs, targets)
        base_loss = base_losses['total']

        # Compute distillation loss
        distill_loss = self.compute_distillation_loss(
            student_outputs['food_logits'],
            teacher_outputs.get('food_logits', teacher_outputs)
        )

        # Compute feature alignment loss if enabled
        if self.feature_alignment:
            # Get teacher features
            teacher_feature_maps = {}
            # This would need actual implementation based on teacher architecture
            feature_align_loss = 0.0  # Placeholder
        else:
            feature_align_loss = 0.0

        # Combined loss
        total_loss = (
            (1 - self.lambda_distill) * base_loss +
            self.lambda_distill * distill_loss +
            0.1 * feature_align_loss  # Small weight for feature alignment
        )

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return {
            'total': total_loss.item(),
            'base': base_loss.item(),
            'distill': distill_loss.item(),
            'feature_align': feature_align_loss if isinstance(feature_align_loss, float)
                           else feature_align_loss.item(),
            **{k: v.item() for k, v in base_losses.items() if k != 'total'}
        }


class SelfDistillation:
    """
    Self-distillation using ensemble of previous model checkpoints.

    Useful when teacher model is not available.
    """
    def __init__(self, student_model, temperature=2.0,
                 alpha=0.5, num_generations=3):
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.num_generations = num_generations
        self.previous_models = []

    def add_checkpoint(self, checkpoint_path):
        """Add a previous model checkpoint for ensemble."""
        model = torch.load(checkpoint_path)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.previous_models.append(model)

        # Keep only the most recent checkpoints
        if len(self.previous_models) > self.num_generations:
            self.previous_models.pop(0)

    def get_ensemble_predictions(self, x):
        """Get averaged predictions from ensemble."""
        if not self.previous_models:
            return None

        with torch.no_grad():
            ensemble_logits = []
            for model in self.previous_models:
                outputs = model(x)
                ensemble_logits.append(outputs['food_logits'])

            # Average logits
            return torch.stack(ensemble_logits).mean(dim=0)


if __name__ == "__main__":
    print("Testing Knowledge Distillation...")

    from src.models.nutrient_model import NutrientAnalysisModel

    # Create student model
    student = NutrientAnalysisModel(num_classes=500)

    # Create teacher model (simplified)
    teacher = NutrientAnalysisModel(num_classes=500)

    # Initialize distillation
    kd = KnowledgeDistillation(
        student_model=student,
        teacher_model=teacher,
        temperature=2.0,
        lambda_distill=0.5
    )

    # Test distillation loss
    student_logits = torch.randn(4, 500)
    teacher_logits = torch.randn(4, 500)

    distill_loss = kd.compute_distillation_loss(student_logits, teacher_logits)
    print(f"Distillation loss: {distill_loss.item():.4f}")

    print("\nKnowledge distillation test completed!")
