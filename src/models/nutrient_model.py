"""
Complete Nutrient Analysis Model with Multi-Task Learning

Implements the full model architecture with:
1. MobileNetV3 backbone
2. Multi-task output heads (food recognition, portion estimation, nutrient prediction)
3. Knowledge distillation support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mobilenetv3 import MobileNetV3
from .attention import LightweightAttention


class MultiTaskHead(nn.Module):
    """
    Multi-task output heads for comprehensive nutrient analysis.

    Outputs:
    1. Food recognition (classification)
    2. Portion estimation (regression)
    3. Macronutrient prediction (regression)
    4. Micronutrient prediction (regression)
    """
    def __init__(self, in_features, num_classes, num_macronutrients=3,
                 num_micronutrients=6, dropout=0.2):
        super(MultiTaskHead, self).__init__()

        # Food recognition head
        self.food_classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

        # Portion estimation head
        self.portion_estimator = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),  # Single value for portion size
            nn.ReLU()  # Ensure positive portion size
        )

        # Macronutrient prediction head (proteins, carbs, fats)
        self.macro_predictor = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_macronutrients),
            nn.ReLU()  # Ensure positive nutrient values
        )

        # Micronutrient prediction head (vitamins, minerals)
        # vitamin_a, vitamin_c, vitamin_d, iron, calcium, zinc
        self.micro_predictor = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_micronutrients),
            nn.ReLU()  # Ensure positive nutrient values
        )

    def forward(self, features):
        """
        Args:
            features: Feature tensor from backbone (B, in_features)

        Returns:
            Dictionary with:
                - food_logits: (B, num_classes)
                - portion: (B, 1)
                - macronutrients: (B, 3)
                - micronutrients: (B, 6)
        """
        food_logits = self.food_classifier(features)
        portion = self.portion_estimator(features)
        macronutrients = self.macro_predictor(features)
        micronutrients = self.micro_predictor(features)

        return {
            'food_logits': food_logits,
            'portion': portion,
            'macronutrients': macronutrients,
            'micronutrients': micronutrients
        }


class NutrientAnalysisModel(nn.Module):
    """
    Complete lightweight interpretable model for nutrient analysis.

    Architecture:
    - MobileNetV3 backbone with attention mechanisms
    - Multi-task output heads
    - Support for knowledge distillation

    Args:
        num_classes: Number of food categories (default: 500)
        width_multiplier: Width multiplier for backbone
        use_se: Use Squeeze-Excitation blocks
        use_sa: Use Shuffle Attention
        dropout: Dropout rate
        num_macronutrients: Number of macronutrients to predict
        num_micronutrients: Number of micronutrients to predict
    """
    def __init__(self, num_classes=500, width_multiplier=1.0,
                 use_se=True, use_sa=True, dropout=0.2,
                 num_macronutrients=3, num_micronutrients=6):
        super(NutrientAnalysisModel, self).__init__()

        # Backbone
        self.backbone = MobileNetV3(
            num_classes=num_classes,
            width_multiplier=width_multiplier,
            use_se=use_se,
            use_sa=use_sa,
            dropout=dropout
        )

        # Get feature dimension from backbone
        # MobileNetV3 final channels = 1280 * width_multiplier
        feature_dim = int(1280 * width_multiplier)

        # Multi-task heads
        self.multi_task_head = MultiTaskHead(
            in_features=feature_dim,
            num_classes=num_classes,
            num_macronutrients=num_macronutrients,
            num_micronutrients=num_micronutrients,
            dropout=dropout
        )

        # Store configuration
        self.num_classes = num_classes
        self.feature_dim = feature_dim

    def forward(self, x, return_features=False):
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (B, 3, 224, 224)
            return_features: Whether to return intermediate features (for distillation)

        Returns:
            If return_features=False:
                Dictionary with predictions
            If return_features=True:
                Tuple of (predictions, features)
        """
        # Extract features from backbone
        features, _ = self.backbone(x)

        # Multi-task predictions
        predictions = self.multi_task_head(features)

        if return_features:
            return predictions, features
        else:
            return predictions

    def get_feature_maps(self, x):
        """
        Extract feature maps from intermediate layers for interpretability.

        Args:
            x: Input tensor of shape (B, 3, 224, 224)

        Returns:
            Dictionary of feature maps from different stages
        """
        feature_maps = {}

        # Forward through stem
        x = self.backbone.conv_stem(x)
        feature_maps['stem'] = x

        # Forward through feature blocks
        for idx, block in enumerate(self.backbone.features):
            x = block(x)
            # Save feature maps from key stages
            if idx in [2, 5, 9, 15]:  # Representative stages
                feature_maps[f'stage_{idx}'] = x

        # Final convolution
        x = self.backbone.conv_head(x)
        feature_maps['head'] = x

        return feature_maps

    @classmethod
    def from_pretrained(cls, checkpoint_path, **kwargs):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            **kwargs: Additional arguments for model initialization

        Returns:
            Loaded model
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract model configuration if saved
        if 'config' in checkpoint:
            config = checkpoint['config']
            model = cls(**config)
        else:
            model = cls(**kwargs)

        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        return model

    def save(self, path, optimizer=None, epoch=None, metrics=None):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            optimizer: Optimizer state (optional)
            epoch: Current epoch (optional)
            metrics: Training metrics (optional)
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'num_classes': self.num_classes,
                'width_multiplier': 1.0,
                'use_se': True,
                'use_sa': True,
                'dropout': 0.2
            }
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if epoch is not None:
            checkpoint['epoch'] = epoch

        if metrics is not None:
            checkpoint['metrics'] = metrics

        torch.save(checkpoint, path)


class TeacherModel(nn.Module):
    """
    Teacher model for knowledge distillation (EfficientNet-B0).

    This is a simplified placeholder. In practice, you would use
    the actual EfficientNet-B0 implementation.
    """
    def __init__(self, num_classes=500):
        super(TeacherModel, self).__init__()

        # Placeholder - use actual EfficientNet-B0 in practice
        from torchvision.models import efficientnet_b0

        self.model = efficientnet_b0(pretrained=True)

        # Replace classifier
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Test the complete model
    print("Testing Nutrient Analysis Model...")

    model = NutrientAnalysisModel(
        num_classes=500,
        width_multiplier=1.0,
        use_se=True,
        use_sa=True,
        dropout=0.2
    )

    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Estimate model size
    param_size = total_params * 4 / (1024 ** 2)  # 4 bytes per float32, convert to MB
    print(f"Estimated model size (FP32): {param_size:.2f} MB")

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)

    print(f"\nInput shape: {x.shape}")

    # Standard forward pass
    predictions = model(x)
    print("\nPredictions:")
    for key, value in predictions.items():
        print(f"  {key}: {value.shape}")

    # Forward pass with features
    predictions, features = model(x, return_features=True)
    print(f"\nFeatures shape: {features.shape}")

    # Get feature maps
    feature_maps = model.get_feature_maps(x)
    print("\nFeature maps:")
    for name, fmap in feature_maps.items():
        print(f"  {name}: {fmap.shape}")

    print("\nModel test completed successfully!")
