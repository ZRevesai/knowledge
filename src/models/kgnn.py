"""
Knowledge-Guided Neural Network (KGNN) Model.

Main implementation of the KGNN architecture for micronutrient
deficiency detection in elderly populations.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np

from .layers import (
    KnowledgeEmbeddingLayer,
    KnowledgeGuidedLayer,
    AttentionMechanism,
    InterpretableOutputLayer,
    MultiHeadAttention
)


class KGNN(nn.Module):
    """
    Knowledge-Guided Neural Network for micronutrient deficiency detection.

    This model integrates nutritional domain knowledge through:
    1. Knowledge Embedding Layer (Equation 1)
    2. Knowledge-Guided Interaction Layers (Equation 2)
    3. Attention Mechanism (Equations 3-4)
    4. Interpretable Output Layer (Equation 5)
    """

    def __init__(self,
                 input_dim: int,
                 embedding_dim: int = 128,
                 hidden_dims: List[int] = [256, 128, 64],
                 num_micronutrients: int = 12,
                 micronutrient_names: Optional[List[str]] = None,
                 embedding_weights: Optional[torch.Tensor] = None,
                 mask_matrices: Optional[List[torch.Tensor]] = None,
                 dropout_rate: float = 0.3,
                 activation: str = 'relu',
                 use_batch_norm: bool = True,
                 use_multi_head_attention: bool = False,
                 num_attention_heads: int = 4):
        """
        Initialize KGNN model.

        Args:
            input_dim: Number of input features
            embedding_dim: Dimension of embedding space
            hidden_dims: List of hidden layer dimensions
            num_micronutrients: Number of micronutrients to predict
            micronutrient_names: Names of micronutrients
            embedding_weights: Knowledge-constrained initial weights for embedding
            mask_matrices: Knowledge-based mask matrices for each layer
            dropout_rate: Dropout probability
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            use_multi_head_attention: Whether to use multi-head attention
            num_attention_heads: Number of attention heads (if multi-head)
        """
        super(KGNN, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.num_micronutrients = num_micronutrients
        self.micronutrient_names = micronutrient_names or \
                                   [f"Micronutrient_{i}" for i in range(num_micronutrients)]

        # 1. Knowledge Embedding Layer (Equation 1)
        self.embedding_layer = KnowledgeEmbeddingLayer(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            initial_weights=embedding_weights,
            dropout_rate=dropout_rate,
            activation=activation
        )

        # 2. Knowledge-Guided Interaction Layers (Equation 2)
        self.hidden_layers = nn.ModuleList()
        prev_dim = embedding_dim

        for idx, hidden_dim in enumerate(hidden_dims):
            mask = mask_matrices[idx] if mask_matrices and idx < len(mask_matrices) else None
            layer = KnowledgeGuidedLayer(
                input_dim=prev_dim,
                output_dim=hidden_dim,
                mask_matrix=mask,
                dropout_rate=dropout_rate,
                activation=activation,
                use_batch_norm=use_batch_norm
            )
            self.hidden_layers.append(layer)
            prev_dim = hidden_dim

        # 3. Attention Mechanism (Equations 3-4)
        self.use_multi_head_attention = use_multi_head_attention
        if use_multi_head_attention:
            self.attention = MultiHeadAttention(
                hidden_dim=prev_dim,
                num_heads=num_attention_heads,
                dropout_rate=dropout_rate
            )
        else:
            self.attention = AttentionMechanism(
                hidden_dim=prev_dim,
                attention_dim=prev_dim // 2
            )

        # 4. Interpretable Output Layer (Equation 5)
        self.output_layer = InterpretableOutputLayer(
            input_dim=prev_dim,
            num_micronutrients=num_micronutrients,
            micronutrient_names=self.micronutrient_names
        )

    def forward(self,
                x: torch.Tensor,
                return_attention: bool = False,
                return_hidden: bool = False) -> Dict:
        """
        Forward pass through KGNN.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            return_attention: Whether to return attention weights
            return_hidden: Whether to return hidden representations

        Returns:
            Dictionary containing:
                - predictions: Deficiency probabilities (batch_size, num_micronutrients)
                - attention_weights: Attention weights (if return_attention=True)
                - hidden_representations: Hidden states (if return_hidden=True)
                - interpretability_info: Feature contributions and other interpretation data
        """
        outputs = {}
        hidden_states = []

        # 1. Knowledge Embedding Layer
        embedded = self.embedding_layer(x)
        if return_hidden:
            hidden_states.append(embedded)

        # 2. Knowledge-Guided Interaction Layers
        hidden = embedded
        for layer in self.hidden_layers:
            hidden = layer(hidden)
            if return_hidden:
                hidden_states.append(hidden)

        # 3. Attention Mechanism
        attended, attention_weights = self.attention(hidden)

        # 4. Interpretable Output Layer
        predictions, interpretability_info = self.output_layer(attended)

        # Prepare outputs
        outputs['predictions'] = predictions

        if return_attention:
            outputs['attention_weights'] = attention_weights

        if return_hidden:
            outputs['hidden_representations'] = hidden_states

        outputs['interpretability_info'] = interpretability_info

        return outputs

    def get_feature_importance(self,
                              x: torch.Tensor,
                              micronutrient_idx: Optional[int] = None) -> Dict:
        """
        Calculate feature importance scores for interpretability.

        Args:
            x: Input tensor
            micronutrient_idx: Optional index of specific micronutrient

        Returns:
            Dictionary with feature importance scores
        """
        self.eval()
        with torch.no_grad():
            # Get attention weights
            outputs = self.forward(x, return_attention=True)
            attention_weights = outputs['attention_weights']

            # Get output layer weights
            importance_dict = {}

            if micronutrient_idx is not None:
                # Importance for specific micronutrient
                layer = self.output_layer.micronutrient_layers[micronutrient_idx]
                weights = layer.weight.squeeze().abs()
                importance = weights * attention_weights.mean(dim=0)
                importance_dict[self.micronutrient_names[micronutrient_idx]] = \
                    importance.cpu().numpy()
            else:
                # Importance for all micronutrients
                for idx, name in enumerate(self.micronutrient_names):
                    layer = self.output_layer.micronutrient_layers[idx]
                    weights = layer.weight.squeeze().abs()
                    importance = weights * attention_weights.mean(dim=0)
                    importance_dict[name] = importance.cpu().numpy()

        return importance_dict

    def get_counterfactual_explanations(self,
                                       x: torch.Tensor,
                                       target_micronutrient: int,
                                       num_features_to_change: int = 3) -> Dict:
        """
        Generate counterfactual explanations.

        Shows what changes in input would alter the prediction.

        Args:
            x: Input tensor (single sample)
            target_micronutrient: Index of micronutrient to explain
            num_features_to_change: Number of features to suggest changing

        Returns:
            Dictionary with counterfactual recommendations
        """
        self.eval()
        with torch.no_grad():
            # Get current prediction
            outputs = self.forward(x)
            current_pred = outputs['predictions'][0, target_micronutrient].item()

            # Get feature importance for this micronutrient
            importance = self.get_feature_importance(x, target_micronutrient)
            feature_scores = importance[self.micronutrient_names[target_micronutrient]]

            # Find top influential features
            top_indices = np.argsort(np.abs(feature_scores))[-num_features_to_change:]

            counterfactuals = {
                'current_prediction': current_pred,
                'micronutrient': self.micronutrient_names[target_micronutrient],
                'recommended_changes': []
            }

            # Generate recommendations
            for idx in top_indices:
                direction = "increase" if feature_scores[idx] > 0 else "decrease"
                counterfactuals['recommended_changes'].append({
                    'feature_index': int(idx),
                    'importance': float(feature_scores[idx]),
                    'recommendation': direction
                })

        return counterfactuals

    def explain_prediction(self,
                          x: torch.Tensor,
                          feature_names: Optional[List[str]] = None) -> Dict:
        """
        Provide comprehensive explanation for a prediction.

        Args:
            x: Input tensor (single sample)
            feature_names: Optional names of input features

        Returns:
            Dictionary with comprehensive explanation
        """
        self.eval()
        with torch.no_grad():
            # Get full forward pass with all intermediate outputs
            outputs = self.forward(x, return_attention=True, return_hidden=True)

            explanation = {
                'predictions': {},
                'attention_weights': outputs['attention_weights'][0].cpu().numpy(),
                'feature_contributions': outputs['interpretability_info']['feature_contributions']
            }

            # Add predictions with confidence levels
            predictions = outputs['predictions'][0]
            for idx, name in enumerate(self.micronutrient_names):
                prob = predictions[idx].item()
                explanation['predictions'][name] = {
                    'probability': prob,
                    'deficiency_risk': 'High' if prob > 0.7 else 'Medium' if prob > 0.4 else 'Low',
                    'confidence': max(prob, 1 - prob)
                }

            # Add feature names if provided
            if feature_names:
                explanation['feature_names'] = feature_names

            return explanation

    def get_model_summary(self) -> str:
        """
        Get a summary of the model architecture.

        Returns:
            String description of model structure
        """
        summary = "KGNN Model Architecture:\n"
        summary += "=" * 50 + "\n"
        summary += f"Input dimension: {self.input_dim}\n"
        summary += f"Embedding dimension: {self.embedding_dim}\n"
        summary += f"Hidden layers: {self.hidden_dims}\n"
        summary += f"Number of micronutrients: {self.num_micronutrients}\n"
        summary += f"Micronutrients: {', '.join(self.micronutrient_names)}\n"
        summary += "=" * 50 + "\n"
        summary += f"Total parameters: {sum(p.numel() for p in self.parameters())}\n"
        summary += f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}\n"
        return summary


if __name__ == "__main__":
    # Test the KGNN model
    print("Testing KGNN Model...")

    # Model parameters
    input_dim = 105
    embedding_dim = 128
    hidden_dims = [256, 128, 64]
    num_micronutrients = 12
    batch_size = 32

    micronutrient_names = [
        "Vitamin_B12", "Vitamin_D", "Vitamin_B6", "Folate",
        "Iron", "Calcium", "Magnesium", "Zinc",
        "Vitamin_C", "Vitamin_E", "Vitamin_A", "Selenium"
    ]

    # Create model
    model = KGNN(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        num_micronutrients=num_micronutrients,
        micronutrient_names=micronutrient_names
    )

    # Print model summary
    print(model.get_model_summary())

    # Test forward pass
    x = torch.randn(batch_size, input_dim)
    outputs = model(x, return_attention=True, return_hidden=True)

    print(f"\nInput shape: {x.shape}")
    print(f"Predictions shape: {outputs['predictions'].shape}")
    print(f"Attention weights shape: {outputs['attention_weights'].shape}")
    print(f"Number of hidden layers: {len(outputs['hidden_representations'])}")

    # Test feature importance
    importance = model.get_feature_importance(x[:1])
    print(f"\nFeature importance calculated for {len(importance)} micronutrients")

    # Test explanation
    explanation = model.explain_prediction(x[:1])
    print(f"\nExplanation generated with predictions for {len(explanation['predictions'])} micronutrients")

    print("\nKGNN Model test completed successfully!")
