"""
Custom neural network layers for KGNN architecture.

Implements the specialized layers described in the paper:
- Knowledge Embedding Layer (Equation 1)
- Knowledge-Guided Interaction Layers (Equation 2)
- Attention Mechanism (Equations 3-4)
- Interpretable Output Layer (Equation 5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class KnowledgeEmbeddingLayer(nn.Module):
    """
    Knowledge Embedding Layer from Equation 1.

    Transforms raw nutritional data into semantically meaningful
    representation space guided by nutritional ontology.

    φ(x) = σ(W_e × x + b_e)
    """

    def __init__(self,
                 input_dim: int,
                 embedding_dim: int,
                 initial_weights: Optional[torch.Tensor] = None,
                 dropout_rate: float = 0.3,
                 activation: str = 'relu'):
        """
        Initialize knowledge embedding layer.

        Args:
            input_dim: Number of input features
            embedding_dim: Dimension of embedding space
            initial_weights: Knowledge-constrained initial weights (W_e)
            dropout_rate: Dropout probability
            activation: Activation function ('relu', 'tanh', 'elu')
        """
        super(KnowledgeEmbeddingLayer, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Linear transformation W_e × x + b_e
        self.linear = nn.Linear(input_dim, embedding_dim)

        # Initialize with knowledge-constrained weights if provided
        if initial_weights is not None:
            with torch.no_grad():
                self.linear.weight.copy_(initial_weights)

        # Activation function σ
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()

        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: φ(x) = σ(W_e × x + b_e)

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Embedded representation of shape (batch_size, embedding_dim)
        """
        # Linear transformation
        embedded = self.linear(x)

        # Batch normalization
        embedded = self.batch_norm(embedded)

        # Non-linear activation
        embedded = self.activation(embedded)

        # Dropout for regularization
        embedded = self.dropout(embedded)

        return embedded


class KnowledgeGuidedLayer(nn.Module):
    """
    Knowledge-Guided Interaction Layer from Equation 2.

    Structures connections according to established nutritional principles
    using mask matrix M_l.

    h_l = σ(M_l ⊙ (W_l × h_{l-1}) + b_l)
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 mask_matrix: Optional[torch.Tensor] = None,
                 dropout_rate: float = 0.3,
                 activation: str = 'relu',
                 use_batch_norm: bool = True):
        """
        Initialize knowledge-guided layer.

        Args:
            input_dim: Number of input neurons
            output_dim: Number of output neurons
            mask_matrix: Knowledge-based mask matrix M_l
            dropout_rate: Dropout probability
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
        """
        super(KnowledgeGuidedLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Learnable weights W_l
        self.linear = nn.Linear(input_dim, output_dim)

        # Knowledge-based mask M_l
        if mask_matrix is not None:
            self.register_buffer('mask', mask_matrix)
        else:
            # Default to fully connected if no mask provided
            self.register_buffer('mask', torch.ones(output_dim, input_dim))

        # Activation function σ
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()

        self.dropout = nn.Dropout(dropout_rate)
        self.use_batch_norm = use_batch_norm

        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: h_l = σ(M_l ⊙ (W_l × h_{l-1}) + b_l)

        Args:
            h: Input activations of shape (batch_size, input_dim)

        Returns:
            Output activations of shape (batch_size, output_dim)
        """
        # Apply linear transformation W_l × h_{l-1}
        output = self.linear(h)

        # Apply mask M_l ⊙ (W_l × h_{l-1})
        # The mask is applied to the weights during computation
        with torch.no_grad():
            self.linear.weight.data *= self.mask

        # Batch normalization (optional)
        if self.use_batch_norm:
            output = self.batch_norm(output)

        # Non-linear activation σ
        output = self.activation(output)

        # Dropout for regularization
        output = self.dropout(output)

        return output


class AttentionMechanism(nn.Module):
    """
    Attention Mechanism from Equations 3-4.

    Highlights influential factors for each prediction to enhance
    interpretability.

    α = softmax(v^T × tanh(W_a × h_l + b_a))
    z = Σ_i α_i × h_{l,i}
    """

    def __init__(self,
                 hidden_dim: int,
                 attention_dim: int = 64):
        """
        Initialize attention mechanism.

        Args:
            hidden_dim: Dimension of hidden representation h_l
            attention_dim: Dimension of attention space
        """
        super(AttentionMechanism, self).__init__()

        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim

        # Attention projection W_a × h_l + b_a
        self.attention_projection = nn.Linear(hidden_dim, attention_dim)

        # Attention vector v
        self.attention_vector = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing attention weights and attended representation.

        Args:
            h: Hidden representation of shape (batch_size, hidden_dim)

        Returns:
            Tuple of:
                - z: Attended representation (batch_size, hidden_dim)
                - α: Attention weights (batch_size, hidden_dim)
        """
        # Project to attention space: W_a × h_l + b_a
        attention_scores = self.attention_projection(h)

        # Apply tanh activation
        attention_scores = torch.tanh(attention_scores)

        # Compute attention weights: v^T × tanh(...)
        attention_weights = self.attention_vector(attention_scores)

        # Apply softmax to get normalized weights α
        # We need to handle the feature dimension properly
        # Reshape for proper softmax application
        batch_size = h.size(0)
        attention_weights = attention_weights.squeeze(-1)
        attention_weights = torch.softmax(attention_weights.unsqueeze(1), dim=1)

        # Compute attended representation: z = Σ_i α_i × h_{l,i}
        # For simplicity, we use the attention as feature weights
        z = h * attention_weights

        return z, attention_weights.squeeze(1)


class InterpretableOutputLayer(nn.Module):
    """
    Interpretable Output Layer from Equation 5.

    Produces deficiency probabilities for each micronutrient with
    transparent decision pathways.

    p(d_m | x) = σ(w_m^T × z + b_m)
    """

    def __init__(self,
                 input_dim: int,
                 num_micronutrients: int,
                 micronutrient_names: Optional[list] = None):
        """
        Initialize interpretable output layer.

        Args:
            input_dim: Dimension of attended representation z
            num_micronutrients: Number of micronutrients to predict
            micronutrient_names: Names of micronutrients (for interpretability)
        """
        super(InterpretableOutputLayer, self).__init__()

        self.input_dim = input_dim
        self.num_micronutrients = num_micronutrients
        self.micronutrient_names = micronutrient_names or \
                                   [f"Micronutrient_{i}" for i in range(num_micronutrients)]

        # Separate linear layer for each micronutrient
        # This allows us to interpret each prediction independently
        self.micronutrient_layers = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(num_micronutrients)
        ])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass: p(d_m | x) = σ(w_m^T × z + b_m)

        Args:
            z: Attended representation of shape (batch_size, input_dim)

        Returns:
            Tuple of:
                - predictions: Deficiency probabilities (batch_size, num_micronutrients)
                - interpretability_info: Dictionary with interpretation details
        """
        predictions = []
        feature_contributions = {}

        for idx, layer in enumerate(self.micronutrient_layers):
            # Compute logit: w_m^T × z + b_m
            logit = layer(z)

            # Apply sigmoid: p(d_m | x) = σ(logit)
            prob = torch.sigmoid(logit)

            predictions.append(prob)

            # Store weight information for interpretability
            with torch.no_grad():
                feature_contributions[self.micronutrient_names[idx]] = \
                    layer.weight.squeeze().cpu().numpy()

        # Stack predictions
        predictions = torch.cat(predictions, dim=1)

        # Prepare interpretability information
        interpretability_info = {
            'feature_contributions': feature_contributions,
            'micronutrient_names': self.micronutrient_names
        }

        return predictions, interpretability_info


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for enhanced interpretability.

    Allows the model to attend to different aspects of the input
    simultaneously, providing richer explanations.
    """

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 4,
                 dropout_rate: float = 0.1):
        """
        Initialize multi-head attention.

        Args:
            hidden_dim: Dimension of hidden representation
            num_heads: Number of attention heads
            dropout_rate: Dropout probability
        """
        super(MultiHeadAttention, self).__init__()

        assert hidden_dim % num_heads == 0, \
            "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Query, Key, Value projections
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.output = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, hidden_dim)

        Returns:
            Tuple of:
                - output: Attended representation
                - attention_weights: Attention weights for interpretability
        """
        batch_size = x.size(0)

        # Expand dimensions for self-attention
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)

        # Compute Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)

        # Reshape and project output
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, 1, self.hidden_dim)
        output = self.output(attended).squeeze(1)

        return output, attention_weights.mean(dim=1).squeeze()


if __name__ == "__main__":
    # Test the layers
    batch_size = 32
    input_dim = 105
    embedding_dim = 128
    hidden_dim = 64
    num_micronutrients = 12

    # Test Knowledge Embedding Layer
    print("Testing Knowledge Embedding Layer...")
    x = torch.randn(batch_size, input_dim)
    emb_layer = KnowledgeEmbeddingLayer(input_dim, embedding_dim)
    embedded = emb_layer(x)
    print(f"Input shape: {x.shape}, Embedded shape: {embedded.shape}")

    # Test Knowledge-Guided Layer
    print("\nTesting Knowledge-Guided Layer...")
    kg_layer = KnowledgeGuidedLayer(embedding_dim, hidden_dim)
    hidden = kg_layer(embedded)
    print(f"Embedded shape: {embedded.shape}, Hidden shape: {hidden.shape}")

    # Test Attention Mechanism
    print("\nTesting Attention Mechanism...")
    attn = AttentionMechanism(hidden_dim)
    z, weights = attn(hidden)
    print(f"Hidden shape: {hidden.shape}, Attended shape: {z.shape}")
    print(f"Attention weights shape: {weights.shape}")

    # Test Interpretable Output Layer
    print("\nTesting Interpretable Output Layer...")
    output_layer = InterpretableOutputLayer(hidden_dim, num_micronutrients)
    predictions, info = output_layer(z)
    print(f"Attended shape: {z.shape}, Predictions shape: {predictions.shape}")
    print(f"Feature contributions available for {len(info['feature_contributions'])} micronutrients")
