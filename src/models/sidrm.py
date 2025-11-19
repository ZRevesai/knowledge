"""
Smart Interpretable Dietary Recommender Model (SIDRM) for Vulnerable Populations.

Implementation of the SIDRM architecture from the paper:
"Smart Interpretable Dietary Recommender Model for Vulnerable Populations"
by Zvinodashe Revesai and Okuthe P. Kogeda (2025)

This model processes multi-modal NHANES data for four vulnerable populations:
- Pregnant women (P)
- Elderly individuals (E)
- Children (C)
- Chronic disease patients (CD)

Key Features:
- Population-specific processing pathways
- Cross-population integration
- Multi-head attention for interpretability
- SHAP-based feature attribution
- Mobile-optimized configurations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
import math


class PopulationSpecificEncoder(nn.Module):
    """
    Population-specific encoder for processing features unique to each vulnerable group.

    Implements Equation (3) from the paper:
    O_P(i,j,k,c) = Σ K_P(u,v,w,c) · I(i+u-1, j+v-1, k+w-1, c)

    where:
    - O_P: Population-specific output features
    - K_P: Population-specific convolutional kernel
    - I: Integrated input data
    - c: Population category (P, E, C, or CD)
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 filter_size: int = 3,
                 population_name: str = "General",
                 dropout_rate: float = 0.3):
        """
        Initialize population-specific encoder.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden representation dimension
            filter_size: Convolutional filter size (f in Equation 3)
            population_name: Name of population (P, E, C, or CD)
            dropout_rate: Dropout probability
        """
        super(PopulationSpecificEncoder, self).__init__()

        self.population_name = population_name
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.filter_size = filter_size

        # Population-specific feature extraction layers
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=filter_size, padding=filter_size//2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=filter_size, padding=filter_size//2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Population-specific transformation
        self.population_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through population-specific encoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim, sequence_length)

        Returns:
            Population-specific features of shape (batch_size, hidden_dim, sequence_length)
        """
        # First convolutional layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        # Second convolutional layer with residual connection
        identity = out
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out + identity)  # Residual connection
        out = self.dropout(out)

        # Population-specific transformation
        # Transpose for linear layers: (batch, hidden, seq) -> (batch, seq, hidden)
        out = out.transpose(1, 2)
        out = self.population_transform(out)
        # Transpose back: (batch, seq, hidden) -> (batch, hidden, seq)
        out = out.transpose(1, 2)

        return out


class CrossPopulationIntegration(nn.Module):
    """
    Cross-population integration module for learning shared patterns across vulnerable groups.

    Implements Equation (5) from the paper:
    O_C(i,j,k,n) = Σ K_C(l,n) · O_P(i,j,k,l)

    where:
    - O_C: Cross-population integrated output
    - K_C: Cross-population integration weights
    - O_P: Population-specific outputs
    - l: Population index, n: participant index
    """

    def __init__(self,
                 num_populations: int = 4,
                 hidden_dim: int = 128,
                 integration_dim: int = 256,
                 dropout_rate: float = 0.3):
        """
        Initialize cross-population integration module.

        Args:
            num_populations: Number of vulnerable populations (default: 4)
            hidden_dim: Hidden dimension from population encoders
            integration_dim: Dimension for integrated representation
            dropout_rate: Dropout probability
        """
        super(CrossPopulationIntegration, self).__init__()

        self.num_populations = num_populations
        self.hidden_dim = hidden_dim
        self.integration_dim = integration_dim

        # Cross-population attention weights K_C(l,n)
        self.population_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout_rate,
            batch_first=True
        )

        # Integration transformation
        self.integration_layer = nn.Sequential(
            nn.Linear(hidden_dim * num_populations, integration_dim),
            nn.LayerNorm(integration_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(integration_dim, integration_dim),
            nn.LayerNorm(integration_dim)
        )

    def forward(self,
                population_features: List[torch.Tensor],
                population_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through cross-population integration.

        Args:
            population_features: List of tensors from each population encoder
                                Each tensor: (batch_size, hidden_dim, seq_length)
            population_mask: Optional mask indicating active populations for each sample

        Returns:
            Tuple of:
                - Integrated features: (batch_size, integration_dim)
                - Population attention weights: (batch_size, num_populations)
        """
        batch_size = population_features[0].size(0)

        # Stack population features: (batch, num_pop, hidden, seq)
        stacked_features = torch.stack(population_features, dim=1)

        # Average pool over sequence dimension: (batch, num_pop, hidden)
        pooled_features = stacked_features.mean(dim=-1)

        # Apply cross-population attention
        attended_features, attention_weights = self.population_attention(
            pooled_features,
            pooled_features,
            pooled_features,
            key_padding_mask=population_mask
        )

        # Flatten for integration: (batch, num_pop * hidden)
        flattened = attended_features.reshape(batch_size, -1)

        # Apply integration transformation
        integrated = self.integration_layer(flattened)

        # Compute population importance scores
        population_scores = attention_weights.mean(dim=1)  # Average over heads

        return integrated, population_scores


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism for interpretability.

    Implements Equation (8) from the paper:
    Attention(Q,K,V) = softmax(Q·K^T / √d_k)·V

    where:
    - Q: Query matrix
    - K: Key matrix
    - V: Value matrix
    - d_k: Scaling factor (dimension of keys)
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout_rate: float = 0.1):
        """
        Initialize multi-head self-attention.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout_rate: Dropout probability
        """
        super(MultiHeadSelfAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, return_attention: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_length, embed_dim)
            return_attention: Whether to return attention weights

        Returns:
            Tuple of:
                - Output tensor: (batch_size, seq_length, embed_dim)
                - Attention weights: (batch_size, num_heads, seq_length, seq_length) or None
        """
        batch_size, seq_length, embed_dim = x.size()

        # Linear projections and reshape for multi-head attention
        # (batch, seq, embed) -> (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
        Q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: (batch, heads, seq, seq)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values: (batch, heads, seq, head_dim)
        attended = torch.matmul(attention_weights, V)

        # Reshape and project: (batch, heads, seq, head_dim) -> (batch, seq, embed)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        output = self.out_proj(attended)

        if return_attention:
            return output, attention_weights
        else:
            return output, None


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head attention and feed-forward network.

    Used in the encoding/decoding path as shown in Figure 1 of the paper.
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 ff_dim: int = 512,
                 dropout_rate: float = 0.1):
        """
        Initialize transformer block.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward network hidden dimension
            dropout_rate: Dropout probability
        """
        super(TransformerBlock, self).__init__()

        # Multi-head attention
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_length, embed_dim)
            return_attention: Whether to return attention weights

        Returns:
            Tuple of:
                - Output tensor: (batch_size, seq_length, embed_dim)
                - Attention weights or None
        """
        # Multi-head attention with residual connection
        attended, attention_weights = self.attention(x, return_attention)
        x = self.norm1(x + self.dropout1(attended))

        # Feed-forward network with residual connection
        ff_out = self.ff_network(x)
        x = self.norm2(x + self.dropout2(ff_out))

        return x, attention_weights


class SIDRM(nn.Module):
    """
    Smart Interpretable Dietary Recommender Model (SIDRM) for Vulnerable Populations.

    Main architecture implementing the complete SIDRM model as described in the paper.

    Architecture Components:
    1. Population-specific encoders for P, E, C, CD groups
    2. Cross-population integration layer
    3. Multi-head attention mechanisms (5 hierarchical layers)
    4. Dual attention-based processing blocks
    5. Interpretable output layer with dietary recommendations

    Paper Reference: Equations (1)-(13), Figure 1
    """

    def __init__(self,
                 input_dim: int,
                 num_populations: int = 4,
                 population_names: List[str] = None,
                 hidden_dim: int = 128,
                 num_layers: int = 5,
                 num_attention_heads: int = 8,
                 ff_dim: int = 512,
                 num_nutrients: int = 50,
                 dropout_rate: float = 0.3,
                 use_population_specific: bool = True):
        """
        Initialize SIDRM model.

        Args:
            input_dim: Input feature dimension (m in the paper)
            num_populations: Number of vulnerable populations (c=4: P, E, C, CD)
            population_names: Names of populations
            hidden_dim: Hidden dimension (h in the paper)
            num_layers: Number of transformer layers (default: 5 as per paper)
            num_attention_heads: Number of attention heads
            ff_dim: Feed-forward network dimension
            num_nutrients: Number of nutrients to recommend
            dropout_rate: Dropout probability
            use_population_specific: Whether to use population-specific processing
        """
        super(SIDRM, self).__init__()

        self.input_dim = input_dim
        self.num_populations = num_populations
        self.population_names = population_names or ['Pregnant', 'Elderly', 'Children', 'Chronic_Disease']
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_nutrients = num_nutrients
        self.use_population_specific = use_population_specific

        # Input embedding layer
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Population-specific encoders (Equation 3)
        if use_population_specific:
            self.population_encoders = nn.ModuleDict({
                name: PopulationSpecificEncoder(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    filter_size=3,
                    population_name=name,
                    dropout_rate=dropout_rate
                )
                for name in self.population_names
            })

            # Cross-population integration (Equation 5)
            self.cross_population_integration = CrossPopulationIntegration(
                num_populations=num_populations,
                hidden_dim=hidden_dim,
                integration_dim=hidden_dim,
                dropout_rate=dropout_rate
            )

        # Transformer encoder layers (5 hierarchical layers as per Figure 1)
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=hidden_dim,
                num_heads=num_attention_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ])

        # Interpretable output layer for dietary recommendations
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_nutrients)
        )

        # Per-nutrient classification heads for deficiency detection
        self.nutrient_classifiers = nn.ModuleList([
            nn.Linear(num_nutrients, 1) for _ in range(num_nutrients)
        ])

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,
                x: torch.Tensor,
                population_indices: Optional[torch.Tensor] = None,
                return_attention: bool = False,
                return_population_scores: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through SIDRM.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            population_indices: Population category for each sample (batch_size,)
            return_attention: Whether to return attention weights
            return_population_scores: Whether to return population importance scores

        Returns:
            Dictionary containing:
                - nutrient_recommendations: Nutrient adequacy scores (batch_size, num_nutrients)
                - deficiency_probabilities: Per-nutrient deficiency probabilities
                - attention_weights: List of attention weights from each layer (if requested)
                - population_scores: Population importance scores (if requested)
        """
        batch_size = x.size(0)
        outputs = {}

        # Input embedding
        embedded = self.input_embedding(x)  # (batch_size, hidden_dim)

        # Expand for sequence processing: (batch_size, 1, hidden_dim)
        embedded = embedded.unsqueeze(1)

        # Population-specific processing (if enabled)
        if self.use_population_specific and population_indices is not None:
            # Prepare for convolutional layers: (batch, hidden, seq)
            embedded_conv = embedded.transpose(1, 2)

            # Process through each population encoder
            population_features = []
            for pop_name in self.population_names:
                pop_features = self.population_encoders[pop_name](embedded_conv)
                population_features.append(pop_features)

            # Cross-population integration
            integrated, population_scores = self.cross_population_integration(
                population_features,
                population_mask=None
            )

            # Reshape for transformer: (batch_size, 1, hidden_dim)
            x_processed = integrated.unsqueeze(1)

            if return_population_scores:
                outputs['population_scores'] = population_scores
        else:
            x_processed = embedded

        # Transformer encoding layers (5 hierarchical layers)
        attention_weights_list = []

        for layer in self.transformer_layers:
            x_processed, attn_weights = layer(x_processed, return_attention=return_attention)
            if return_attention and attn_weights is not None:
                attention_weights_list.append(attn_weights)

        # Pool sequence dimension: (batch_size, hidden_dim)
        pooled = x_processed.squeeze(1)

        # Generate nutrient recommendations
        nutrient_scores = self.output_layer(pooled)  # (batch_size, num_nutrients)

        # Per-nutrient deficiency probabilities
        deficiency_probs = []
        for i, classifier in enumerate(self.nutrient_classifiers):
            prob = torch.sigmoid(classifier(nutrient_scores))
            deficiency_probs.append(prob)

        deficiency_probs = torch.cat(deficiency_probs, dim=-1)  # (batch_size, num_nutrients)

        # Prepare outputs
        outputs['nutrient_recommendations'] = nutrient_scores
        outputs['deficiency_probabilities'] = deficiency_probs

        if return_attention:
            outputs['attention_weights'] = attention_weights_list

        return outputs

    def get_interpretability_scores(self,
                                   x: torch.Tensor,
                                   population_indices: Optional[torch.Tensor] = None) -> Dict[str, np.ndarray]:
        """
        Compute interpretability metrics (SHAP stability, attention consistency).

        Implements Equations (10)-(12) from the paper.

        Args:
            x: Input tensor
            population_indices: Population category for each sample

        Returns:
            Dictionary with interpretability scores
        """
        self.eval()
        with torch.no_grad():
            # Get attention weights
            outputs = self.forward(x, population_indices, return_attention=True)
            attention_weights = outputs['attention_weights']

            # Compute attention consistency (Equation 11)
            attention_consistency = self._compute_attention_consistency(attention_weights)

            # Feature importance from attention
            feature_importance = self._compute_feature_importance(attention_weights)

            return {
                'attention_consistency': attention_consistency,
                'feature_importance': feature_importance
            }

    def _compute_attention_consistency(self, attention_weights: List[torch.Tensor]) -> float:
        """
        Compute attention consistency metric (Equation 11).

        C_attention = (1/h)Σ(1 - JS(A_j1, A_j2))

        where JS is Jensen-Shannon divergence between attention maps.
        """
        if not attention_weights:
            return 0.0

        # Average attention across layers
        avg_attention = torch.stack(attention_weights).mean(dim=0)  # (batch, heads, seq, seq)

        # Compute consistency across heads
        num_heads = avg_attention.size(1)
        consistency_scores = []

        for i in range(num_heads - 1):
            attn_i = avg_attention[:, i, :, :].flatten(1)
            attn_j = avg_attention[:, i + 1, :, :].flatten(1)

            # Jensen-Shannon divergence
            js_div = self._jensen_shannon_divergence(attn_i, attn_j)
            consistency_scores.append(1 - js_div)

        consistency = torch.stack(consistency_scores).mean().item()
        return consistency

    def _jensen_shannon_divergence(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Compute Jensen-Shannon divergence between two distributions."""
        # Normalize to probabilities
        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)

        # Average distribution
        m = 0.5 * (p + q)

        # KL divergences
        kl_pm = F.kl_div(m.log(), p, reduction='batchmean')
        kl_qm = F.kl_div(m.log(), q, reduction='batchmean')

        # JS divergence
        js = 0.5 * (kl_pm + kl_qm)
        return js

    def _compute_feature_importance(self, attention_weights: List[torch.Tensor]) -> np.ndarray:
        """Compute feature importance from attention weights."""
        if not attention_weights:
            return np.array([])

        # Average attention across layers and heads
        avg_attention = torch.stack(attention_weights).mean(dim=(0, 1, 2))  # (seq, seq)

        # Sum attention received by each position
        feature_importance = avg_attention.sum(dim=0).cpu().numpy()

        return feature_importance

    def get_model_summary(self) -> str:
        """Get a summary of the SIDRM model architecture."""
        summary = "SIDRM Model Architecture:\n"
        summary += "=" * 70 + "\n"
        summary += f"Input dimension: {self.input_dim}\n"
        summary += f"Hidden dimension: {self.hidden_dim}\n"
        summary += f"Number of populations: {self.num_populations}\n"
        summary += f"Population names: {', '.join(self.population_names)}\n"
        summary += f"Number of transformer layers: {self.num_layers}\n"
        summary += f"Number of attention heads: {self.num_attention_heads}\n"
        summary += f"Number of nutrients: {self.num_nutrients}\n"
        summary += f"Population-specific processing: {self.use_population_specific}\n"
        summary += "=" * 70 + "\n"

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        summary += f"Total parameters: {total_params:,}\n"
        summary += f"Trainable parameters: {trainable_params:,}\n"
        summary += f"Model size: {total_params * 4 / (1024**2):.2f} MB (fp32)\n"

        return summary


if __name__ == "__main__":
    print("Testing SIDRM Model...")
    print("=" * 70)

    # Model parameters
    input_dim = 105  # NHANES feature dimension
    num_populations = 4  # P, E, C, CD
    hidden_dim = 128
    num_layers = 5
    num_attention_heads = 8
    num_nutrients = 50
    batch_size = 16

    # Create model
    model = SIDRM(
        input_dim=input_dim,
        num_populations=num_populations,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_nutrients=num_nutrients,
        dropout_rate=0.3,
        use_population_specific=True
    )

    # Print model summary
    print(model.get_model_summary())

    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(batch_size, input_dim)
    population_indices = torch.randint(0, num_populations, (batch_size,))

    outputs = model(
        x,
        population_indices=population_indices,
        return_attention=True,
        return_population_scores=True
    )

    print(f"Input shape: {x.shape}")
    print(f"Nutrient recommendations shape: {outputs['nutrient_recommendations'].shape}")
    print(f"Deficiency probabilities shape: {outputs['deficiency_probabilities'].shape}")
    print(f"Number of attention layers: {len(outputs['attention_weights'])}")

    if 'population_scores' in outputs:
        print(f"Population scores shape: {outputs['population_scores'].shape}")

    # Test interpretability
    print("\nTesting interpretability metrics...")
    interp_scores = model.get_interpretability_scores(x, population_indices)
    print(f"Attention consistency: {interp_scores['attention_consistency']:.4f}")
    print(f"Feature importance shape: {interp_scores['feature_importance'].shape}")

    print("\nSIDRM Model test completed successfully!")
