"""
NUTRINET: A Computationally Efficient Graph Neural Model
for Interpretable Nutrient Interaction Analysis

Main model implementation combining all components:
- Hierarchical graph representation
- Edge-conditioned message passing
- Sparse attention mechanism
- Transparent prediction module
- Computational efficiency optimizations

Reference:
Revesai, Z. and Kogeda, O. P. (2026). NUTRINET: A Computationally Efficient
Graph Neural Model for Interpretable Nutrient Interaction Analysis.
SAICSIT 2025, CCIS 2583, pp. 189-205.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .graph_construction import HierarchicalNutrientGraph
from .message_passing import MultiStepMessagePassing
from .attention import MultiHeadSparseAttention
from .transparent_prediction import TransparentPredictionModule
from .optimization import (
    QuantizedRepresentation,
    LazyEvaluation,
    EarlyStopping,
    AdaptiveComputation,
    EfficientEmbedding,
)


class NUTRINET(nn.Module):
    """
    NUTRINET: A Computationally Efficient Graph Neural Model

    Architecture overview:
    1. Input: Hierarchical nutrient graph G = (V, E, X, R)
    2. Knowledge Embedding: Transform inputs to semantic representations
    3. Edge-Conditioned Message Passing: Multi-step propagation with context
    4. Sparse Attention: Selective focus on nutritionally significant relationships
    5. Transparent Prediction: Interpretable output with explanations
    6. Efficiency Optimizations: Quantization, lazy evaluation, early stopping

    Args:
        input_dim: Dimension of input node features
        hidden_dim: Dimension of hidden representations (default: 64)
        edge_dim: Dimension of edge features (default: 16)
        context_dim: Dimension of context features (age, pregnancy, etc.) (default: 10)
        output_dim: Dimension of output predictions (default: 1)
        num_message_passing_steps: Maximum number of message passing iterations (default: 5)
        num_attention_heads: Number of attention heads (default: 4)
        enable_quantization: Enable feature quantization (default: True)
        num_quantization_bits: Number of bits for quantization (default: 8)
        enable_adaptive_dim: Enable adaptive dimension selection (default: True)
        sparse_attention_threshold: Threshold for sparse attention gating (default: 0.5)
        sparse_attention_top_k: Top-k for sparse attention (default: None)
        convergence_threshold: Threshold for early stopping (default: 0.01)
        node_names: Human-readable node names for explanations (default: None)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        edge_dim: int = 16,
        context_dim: int = 10,
        output_dim: int = 1,
        num_message_passing_steps: int = 5,
        num_attention_heads: int = 4,
        enable_quantization: bool = True,
        num_quantization_bits: int = 8,
        enable_adaptive_dim: bool = True,
        sparse_attention_threshold: Optional[float] = 0.5,
        sparse_attention_top_k: Optional[int] = None,
        convergence_threshold: float = 0.01,
        node_names: Optional[List[str]] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.context_dim = context_dim
        self.output_dim = output_dim
        self.num_message_passing_steps = num_message_passing_steps
        self.num_attention_heads = num_attention_heads

        # 1. Efficient embedding layer (with quantization and adaptive dimension)
        self.embedding = EfficientEmbedding(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            adaptive_dim=enable_adaptive_dim,
            quantize=enable_quantization,
            num_bits=num_quantization_bits,
        )

        # 2. Edge-conditioned message passing
        self.message_passing = MultiStepMessagePassing(
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            context_dim=context_dim,
            max_steps=num_message_passing_steps,
            min_steps=2,
            convergence_threshold=convergence_threshold,
        )

        # 3. Sparse attention mechanism
        self.sparse_attention = MultiHeadSparseAttention(
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_heads=num_attention_heads,
            dropout=0.1,
            threshold_tau=sparse_attention_threshold,
            top_k=sparse_attention_top_k,
        )

        # 4. Graph-level pooling (for graph-level predictions)
        self.graph_pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 5. Transparent prediction module
        self.prediction_module = TransparentPredictionModule(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            node_names=node_names,
            significance_threshold=0.1,
            max_path_length=5,
            max_paths=10,
        )

        # 6. Optimization modules
        self.lazy_evaluation = LazyEvaluation(activation_threshold=0.5)
        self.early_stopping = EarlyStopping(
            convergence_threshold=convergence_threshold,
            patience=2,
            min_iterations=2,
        )

        # Track efficiency metrics
        self.efficiency_metrics = {
            'num_iterations': 0,
            'active_edge_ratio': 0.0,
            'adaptive_dimension': hidden_dim,
        }

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_explanation: bool = False,
        source_nodes: Optional[List[int]] = None,
        target_nodes: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through NUTRINET.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_features: Edge features [num_edges, edge_dim]
            context: Context vector [batch_size, context_dim] (optional)
            return_attention: Whether to return attention weights
            return_explanation: Whether to generate explanations
            source_nodes: Source nodes for explanation (optional)
            target_nodes: Target nodes for explanation (optional)

        Returns:
            Dictionary containing:
                - 'predictions': Model predictions [num_nodes, output_dim]
                - 'node_embeddings': Final node embeddings [num_nodes, hidden_dim]
                - 'attention_weights': Attention weights (if return_attention=True)
                - 'active_edges': Active edge mask (if return_attention=True)
                - 'explanation': Explanation dict (if return_explanation=True)
                - 'efficiency_metrics': Dictionary of efficiency metrics
        """
        # Reset efficiency metrics
        self.early_stopping.reset()
        self.efficiency_metrics = {
            'num_iterations': 0,
            'active_edge_ratio': 0.0,
            'adaptive_dimension': self.hidden_dim,
        }

        # 1. Embed input features (with quantization and adaptive dimension)
        h = self.embedding(x, edge_index)

        # Track adaptive dimension
        self.efficiency_metrics['adaptive_dimension'] = h.size(-1)

        # 2. Edge-conditioned message passing with early stopping
        h, num_iterations = self.message_passing(
            x=h,
            edge_index=edge_index,
            edge_features=edge_features,
            context=context,
        )

        # Track number of iterations
        self.efficiency_metrics['num_iterations'] = num_iterations

        # 3. Sparse attention mechanism
        h_attn, attention_weights, active_edges = self.sparse_attention(
            x=h,
            edge_index=edge_index,
            edge_features=edge_features,
            return_attention=True,
        )

        # Track active edge ratio
        if active_edges is not None:
            self.efficiency_metrics['active_edge_ratio'] = (
                active_edges.sum().item() / active_edges.size(0)
            )

        # Combine message passing and attention outputs
        h = h + h_attn

        # 4. Graph-level pooling (optional, for graph-level predictions)
        # Here we use global mean pooling
        graph_embedding = h.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        graph_embedding = self.graph_pooling(graph_embedding)

        # 5. Make predictions
        # Node-level predictions
        node_predictions = self.prediction_module(h)

        # Graph-level prediction
        graph_prediction = self.prediction_module(None, graph_embedding)

        # Prepare output
        output = {
            'predictions': graph_prediction,
            'node_predictions': node_predictions,
            'node_embeddings': h,
            'graph_embedding': graph_embedding,
            'efficiency_metrics': self.efficiency_metrics,
        }

        # Add attention weights if requested
        if return_attention:
            output['attention_weights'] = attention_weights
            output['active_edges'] = active_edges

        # Generate explanation if requested
        if return_explanation:
            # Enable gradients for explanation
            x.requires_grad_(True)
            edge_features.requires_grad_(True)

            # Generate explanation
            explanation = self.prediction_module.explain(
                prediction=graph_prediction,
                node_features=h,
                edge_index=edge_index,
                edge_features=edge_features,
                source_nodes=source_nodes,
                target_nodes=target_nodes,
            )

            output['explanation'] = explanation

        return output

    def explain_prediction(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        source_nodes: Optional[List[int]] = None,
        target_nodes: Optional[List[int]] = None,
    ) -> Dict:
        """
        Generate explanation for a prediction.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_features: Edge features [num_edges, edge_dim]
            context: Context vector (optional)
            source_nodes: Source nodes for path analysis
            target_nodes: Target nodes for path analysis

        Returns:
            Explanation dictionary
        """
        # Forward pass with explanation
        output = self.forward(
            x=x,
            edge_index=edge_index,
            edge_features=edge_features,
            context=context,
            return_explanation=True,
            source_nodes=source_nodes,
            target_nodes=target_nodes,
        )

        return output['explanation']

    def get_efficiency_metrics(self) -> Dict[str, float]:
        """
        Get current efficiency metrics.

        Returns:
            Dictionary of efficiency metrics:
                - num_iterations: Number of message passing iterations
                - active_edge_ratio: Ratio of active edges in sparse attention
                - adaptive_dimension: Current hidden dimension
        """
        return self.efficiency_metrics

    def estimate_computational_cost(
        self,
        num_nodes: int,
        num_edges: int,
    ) -> Dict[str, float]:
        """
        Estimate computational cost for a graph.

        Args:
            num_nodes: Number of nodes
            num_edges: Number of edges

        Returns:
            Dictionary of cost estimates:
                - flops: Estimated floating-point operations
                - memory_mb: Estimated memory usage in MB
                - relative_to_full_attention: Cost relative to full O(n²) attention
        """
        # Estimate FLOPs
        d = self.hidden_dim
        k = int(num_edges / num_nodes)  # Average degree
        t = self.efficiency_metrics.get('num_iterations', 3)

        # Message passing: O(|E| * d²)
        mp_flops = num_edges * d * d * t

        # Sparse attention: O(k * n * d²) instead of O(n² * d²)
        sparse_attn_flops = k * num_nodes * d * d
        full_attn_flops = num_nodes * num_nodes * d * d

        total_flops = mp_flops + sparse_attn_flops

        # Estimate memory (in MB)
        node_memory = num_nodes * d * 4 / (1024 * 1024)  # 4 bytes per float
        edge_memory = num_edges * self.edge_dim * 4 / (1024 * 1024)
        total_memory = node_memory + edge_memory

        return {
            'flops': total_flops,
            'memory_mb': total_memory,
            'relative_to_full_attention': sparse_attn_flops / full_attn_flops,
        }


def create_nutrinet_model(
    num_micronutrients: int = 50,
    num_food_components: int = 30,
    num_health_outcomes: int = 20,
    feature_dim: int = 64,
    hidden_dim: int = 64,
    output_dim: int = 1,
    context_dim: int = 10,
    **kwargs
) -> Tuple[NUTRINET, HierarchicalNutrientGraph]:
    """
    Convenience function to create NUTRINET model with hierarchical graph.

    Args:
        num_micronutrients: Number of micronutrient nodes
        num_food_components: Number of food component nodes
        num_health_outcomes: Number of health outcome nodes
        feature_dim: Dimension of node features
        hidden_dim: Dimension of hidden representations
        output_dim: Dimension of output predictions
        context_dim: Dimension of context features
        **kwargs: Additional arguments for NUTRINET

    Returns:
        Tuple of (NUTRINET model, HierarchicalNutrientGraph)
    """
    # Create hierarchical graph
    graph = HierarchicalNutrientGraph(
        num_micronutrients=num_micronutrients,
        num_food_components=num_food_components,
        num_health_outcomes=num_health_outcomes,
        feature_dim=feature_dim,
    )

    # Build graph
    node_features, edge_index, edge_features = graph.build_graph()

    # Get node names for explanations
    node_names = [node.name for node in graph.nodes]

    # Create model
    model = NUTRINET(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        edge_dim=edge_features.size(-1),
        context_dim=context_dim,
        output_dim=output_dim,
        node_names=node_names,
        **kwargs
    )

    return model, graph


# Example usage
if __name__ == "__main__":
    # Create model and graph
    model, graph = create_nutrinet_model(
        num_micronutrients=20,
        num_food_components=15,
        num_health_outcomes=10,
        feature_dim=32,
        hidden_dim=64,
    )

    # Get graph data
    x, edge_index, edge_features = graph.node_features, graph.edge_index, graph.edge_features

    # Create dummy context (age, pregnancy status, medications, etc.)
    context = torch.randn(1, 10)

    # Forward pass
    output = model(
        x=x,
        edge_index=edge_index,
        edge_features=edge_features,
        context=context,
        return_attention=True,
        return_explanation=True,
    )

    print("NUTRINET Model Output:")
    print(f"- Prediction shape: {output['predictions'].shape}")
    print(f"- Node embeddings shape: {output['node_embeddings'].shape}")
    print(f"- Efficiency metrics: {output['efficiency_metrics']}")
    print(f"- Active edge ratio: {output['efficiency_metrics']['active_edge_ratio']:.2%}")
    print(f"- Number of iterations: {output['efficiency_metrics']['num_iterations']}")

    if 'explanation' in output:
        print(f"\nExplanation:")
        print(f"- Summary: {output['explanation']['summary']}")
        print(f"- Number of critical paths: {len(output['explanation']['critical_paths'])}")
        print(f"- Key nutrients: {[n['name'] for n in output['explanation']['key_nutrients'][:3]]}")

    # Estimate computational cost
    cost = model.estimate_computational_cost(
        num_nodes=x.size(0),
        num_edges=edge_index.size(1),
    )
    print(f"\nComputational Cost:")
    print(f"- FLOPs: {cost['flops']:,.0f}")
    print(f"- Memory: {cost['memory_mb']:.2f} MB")
    print(f"- Sparse vs Full Attention: {cost['relative_to_full_attention']:.2%}")
