"""
Sparse Attention Mechanism

Implements the sparse attention mechanism as described in Section 3.5.

This module reduces computational complexity from O(n²) to O(k·n), where k << n,
by selectively activating only nutritionally significant connections.

Key equations:
- Attention (Eq. 11): A(i,j) = softmax_j(S(h_i, h_j, r_ij)) · δ(h_i, h_j, r_ij)
- Gating function (Eq. 12): δ(h_i, h_j, r_ij) = 1 if I(h_i, h_j, r_ij) > τ, else 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class ImportanceEstimator(nn.Module):
    """
    Estimates importance I(h_i, h_j, r_ij) for gating function δ.

    This module determines which connections are nutritionally significant
    and should be activated during message passing.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_heads: int = 1,
    ):
        """
        Args:
            hidden_dim: Dimension of node hidden states
            edge_dim: Dimension of edge features
            num_heads: Number of attention heads
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Importance scoring network
        self.importance_net = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(
        self,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        r_ij: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute importance scores for edges.

        Args:
            h_i: Target node features [num_edges, hidden_dim]
            h_j: Source node features [num_edges, hidden_dim]
            r_ij: Edge features [num_edges, edge_dim]

        Returns:
            Importance scores [num_edges, 1]
        """
        # Concatenate node and edge features
        combined = torch.cat([h_i, h_j, r_ij], dim=-1)

        # Compute importance
        importance = self.importance_net(combined)

        return importance


class SparseAttention(nn.Module):
    """
    Sparse attention mechanism with selective edge activation.

    Implements Equations (11) and (12) from the paper:

    Attention (Eq. 11):
        A(i,j) = softmax_j(S(h_i, h_j, r_ij)) · δ(h_i, h_j, r_ij)

    Gating function (Eq. 12):
        δ(h_i, h_j, r_ij) = 1 if I(h_i, h_j, r_ij) > τ, else 0

    This reduces complexity from O(n²) to O(k·n) where k is the average
    number of activated connections per node.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_heads: int = 4,
        threshold_tau: Optional[float] = None,
        learnable_threshold: bool = True,
        top_k: Optional[int] = None,
    ):
        """
        Args:
            hidden_dim: Dimension of node hidden states
            edge_dim: Dimension of edge features
            num_heads: Number of attention heads
            threshold_tau: Initial threshold value (default: 0.5)
            learnable_threshold: Whether threshold τ is learnable
            top_k: If set, keep only top-k edges per node (alternative to threshold)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.top_k = top_k

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Query, Key projections for attention scoring
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Edge importance function θ(r_ij)
        self.edge_importance = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.Tanh(),
        )

        # Importance estimator for gating
        self.importance_estimator = ImportanceEstimator(
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
        )

        # Learnable threshold τ
        if threshold_tau is None:
            threshold_tau = 0.5

        if learnable_threshold:
            self.tau = nn.Parameter(torch.tensor(threshold_tau))
        else:
            self.register_buffer('tau', torch.tensor(threshold_tau))

        # Output projection
        self.W_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass implementing sparse attention.

        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_features: Edge features [num_edges, edge_dim]
            return_attention: Whether to return attention weights

        Returns:
            Tuple of:
                - Attended features [num_nodes, hidden_dim]
                - Attention weights [num_edges] (if return_attention=True)
                - Active edge mask [num_edges] (if return_attention=True)
        """
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        # Split into source and target nodes
        row, col = edge_index  # row: target, col: source
        h_i = x[row]  # Target node features [num_edges, hidden_dim]
        h_j = x[col]  # Source node features [num_edges, hidden_dim]

        # Compute importance scores I(h_i, h_j, r_ij)
        importance = self.importance_estimator(h_i, h_j, edge_features)  # [num_edges, 1]
        importance = importance.squeeze(-1)  # [num_edges]

        # Compute gating function δ (Eq. 12)
        if self.top_k is not None:
            # Top-k gating: keep only top-k edges per target node
            active_edges = self._top_k_gating(importance, row, num_nodes)
        else:
            # Threshold-based gating
            active_edges = (importance > self.tau).float()

        # Compute attention scores S(h_i, h_j, r_ij) (Eq. 11)
        attention_scores = self._compute_attention_scores(
            h_i, h_j, edge_features
        )  # [num_edges, num_heads]

        # Apply gating (multiply by δ)
        active_edges_expanded = active_edges.unsqueeze(-1)  # [num_edges, 1]
        attention_scores = attention_scores * active_edges_expanded

        # Apply softmax per target node (Eq. 11)
        attention_weights = self._softmax_per_node(
            attention_scores, row, num_nodes
        )  # [num_edges, num_heads]

        # Apply attention to compute attended features
        attended_features = self._apply_attention(
            x, edge_index, attention_weights, active_edges
        )  # [num_nodes, hidden_dim]

        # Output projection
        out = self.W_out(attended_features)

        if return_attention:
            # Return average attention across heads
            avg_attention = attention_weights.mean(dim=-1)  # [num_edges]
            return out, avg_attention, active_edges
        else:
            return out, None, None

    def _compute_attention_scores(
        self,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        r_ij: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention scores S(h_i, h_j, r_ij).

        Implements the score function in Eq. 11:
            S(h_i, h_j, r_ij) = (W_q·h_i)^T · (W_k·h_j) · θ(r_ij)
        """
        # Project nodes
        q = self.W_q(h_i)  # [num_edges, hidden_dim]
        k = self.W_k(h_j)  # [num_edges, hidden_dim]

        # Transform edge features
        edge_importance = self.edge_importance(r_ij)  # [num_edges, hidden_dim]

        # Reshape for multi-head attention
        q = q.view(-1, self.num_heads, self.head_dim)  # [num_edges, num_heads, head_dim]
        k = k.view(-1, self.num_heads, self.head_dim)
        edge_importance = edge_importance.view(-1, self.num_heads, self.head_dim)

        # Compute scores with edge modulation
        scores = (q * k * edge_importance).sum(dim=-1)  # [num_edges, num_heads]

        # Scale by sqrt(d_k) for stability
        scores = scores / math.sqrt(self.head_dim)

        return scores

    def _softmax_per_node(
        self,
        scores: torch.Tensor,
        row: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Apply softmax over incoming edges for each node.

        Args:
            scores: Attention scores [num_edges, num_heads]
            row: Target node indices [num_edges]
            num_nodes: Total number of nodes

        Returns:
            Normalized attention weights [num_edges, num_heads]
        """
        # Compute max per node for numerical stability
        max_scores = torch.full(
            (num_nodes, self.num_heads),
            float('-inf'),
            device=scores.device,
        )
        max_scores.scatter_reduce_(
            0,
            row.unsqueeze(-1).expand(-1, self.num_heads),
            scores,
            reduce='amax',
            include_self=False,
        )

        # Subtract max and exponentiate
        scores_shifted = scores - max_scores[row]
        exp_scores = torch.exp(scores_shifted)

        # Sum per node
        sum_exp = torch.zeros(num_nodes, self.num_heads, device=scores.device)
        sum_exp.scatter_add_(
            0,
            row.unsqueeze(-1).expand(-1, self.num_heads),
            exp_scores,
        )

        # Normalize
        attention = exp_scores / (sum_exp[row] + 1e-8)

        return attention

    def _top_k_gating(
        self,
        importance: torch.Tensor,
        row: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Keep only top-k most important edges per node.

        Args:
            importance: Importance scores [num_edges]
            row: Target node indices [num_edges]
            num_nodes: Total number of nodes

        Returns:
            Binary mask [num_edges]
        """
        # Create mask
        mask = torch.zeros_like(importance)

        # For each node, find top-k edges
        for node_id in range(num_nodes):
            node_edges = (row == node_id)
            if node_edges.sum() == 0:
                continue

            node_importance = importance[node_edges]
            k = min(self.top_k, node_importance.size(0))

            # Get top-k indices
            _, top_indices = torch.topk(node_importance, k)

            # Convert to global indices
            global_indices = torch.where(node_edges)[0][top_indices]
            mask[global_indices] = 1.0

        return mask

    def _apply_attention(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        attention_weights: torch.Tensor,
        active_edges: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply attention weights to aggregate neighbor features.

        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            attention_weights: Attention weights [num_edges, num_heads]
            active_edges: Binary mask for active edges [num_edges]

        Returns:
            Aggregated features [num_nodes, hidden_dim]
        """
        num_nodes = x.size(0)
        row, col = edge_index

        # Get source node features
        x_j = x[col]  # [num_edges, hidden_dim]

        # Reshape for multi-head attention
        x_j = x_j.view(-1, self.num_heads, self.head_dim)  # [num_edges, num_heads, head_dim]

        # Apply attention weights
        weighted = x_j * attention_weights.unsqueeze(-1)  # [num_edges, num_heads, head_dim]

        # Apply active edge mask
        weighted = weighted * active_edges.unsqueeze(-1).unsqueeze(-1)

        # Aggregate messages per node
        out = torch.zeros(
            num_nodes, self.num_heads, self.head_dim,
            device=x.device,
        )
        out.scatter_add_(
            0,
            row.view(-1, 1, 1).expand(-1, self.num_heads, self.head_dim),
            weighted,
        )

        # Concatenate heads
        out = out.view(num_nodes, self.hidden_dim)

        return out

    def get_active_edge_ratio(self, active_edges: torch.Tensor) -> float:
        """
        Compute ratio of active edges (for monitoring efficiency).

        Args:
            active_edges: Binary mask [num_edges]

        Returns:
            Ratio of active edges
        """
        return active_edges.sum().item() / active_edges.size(0)


class MultiHeadSparseAttention(nn.Module):
    """
    Multi-head sparse attention with residual connections.

    Combines multiple sparse attention heads with layer normalization
    and residual connections for stable training.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        threshold_tau: Optional[float] = None,
        top_k: Optional[int] = None,
    ):
        """
        Args:
            hidden_dim: Dimension of node hidden states
            edge_dim: Dimension of edge features
            num_heads: Number of attention heads
            dropout: Dropout rate
            threshold_tau: Threshold for sparse attention
            top_k: If set, keep only top-k edges per node
        """
        super().__init__()

        self.attention = SparseAttention(
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            threshold_tau=threshold_tau,
            top_k=top_k,
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with residual connections.

        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_features: Edge features [num_edges, edge_dim]
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (output features, attention weights, active edges)
        """
        # Attention with residual
        attn_out, attn_weights, active_edges = self.attention(
            x, edge_index, edge_features, return_attention=return_attention
        )
        x = self.layer_norm(x + self.dropout(attn_out))

        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + self.dropout(ffn_out))

        return x, attn_weights, active_edges
