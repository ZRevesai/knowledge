"""
Edge-Conditioned Message Passing

Implements the edge-conditioned message passing mechanism as described in Section 3.2.

This module provides message passing that accounts for heterogeneous nutritional interactions:
- Synergistic effects (e.g., vitamin C enhancing iron absorption)
- Antagonistic interactions (e.g., calcium inhibiting zinc absorption)
- Threshold-dependent effects (e.g., toxicity at high doses)

Key equations:
- Message function (Eq. 2): M(i,j) = σ(W1·h_i + W2·h_j ⊙ φ(r_ij) + W3·c)
- Node update (Eq. 3): h_i^(t+1) = h_i^(t) + α·Σ_{j∈N(i)} M(i,j)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class EdgeTransformationNetwork(nn.Module):
    """
    Learned edge transformation function φ(r_ij).

    Transforms edge features to modulate message passing based on interaction type.
    """

    def __init__(self, edge_dim: int, hidden_dim: int):
        """
        Args:
            edge_dim: Dimension of edge features
            hidden_dim: Dimension of hidden representation
        """
        super().__init__()
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        self.transform = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),  # Bounded output for stable multiplication
        )

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Transform edge features.

        Args:
            edge_features: [num_edges, edge_dim]

        Returns:
            Transformed features: [num_edges, hidden_dim]
        """
        return self.transform(edge_features)


class ContextEncoder(nn.Module):
    """
    Encodes physiological context (age, pregnancy status, medications, etc.).

    This is crucial for modeling nutrition in vulnerable populations.
    """

    def __init__(self, context_dim: int, hidden_dim: int):
        """
        Args:
            context_dim: Dimension of context features
            hidden_dim: Dimension of hidden representation
        """
        super().__init__()
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Encode context vector.

        Args:
            context: [batch_size, context_dim]

        Returns:
            Encoded context: [batch_size, hidden_dim]
        """
        return self.encoder(context)


class EdgeConditionedMessagePassing(MessagePassing):
    """
    Edge-conditioned message passing layer.

    Implements Equations (2) and (3) from the paper:

    Message function (Eq. 2):
        M(i,j) = σ(W1·h_i + W2·h_j ⊙ φ(r_ij) + W3·c)

    Node update (Eq. 3):
        h_i^(t+1) = h_i^(t) + α·Σ_{j∈N(i)} M(i,j)
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        context_dim: int = 0,
        aggr: str = "add",
        learnable_alpha: bool = True,
        add_self_loops_flag: bool = True,
    ):
        """
        Args:
            hidden_dim: Dimension of node hidden states
            edge_dim: Dimension of edge features
            context_dim: Dimension of context features (0 if no context)
            aggr: Aggregation scheme ('add', 'mean', 'max')
            learnable_alpha: Whether scaling factor α is learnable
            add_self_loops_flag: Whether to add self-loops to the graph
        """
        super().__init__(aggr=aggr)

        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.context_dim = context_dim
        self.add_self_loops_flag = add_self_loops_flag

        # Weight matrices (Eq. 2)
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)  # Source node
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)  # Target node
        self.W3 = nn.Linear(hidden_dim, hidden_dim, bias=True) if context_dim > 0 else None

        # Edge transformation function φ(r_ij)
        self.edge_transform = EdgeTransformationNetwork(edge_dim, hidden_dim)

        # Context encoder
        if context_dim > 0:
            self.context_encoder = ContextEncoder(context_dim, hidden_dim)
        else:
            self.context_encoder = None

        # Scaling factor α (Eq. 3)
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('alpha', torch.tensor(1.0))

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        active_edges: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass implementing edge-conditioned message passing.

        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_features: Edge features [num_edges, edge_dim]
            context: Context vector [batch_size, context_dim] or [num_nodes, context_dim]
            active_edges: Binary mask for active edges [num_edges] (for lazy evaluation)

        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        # Add self-loops to the edge index
        if self.add_self_loops_flag:
            edge_index, edge_features = self._add_self_loops(
                edge_index, edge_features, x.size(0)
            )

        # Apply active edge mask (for lazy evaluation optimization)
        if active_edges is not None:
            edge_index = edge_index[:, active_edges]
            edge_features = edge_features[active_edges]

        # Encode context if provided
        if context is not None and self.context_encoder is not None:
            context_encoded = self.context_encoder(context)
            # Broadcast context to all nodes if needed
            if context_encoded.size(0) != x.size(0):
                context_encoded = context_encoded.expand(x.size(0), -1)
        else:
            context_encoded = None

        # Propagate messages
        out = self.propagate(
            edge_index,
            x=x,
            edge_features=edge_features,
            context=context_encoded,
        )

        # Residual connection with scaling (Eq. 3)
        x_updated = x + self.alpha * out

        # Layer normalization
        x_updated = self.layer_norm(x_updated)

        return x_updated

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_features: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Construct messages from node j to node i.

        Implements Equation (2):
            M(i,j) = σ(W1·h_i + W2·h_j ⊙ φ(r_ij) + W3·c)

        Args:
            x_i: Target node features [num_edges, hidden_dim]
            x_j: Source node features [num_edges, hidden_dim]
            edge_features: Edge features [num_edges, edge_dim]
            context: Context features [num_edges, hidden_dim]

        Returns:
            Messages [num_edges, hidden_dim]
        """
        # Transform edge features
        edge_transformed = self.edge_transform(edge_features)  # [num_edges, hidden_dim]

        # Compute message components
        source_component = self.W1(x_i)  # W1·h_i
        target_component = self.W2(x_j) * edge_transformed  # W2·h_j ⊙ φ(r_ij)

        # Combine components
        message = source_component + target_component

        # Add context if available
        if context is not None and self.W3 is not None:
            # Expand context to match number of edges if needed
            if context.size(0) != message.size(0):
                context = context[0].unsqueeze(0).expand(message.size(0), -1)
            context_component = self.W3(context)
            message = message + context_component

        # Apply activation
        message = F.relu(message)

        return message

    def _add_self_loops(
        self,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add self-loops to the graph"""
        # Add self-loop edges to edge_index
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # Create self-loop edge features (neutral interaction)
        num_self_loops = num_nodes
        self_loop_features = torch.zeros(
            num_self_loops, self.edge_dim, device=edge_features.device
        )
        # Mark as neutral interaction
        self_loop_features[:, 3] = 1.0  # Neutral type (assuming one-hot encoding)
        self_loop_features[:, 4] = 1.0  # Full strength for self-connection

        # Concatenate with existing edge features
        edge_features = torch.cat([edge_features, self_loop_features], dim=0)

        return edge_index, edge_features


class MultiStepMessagePassing(nn.Module):
    """
    Multi-step message passing with early stopping.

    Implements iterative message passing with convergence detection (Eq. 6).
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        context_dim: int = 0,
        max_steps: int = 5,
        min_steps: int = 2,
        convergence_threshold: float = 0.01,
    ):
        """
        Args:
            hidden_dim: Dimension of node hidden states
            edge_dim: Dimension of edge features
            context_dim: Dimension of context features
            max_steps: Maximum number of message passing steps
            min_steps: Minimum number of message passing steps
            convergence_threshold: Threshold γ for early stopping (Eq. 6)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.convergence_threshold = convergence_threshold

        # Message passing layers (shared weights across steps)
        self.mp_layer = EdgeConditionedMessagePassing(
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            context_dim=context_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        active_edges: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Perform multi-step message passing with early stopping.

        Implements Equation (6):
            ||h^(t+1) - h^(t)||_2 < γ

        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_features: Edge features [num_edges, edge_dim]
            context: Context vector
            active_edges: Binary mask for active edges

        Returns:
            Tuple of (final node features, number of steps taken)
        """
        h = x
        num_steps = 0

        for t in range(self.max_steps):
            h_prev = h

            # Message passing step
            h = self.mp_layer(
                h, edge_index, edge_features, context, active_edges
            )

            num_steps = t + 1

            # Check convergence (Eq. 6)
            if t >= self.min_steps:
                delta = torch.norm(h - h_prev, p=2)
                if delta < self.convergence_threshold:
                    break

        return h, num_steps
