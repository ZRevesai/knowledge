"""
Computational Efficiency Optimizations

Implements computational efficiency techniques as described in Section 3.3:
1. Quantized Representation (Eq. 4): Q(x) = round(x/s) · s
2. Lazy Evaluation (Eq. 5): Only compute along active paths
3. Early Stopping (Eq. 6): ||h^(t+1) - h^(t)||_2 < γ
4. Adaptive Computation (Eq. 7): d(x) = max(d_min, min(d_max, f(x)))

These optimizations reduce energy consumption by up to 73% while
maintaining predictive performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class QuantizedRepresentation(nn.Module):
    """
    Non-uniform quantization for node and edge features.

    Implements Equation (4):
        Q(x) = round(x/s) · s

    This reduces memory footprint while preserving essential information
    through adaptive scaling.
    """

    def __init__(
        self,
        num_bits: int = 8,
        adaptive_scale: bool = True,
        per_channel: bool = True,
    ):
        """
        Args:
            num_bits: Number of bits for quantization (4, 8, or 16)
            adaptive_scale: Whether to use adaptive scaling
            per_channel: Whether to scale per channel (vs. global)
        """
        super().__init__()

        self.num_bits = num_bits
        self.adaptive_scale = adaptive_scale
        self.per_channel = per_channel

        # Quantization levels
        self.num_levels = 2 ** num_bits
        self.min_val = 0
        self.max_val = self.num_levels - 1

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor.

        Args:
            x: Input tensor [..., feature_dim]

        Returns:
            Tuple of (quantized tensor, scale factors)
        """
        # Compute scale factors
        if self.adaptive_scale:
            if self.per_channel:
                # Per-channel scaling
                x_min = x.min(dim=tuple(range(x.dim() - 1)), keepdim=True)[0]
                x_max = x.max(dim=tuple(range(x.dim() - 1)), keepdim=True)[0]
            else:
                # Global scaling
                x_min = x.min()
                x_max = x.max()

            # Compute scale
            scale = (x_max - x_min) / (self.max_val - self.min_val + 1e-8)

            # Quantize (Eq. 4)
            x_normalized = (x - x_min) / (scale + 1e-8)
            x_quantized = torch.round(x_normalized) * scale + x_min

        else:
            # Fixed-point quantization
            scale = torch.tensor(1.0 / self.num_levels)
            x_quantized = torch.round(x / scale) * scale

        return x_quantized, scale

    def dequantize(
        self,
        x_quantized: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dequantize tensor (identity if already in float format).

        Args:
            x_quantized: Quantized tensor
            scale: Scale factors

        Returns:
            Dequantized tensor
        """
        # In this implementation, we store in float but with reduced precision
        # For actual deployment, this would convert from int to float
        return x_quantized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize and dequantize (for training).

        Args:
            x: Input tensor

        Returns:
            Quantized tensor
        """
        x_quantized, scale = self.quantize(x)
        return x_quantized


class LazyEvaluation(nn.Module):
    """
    Lazy evaluation for message passing.

    Implements Equation (5):
        h_i^(t+1) = h_i^(t) + α·Σ_{j∈N_a(i)} M(i,j)

    where N_a(i) = {j∈N(i) | δ(h_i, h_j, r_ij) = 1} is the set of active neighbors.

    This restricts computation to only nutritionally significant pathways.
    """

    def __init__(self, activation_threshold: float = 0.5):
        """
        Args:
            activation_threshold: Threshold for activating computations
        """
        super().__init__()
        self.activation_threshold = activation_threshold

    def compute_active_edges(
        self,
        importance_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Determine which edges should be active.

        Args:
            importance_scores: Importance scores for edges [num_edges]

        Returns:
            Binary mask for active edges [num_edges]
        """
        # Binary gating based on threshold
        active_edges = (importance_scores > self.activation_threshold).float()

        return active_edges

    def forward(
        self,
        messages: torch.Tensor,
        active_edges: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply lazy evaluation to messages.

        Args:
            messages: Messages [num_edges, hidden_dim]
            active_edges: Binary mask [num_edges]

        Returns:
            Masked messages [num_edges, hidden_dim]
        """
        # Apply mask
        masked_messages = messages * active_edges.unsqueeze(-1)

        return masked_messages


class EarlyStopping(nn.Module):
    """
    Early stopping for message passing iterations.

    Implements Equation (6):
        ||h^(t+1) - h^(t)||_2 < γ

    This reduces the average number of iterations by 42% with negligible
    impact on performance (<0.5% change in metrics).
    """

    def __init__(
        self,
        convergence_threshold: float = 0.01,
        patience: int = 2,
        min_iterations: int = 2,
    ):
        """
        Args:
            convergence_threshold: Threshold γ for convergence
            patience: Number of steps to wait for improvement
            min_iterations: Minimum number of iterations before stopping
        """
        super().__init__()

        self.convergence_threshold = convergence_threshold
        self.patience = patience
        self.min_iterations = min_iterations

        self.reset()

    def reset(self):
        """Reset internal state"""
        self.iteration = 0
        self.patience_counter = 0
        self.prev_h = None

    def check_convergence(
        self,
        h_current: torch.Tensor,
    ) -> bool:
        """
        Check if message passing has converged.

        Args:
            h_current: Current node features [num_nodes, hidden_dim]

        Returns:
            True if converged, False otherwise
        """
        self.iteration += 1

        # Always continue for minimum iterations
        if self.iteration < self.min_iterations:
            self.prev_h = h_current.detach()
            return False

        # Check convergence (Eq. 6)
        if self.prev_h is not None:
            delta = torch.norm(h_current - self.prev_h, p=2)

            if delta < self.convergence_threshold:
                self.patience_counter += 1

                if self.patience_counter >= self.patience:
                    return True
            else:
                self.patience_counter = 0

        self.prev_h = h_current.detach()
        return False

    def forward(self, h_current: torch.Tensor) -> bool:
        """
        Forward pass (checks convergence).

        Args:
            h_current: Current node features

        Returns:
            True if converged
        """
        return self.check_convergence(h_current)


class AdaptiveComputation(nn.Module):
    """
    Adaptive computation based on input complexity.

    Implements Equation (7):
        d(x) = max(d_min, min(d_max, f(x)))

    where d(x) is the dimension used for internal representations and
    f(x) estimates required capacity based on input graph properties.
    """

    def __init__(
        self,
        d_min: int = 32,
        d_max: int = 128,
        default_dim: int = 64,
    ):
        """
        Args:
            d_min: Minimum hidden dimension
            d_max: Maximum hidden dimension
            default_dim: Default hidden dimension
        """
        super().__init__()

        self.d_min = d_min
        self.d_max = d_max
        self.default_dim = default_dim

        # Complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(5, 16),  # 5 graph properties
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def estimate_complexity(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> float:
        """
        Estimate input complexity based on graph properties.

        Args:
            x: Node features [num_nodes, feature_dim]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Complexity score in [0, 1]
        """
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        # Compute graph properties
        avg_degree = num_edges / max(num_nodes, 1)
        density = num_edges / max(num_nodes * (num_nodes - 1), 1)

        # Feature statistics
        feature_std = x.std().item()
        feature_range = (x.max() - x.min()).item()
        feature_sparsity = (x == 0).float().mean().item()

        # Create feature vector
        properties = torch.tensor([
            avg_degree / 10.0,  # Normalize
            density * 100.0,
            feature_std,
            feature_range,
            feature_sparsity,
        ], device=x.device).unsqueeze(0)

        # Estimate complexity
        complexity = self.complexity_estimator(properties).item()

        return complexity

    def compute_dimension(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> int:
        """
        Compute adaptive dimension based on input complexity.

        Implements Equation (7):
            d(x) = max(d_min, min(d_max, f(x)))

        Args:
            x: Node features
            edge_index: Edge indices

        Returns:
            Hidden dimension to use
        """
        # Estimate complexity
        complexity = self.estimate_complexity(x, edge_index)

        # Map complexity to dimension (Eq. 7)
        dim_range = self.d_max - self.d_min
        dim = int(self.d_min + complexity * dim_range)

        # Clamp to valid range
        dim = max(self.d_min, min(self.d_max, dim))

        # Round to nearest multiple of 8 for efficiency
        dim = ((dim + 7) // 8) * 8

        return dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> int:
        """
        Forward pass (computes dimension).

        Args:
            x: Node features
            edge_index: Edge indices

        Returns:
            Adaptive hidden dimension
        """
        return self.compute_dimension(x, edge_index)


class EfficientEmbedding(nn.Module):
    """
    Efficient embedding layer with adaptive dimension and quantization.

    Combines multiple optimization techniques for maximum efficiency.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        adaptive_dim: bool = True,
        quantize: bool = True,
        num_bits: int = 8,
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension (may be adapted)
            adaptive_dim: Whether to use adaptive dimension
            quantize: Whether to quantize features
            num_bits: Number of bits for quantization
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.adaptive_dim = adaptive_dim
        self.quantize_features = quantize

        # Embedding layers for different dimensions
        self.embeddings = nn.ModuleDict({
            '32': nn.Linear(input_dim, 32),
            '64': nn.Linear(input_dim, 64),
            '128': nn.Linear(input_dim, 128),
        })

        # Optimization modules
        if adaptive_dim:
            self.adaptive_comp = AdaptiveComputation(d_min=32, d_max=128, default_dim=hidden_dim)

        if quantize:
            self.quantizer = QuantizedRepresentation(num_bits=num_bits)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with adaptive computation and quantization.

        Args:
            x: Input features [num_nodes, input_dim]
            edge_index: Edge indices (optional, for adaptive dimension)

        Returns:
            Embedded features [num_nodes, hidden_dim]
        """
        # Quantize input if enabled
        if self.quantize_features:
            x = self.quantizer(x)

        # Determine dimension
        if self.adaptive_dim and edge_index is not None:
            dim = self.adaptive_comp(x, edge_index)
        else:
            dim = self.hidden_dim

        # Select appropriate embedding layer
        dim_str = str(dim)
        if dim_str in self.embeddings:
            h = self.embeddings[dim_str](x)
        else:
            # Fallback to closest dimension
            closest_dim = min(self.embeddings.keys(), key=lambda k: abs(int(k) - dim))
            h = self.embeddings[closest_dim](x)

            # Pad or truncate if necessary
            if h.size(-1) < dim:
                padding = torch.zeros(
                    *h.size()[:-1], dim - h.size(-1),
                    device=h.device
                )
                h = torch.cat([h, padding], dim=-1)
            elif h.size(-1) > dim:
                h = h[..., :dim]

        return h


class MemoryEfficientAttention(nn.Module):
    """
    Memory-efficient attention using chunking and quantization.

    Reduces memory footprint for large graphs while maintaining quality.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        chunk_size: int = 512,
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            chunk_size: Chunk size for processing
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.head_dim = hidden_dim // num_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Memory-efficient attention computation.

        Args:
            query: Query tensor [num_nodes, hidden_dim]
            key: Key tensor [num_nodes, hidden_dim]
            value: Value tensor [num_nodes, hidden_dim]
            edge_mask: Optional edge mask

        Returns:
            Attention output [num_nodes, hidden_dim]
        """
        num_nodes = query.size(0)

        # Reshape for multi-head attention
        q = query.view(num_nodes, self.num_heads, self.head_dim)
        k = key.view(num_nodes, self.num_heads, self.head_dim)
        v = value.view(num_nodes, self.num_heads, self.head_dim)

        # Process in chunks to reduce memory
        output_chunks = []

        for i in range(0, num_nodes, self.chunk_size):
            end_i = min(i + self.chunk_size, num_nodes)
            q_chunk = q[i:end_i]

            # Compute attention for this chunk
            scores = torch.einsum('ihd,jhd->ijh', q_chunk, k)
            scores = scores / (self.head_dim ** 0.5)

            if edge_mask is not None:
                scores = scores.masked_fill(~edge_mask[i:end_i], float('-inf'))

            attn = F.softmax(scores, dim=1)

            # Apply attention to values
            output_chunk = torch.einsum('ijh,jhd->ihd', attn, v)
            output_chunks.append(output_chunk)

        # Concatenate chunks
        output = torch.cat(output_chunks, dim=0)

        # Reshape back
        output = output.reshape(num_nodes, self.hidden_dim)

        return output
