"""
Transparent Prediction Module

Implements the transparent prediction mechanism as described in Section 3.4.

This module provides interpretability through:
1. Subgraph extraction (Eq. 8): G_s = {v_i, e_ij | ∂y/∂h_i > ε ∨ ∂y/∂r_ij > ε}
2. Path analysis (Eq. 9): P = {p_1, p_2, ..., p_m}
3. Explanation generation (Eq. 10): E(p_k) = T(...)

This enables human-understandable explanations of nutritional recommendations.
"""

import torch
import torch.nn as nn
import networkx as nx
from typing import List, Dict, Tuple, Optional
import numpy as np


class GradientBasedSubgraphExtractor(nn.Module):
    """
    Extracts relevant subgraphs using gradient-based importance.

    Implements Equation (8):
        G_s = {v_i, e_ij | ∂y/∂h_i > ε ∨ ∂y/∂r_ij > ε}
    """

    def __init__(self, significance_threshold: float = 0.1):
        """
        Args:
            significance_threshold: Threshold ε for gradient significance
        """
        super().__init__()
        self.significance_threshold = significance_threshold

    def extract_subgraph(
        self,
        prediction: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract minimal subgraph contributing to prediction.

        Args:
            prediction: Model prediction (scalar or vector)
            node_features: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_features: Edge features [num_edges, edge_dim]

        Returns:
            Tuple of (significant_node_mask, significant_edge_mask)
        """
        # Ensure gradients are enabled
        node_features.requires_grad_(True)
        edge_features.requires_grad_(True)

        # Compute gradients
        if prediction.dim() == 0:
            grad_outputs = torch.ones_like(prediction)
        else:
            grad_outputs = torch.ones_like(prediction)

        # Gradient with respect to node features
        grad_nodes = torch.autograd.grad(
            outputs=prediction,
            inputs=node_features,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=True,
        )[0]

        # Gradient with respect to edge features
        grad_edges = torch.autograd.grad(
            outputs=prediction,
            inputs=edge_features,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False,
        )[0]

        # Compute importance scores (L2 norm of gradients)
        node_importance = torch.norm(grad_nodes, p=2, dim=-1)  # [num_nodes]
        edge_importance = torch.norm(grad_edges, p=2, dim=-1)  # [num_edges]

        # Normalize
        node_importance = node_importance / (node_importance.max() + 1e-8)
        edge_importance = edge_importance / (edge_importance.max() + 1e-8)

        # Create masks based on threshold (Eq. 8)
        significant_nodes = node_importance > self.significance_threshold
        significant_edges = edge_importance > self.significance_threshold

        return significant_nodes, significant_edges


class PathAnalyzer(nn.Module):
    """
    Identifies critical paths connecting inputs to outputs.

    Implements Equation (9):
        P = {p_1, p_2, ..., p_m} where p_k = (v_k1, v_k2, ..., v_kt)
    """

    def __init__(self, max_path_length: int = 5, max_paths: int = 10):
        """
        Args:
            max_path_length: Maximum length of paths to consider
            max_paths: Maximum number of paths to return
        """
        super().__init__()
        self.max_path_length = max_path_length
        self.max_paths = max_paths

    def find_critical_paths(
        self,
        source_nodes: List[int],
        target_nodes: List[int],
        edge_index: torch.Tensor,
        node_importance: torch.Tensor,
        edge_importance: torch.Tensor,
    ) -> List[List[int]]:
        """
        Find critical paths from source to target nodes.

        Args:
            source_nodes: List of source node IDs (input nutrients)
            target_nodes: List of target node IDs (output predictions)
            edge_index: Edge indices [2, num_edges]
            node_importance: Importance score for each node [num_nodes]
            edge_importance: Importance score for each edge [num_edges]

        Returns:
            List of paths, where each path is a list of node IDs
        """
        # Build NetworkX graph
        G = self._build_networkx_graph(
            edge_index, node_importance, edge_importance
        )

        # Find paths
        all_paths = []
        for source in source_nodes:
            for target in target_nodes:
                try:
                    # Find all simple paths up to max_path_length
                    paths = nx.all_simple_paths(
                        G, source, target,
                        cutoff=self.max_path_length
                    )

                    for path in paths:
                        path_score = self._compute_path_score(
                            path, node_importance, edge_index, edge_importance
                        )
                        all_paths.append((path, path_score))

                except nx.NetworkXNoPath:
                    continue

        # Sort by score and return top paths
        all_paths.sort(key=lambda x: x[1], reverse=True)
        return [path for path, score in all_paths[:self.max_paths]]

    def _build_networkx_graph(
        self,
        edge_index: torch.Tensor,
        node_importance: torch.Tensor,
        edge_importance: torch.Tensor,
    ) -> nx.DiGraph:
        """Build NetworkX directed graph from edge_index"""
        G = nx.DiGraph()

        # Add nodes with importance
        num_nodes = node_importance.size(0)
        for i in range(num_nodes):
            G.add_node(i, importance=node_importance[i].item())

        # Add edges with importance
        edge_index_np = edge_index.cpu().numpy()
        edge_importance_np = edge_importance.cpu().numpy()

        for i in range(edge_index.size(1)):
            source = int(edge_index_np[0, i])
            target = int(edge_index_np[1, i])
            importance = float(edge_importance_np[i])

            G.add_edge(source, target, importance=importance)

        return G

    def _compute_path_score(
        self,
        path: List[int],
        node_importance: torch.Tensor,
        edge_index: torch.Tensor,
        edge_importance: torch.Tensor,
    ) -> float:
        """Compute importance score for a path"""
        score = 0.0

        # Add node importances
        for node in path:
            score += node_importance[node].item()

        # Add edge importances
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            # Find edge index
            edge_mask = (edge_index[0] == source) & (edge_index[1] == target)
            if edge_mask.any():
                edge_idx = edge_mask.nonzero(as_tuple=True)[0][0]
                score += edge_importance[edge_idx].item()

        # Normalize by path length
        score = score / len(path)

        return score


class ExplanationGenerator:
    """
    Generates human-readable explanations from paths.

    Implements Equation (10):
        E(p_k) = T(v_k1, v_k2, ..., v_kt, r_k1k2, ..., r_k(t-1)kt)
    """

    def __init__(self, node_names: Optional[List[str]] = None):
        """
        Args:
            node_names: Human-readable names for nodes
        """
        self.node_names = node_names
        self.templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize explanation templates"""
        return {
            "synergistic": "{source} enhances the absorption/utilization of {target}",
            "antagonistic": "{source} inhibits the absorption/utilization of {target}",
            "threshold_dependent": "{source} affects {target} in a dose-dependent manner",
            "neutral": "{source} is metabolically connected to {target}",
            "path": "{nutrient} → {intermediates} → {outcome}",
        }

    def generate_explanation(
        self,
        paths: List[List[int]],
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        prediction: float,
        prediction_type: str = "deficiency_risk",
    ) -> Dict[str, any]:
        """
        Generate explanation from critical paths.

        Args:
            paths: List of critical paths
            edge_index: Edge indices
            edge_features: Edge features
            prediction: Model prediction
            prediction_type: Type of prediction

        Returns:
            Dictionary containing explanation components
        """
        explanation = {
            "prediction": prediction,
            "prediction_type": prediction_type,
            "critical_paths": [],
            "key_nutrients": [],
            "interactions": [],
            "summary": "",
        }

        # Analyze each path
        for path in paths:
            path_explanation = self._explain_path(
                path, edge_index, edge_features
            )
            explanation["critical_paths"].append(path_explanation)

        # Extract key nutrients (nodes appearing in multiple paths)
        node_counts = {}
        for path in paths:
            for node in path:
                node_counts[node] = node_counts.get(node, 0) + 1

        # Sort by frequency
        key_nutrients = sorted(
            node_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        explanation["key_nutrients"] = [
            {
                "node_id": node,
                "name": self._get_node_name(node),
                "frequency": count,
            }
            for node, count in key_nutrients
        ]

        # Generate summary
        explanation["summary"] = self._generate_summary(
            prediction, prediction_type, key_nutrients, paths
        )

        return explanation

    def _explain_path(
        self,
        path: List[int],
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> Dict[str, any]:
        """Generate explanation for a single path"""
        path_explanation = {
            "nodes": [self._get_node_name(node) for node in path],
            "interactions": [],
        }

        # Explain each edge in the path
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]

            # Find edge
            edge_mask = (edge_index[0] == source) & (edge_index[1] == target)
            if edge_mask.any():
                edge_idx = edge_mask.nonzero(as_tuple=True)[0][0]
                edge_feature = edge_features[edge_idx]

                # Determine interaction type from edge features
                interaction_type = self._determine_interaction_type(edge_feature)

                # Generate interaction explanation
                interaction_text = self.templates[interaction_type].format(
                    source=self._get_node_name(source),
                    target=self._get_node_name(target),
                )

                path_explanation["interactions"].append({
                    "source": self._get_node_name(source),
                    "target": self._get_node_name(target),
                    "type": interaction_type,
                    "explanation": interaction_text,
                })

        return path_explanation

    def _get_node_name(self, node_id: int) -> str:
        """Get human-readable name for a node"""
        if self.node_names and node_id < len(self.node_names):
            return self.node_names[node_id]
        return f"Node_{node_id}"

    def _determine_interaction_type(self, edge_feature: torch.Tensor) -> str:
        """Determine interaction type from edge features"""
        # Assuming one-hot encoding in first 4 positions
        interaction_types = ["synergistic", "antagonistic", "threshold_dependent", "neutral"]

        if edge_feature.size(0) >= 4:
            type_idx = edge_feature[:4].argmax().item()
            return interaction_types[type_idx]

        return "neutral"

    def _generate_summary(
        self,
        prediction: float,
        prediction_type: str,
        key_nutrients: List[Tuple[int, int]],
        paths: List[List[int]],
    ) -> str:
        """Generate natural language summary"""
        summary_parts = []

        # Prediction statement
        if prediction_type == "deficiency_risk":
            risk_level = "high" if prediction > 0.7 else "moderate" if prediction > 0.4 else "low"
            summary_parts.append(
                f"The model predicts a {risk_level} deficiency risk (score: {prediction:.2f})."
            )
        elif prediction_type == "nutrient_level":
            summary_parts.append(
                f"The predicted nutrient level is {prediction:.2f}."
            )

        # Key factors
        if key_nutrients:
            nutrient_names = [self._get_node_name(node) for node, _ in key_nutrients[:3]]
            summary_parts.append(
                f"Key contributing factors include: {', '.join(nutrient_names)}."
            )

        # Pathway information
        if paths:
            summary_parts.append(
                f"This prediction is supported by {len(paths)} metabolic pathways."
            )

        return " ".join(summary_parts)


class TransparentPredictionModule(nn.Module):
    """
    Complete transparent prediction module.

    Combines subgraph extraction, path analysis, and explanation generation
    to provide interpretable predictions.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        node_names: Optional[List[str]] = None,
        significance_threshold: float = 0.1,
        max_path_length: int = 5,
        max_paths: int = 10,
    ):
        """
        Args:
            hidden_dim: Dimension of node hidden states
            output_dim: Dimension of output (number of prediction tasks)
            node_names: Human-readable names for nodes
            significance_threshold: Threshold for subgraph extraction
            max_path_length: Maximum path length for analysis
            max_paths: Maximum number of paths to extract
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        # Interpretability components
        self.subgraph_extractor = GradientBasedSubgraphExtractor(
            significance_threshold=significance_threshold
        )
        self.path_analyzer = PathAnalyzer(
            max_path_length=max_path_length,
            max_paths=max_paths,
        )
        self.explanation_generator = ExplanationGenerator(node_names=node_names)

    def forward(
        self,
        node_features: torch.Tensor,
        graph_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for prediction.

        Args:
            node_features: Node features [num_nodes, hidden_dim]
            graph_embedding: Optional graph-level embedding [batch_size, hidden_dim]

        Returns:
            Predictions [num_nodes, output_dim] or [batch_size, output_dim]
        """
        if graph_embedding is not None:
            # Graph-level prediction
            return self.predictor(graph_embedding)
        else:
            # Node-level predictions
            return self.predictor(node_features)

    def explain(
        self,
        prediction: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        source_nodes: Optional[List[int]] = None,
        target_nodes: Optional[List[int]] = None,
    ) -> Dict[str, any]:
        """
        Generate explanation for a prediction.

        Args:
            prediction: Model prediction
            node_features: Node features (with gradients)
            edge_index: Edge indices
            edge_features: Edge features (with gradients)
            source_nodes: Source nodes for path analysis
            target_nodes: Target nodes for path analysis

        Returns:
            Dictionary containing explanation
        """
        # Extract significant subgraph (Eq. 8)
        significant_nodes, significant_edges = self.subgraph_extractor.extract_subgraph(
            prediction, node_features, edge_index, edge_features
        )

        # Compute importance scores
        node_importance = torch.zeros(node_features.size(0))
        edge_importance = torch.zeros(edge_index.size(1))

        # Use gradient magnitude as importance
        if node_features.grad is not None:
            node_importance = torch.norm(node_features.grad, p=2, dim=-1)
        if edge_features.grad is not None:
            edge_importance = torch.norm(edge_features.grad, p=2, dim=-1)

        # Normalize
        node_importance = node_importance / (node_importance.max() + 1e-8)
        edge_importance = edge_importance / (edge_importance.max() + 1e-8)

        # Find critical paths (Eq. 9)
        if source_nodes is None:
            source_nodes = significant_nodes.nonzero(as_tuple=True)[0][:5].tolist()
        if target_nodes is None:
            # Use nodes with highest importance as targets
            _, target_nodes = torch.topk(node_importance, min(3, node_importance.size(0)))
            target_nodes = target_nodes.tolist()

        paths = self.path_analyzer.find_critical_paths(
            source_nodes, target_nodes, edge_index, node_importance, edge_importance
        )

        # Generate explanation (Eq. 10)
        explanation = self.explanation_generator.generate_explanation(
            paths, edge_index, edge_features,
            prediction.item() if prediction.dim() == 0 else prediction.mean().item()
        )

        # Add subgraph information
        explanation["significant_nodes"] = significant_nodes.nonzero(as_tuple=True)[0].tolist()
        explanation["significant_edges"] = significant_edges.nonzero(as_tuple=True)[0].tolist()
        explanation["node_importance"] = node_importance.tolist()
        explanation["edge_importance"] = edge_importance.tolist()

        return explanation
