"""
Hierarchical Nutrient Graph Representation

Implements the multi-level graph structure as described in Section 3.1:
G = (V, E, X, R)

This module provides:
- Hierarchical graph construction with three levels:
  * Micronutrient level (vitamins, minerals, amino acids)
  * Food component level (proteins, carbohydrates, fats)
  * Health outcome level (biomarkers, clinical indicators)
- Automatic graph construction from nutritional databases
- Edge feature computation based on interaction types
"""

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum


class NodeType(Enum):
    """Types of nodes in the hierarchical graph"""
    MICRONUTRIENT = "micronutrient"
    FOOD_COMPONENT = "food_component"
    HEALTH_OUTCOME = "health_outcome"


class InteractionType(Enum):
    """Types of nutrient interactions"""
    SYNERGISTIC = "synergistic"  # e.g., vitamin C enhancing iron absorption
    ANTAGONISTIC = "antagonistic"  # e.g., calcium inhibiting zinc absorption
    THRESHOLD_DEPENDENT = "threshold_dependent"  # e.g., toxicity at high doses
    NEUTRAL = "neutral"


@dataclass
class NutrientNode:
    """Represents a node in the nutrient graph"""
    id: int
    name: str
    node_type: NodeType
    features: np.ndarray
    level: int  # 0: micronutrient, 1: food component, 2: health outcome


@dataclass
class NutrientEdge:
    """Represents an edge in the nutrient graph"""
    source: int
    target: int
    interaction_type: InteractionType
    strength: float  # Interaction strength [0, 1]
    features: np.ndarray


class HierarchicalNutrientGraph:
    """
    Hierarchical graph representation for nutrient interactions.

    Implements Equation (1) from the paper:
    G = (V, E, X, R)

    where:
    - V: set of nodes (nutrients, biomarkers, health indicators)
    - E: set of edges (established interactions)
    - X: node features
    - R: edge features
    """

    def __init__(
        self,
        num_micronutrients: int = 50,
        num_food_components: int = 30,
        num_health_outcomes: int = 20,
        feature_dim: int = 64,
    ):
        """
        Initialize hierarchical nutrient graph.

        Args:
            num_micronutrients: Number of micronutrient nodes
            num_food_components: Number of food component nodes
            num_health_outcomes: Number of health outcome nodes
            feature_dim: Dimension of node features
        """
        self.num_micronutrients = num_micronutrients
        self.num_food_components = num_food_components
        self.num_health_outcomes = num_health_outcomes
        self.feature_dim = feature_dim

        self.total_nodes = num_micronutrients + num_food_components + num_health_outcomes

        # Initialize graph structure
        self.nodes: List[NutrientNode] = []
        self.edges: List[NutrientEdge] = []
        self.adjacency_list: Dict[int, List[int]] = {}

        # NetworkX graph for analysis
        self.nx_graph = nx.DiGraph()

        # Node and edge feature matrices
        self.node_features = None
        self.edge_index = None
        self.edge_features = None

        # Hierarchical structure
        self.level_ranges = {
            0: (0, num_micronutrients),
            1: (num_micronutrients, num_micronutrients + num_food_components),
            2: (num_micronutrients + num_food_components, self.total_nodes),
        }

    def build_graph(
        self,
        nutrient_database: Optional[Dict] = None,
        interaction_database: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct the hierarchical nutrient graph from domain knowledge.

        Implements Algorithm 1 from the paper.

        Args:
            nutrient_database: Database of nutrients and their properties
            interaction_database: Database of known nutrient interactions

        Returns:
            Tuple of (node_features, edge_index, edge_features)
        """
        # Initialize nodes at each level
        self._initialize_nodes(nutrient_database)

        # Build edges based on known interactions
        self._build_edges(interaction_database)

        # Convert to PyTorch tensors
        return self._to_tensors()

    def _initialize_nodes(self, nutrient_database: Optional[Dict] = None):
        """Initialize nodes at all hierarchical levels"""
        node_id = 0

        # Level 0: Micronutrients (vitamins, minerals, amino acids)
        micronutrient_names = self._get_micronutrient_names(nutrient_database)
        for i, name in enumerate(micronutrient_names):
            features = self._compute_node_features(name, NodeType.MICRONUTRIENT, nutrient_database)
            node = NutrientNode(
                id=node_id,
                name=name,
                node_type=NodeType.MICRONUTRIENT,
                features=features,
                level=0,
            )
            self.nodes.append(node)
            self.nx_graph.add_node(node_id, **vars(node))
            node_id += 1

        # Level 1: Food components (proteins, carbohydrates, fats)
        food_component_names = self._get_food_component_names(nutrient_database)
        for i, name in enumerate(food_component_names):
            features = self._compute_node_features(name, NodeType.FOOD_COMPONENT, nutrient_database)
            node = NutrientNode(
                id=node_id,
                name=name,
                node_type=NodeType.FOOD_COMPONENT,
                features=features,
                level=1,
            )
            self.nodes.append(node)
            self.nx_graph.add_node(node_id, **vars(node))
            node_id += 1

        # Level 2: Health outcomes (biomarkers, clinical indicators)
        health_outcome_names = self._get_health_outcome_names(nutrient_database)
        for i, name in enumerate(health_outcome_names):
            features = self._compute_node_features(name, NodeType.HEALTH_OUTCOME, nutrient_database)
            node = NutrientNode(
                id=node_id,
                name=name,
                node_type=NodeType.HEALTH_OUTCOME,
                features=features,
                level=2,
            )
            self.nodes.append(node)
            self.nx_graph.add_node(node_id, **vars(node))
            node_id += 1

    def _build_edges(self, interaction_database: Optional[Dict] = None):
        """Build edges based on known nutrient interactions"""
        # Within-level edges (same hierarchical level)
        self._build_within_level_edges(interaction_database)

        # Between-level edges (different hierarchical levels)
        self._build_between_level_edges(interaction_database)

    def _build_within_level_edges(self, interaction_database: Optional[Dict] = None):
        """Build edges within the same hierarchical level"""
        # Define known interactions (example data - should come from database)
        known_interactions = self._get_known_interactions(interaction_database)

        for source_id, target_id, interaction_type, strength in known_interactions:
            if self.nodes[source_id].level == self.nodes[target_id].level:
                edge_features = self._compute_edge_features(source_id, target_id, interaction_type, strength)
                edge = NutrientEdge(
                    source=source_id,
                    target=target_id,
                    interaction_type=interaction_type,
                    strength=strength,
                    features=edge_features,
                )
                self.edges.append(edge)
                self.nx_graph.add_edge(source_id, target_id, **vars(edge))

                # Add to adjacency list
                if source_id not in self.adjacency_list:
                    self.adjacency_list[source_id] = []
                self.adjacency_list[source_id].append(target_id)

    def _build_between_level_edges(self, interaction_database: Optional[Dict] = None):
        """Build edges between different hierarchical levels"""
        # Connect micronutrients to food components
        for micro_id in range(*self.level_ranges[0]):
            # Each micronutrient connects to relevant food components
            relevant_components = self._find_relevant_components(micro_id)
            for comp_id in relevant_components:
                edge_features = self._compute_edge_features(
                    micro_id, comp_id, InteractionType.NEUTRAL, 0.5
                )
                edge = NutrientEdge(
                    source=micro_id,
                    target=comp_id,
                    interaction_type=InteractionType.NEUTRAL,
                    strength=0.5,
                    features=edge_features,
                )
                self.edges.append(edge)
                self.nx_graph.add_edge(micro_id, comp_id, **vars(edge))

        # Connect food components to health outcomes
        for comp_id in range(*self.level_ranges[1]):
            relevant_outcomes = self._find_relevant_outcomes(comp_id)
            for outcome_id in relevant_outcomes:
                edge_features = self._compute_edge_features(
                    comp_id, outcome_id, InteractionType.NEUTRAL, 0.5
                )
                edge = NutrientEdge(
                    source=comp_id,
                    target=outcome_id,
                    interaction_type=InteractionType.NEUTRAL,
                    strength=0.5,
                    features=edge_features,
                )
                self.edges.append(edge)
                self.nx_graph.add_edge(comp_id, outcome_id, **vars(edge))

    def _to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert graph to PyTorch tensors"""
        # Node features: [num_nodes, feature_dim]
        node_features = torch.stack([
            torch.from_numpy(node.features).float() for node in self.nodes
        ])
        self.node_features = node_features

        # Edge index: [2, num_edges]
        edge_index = torch.tensor([
            [edge.source for edge in self.edges],
            [edge.target for edge in self.edges]
        ], dtype=torch.long)
        self.edge_index = edge_index

        # Edge features: [num_edges, edge_feature_dim]
        edge_features = torch.stack([
            torch.from_numpy(edge.features).float() for edge in self.edges
        ])
        self.edge_features = edge_features

        return node_features, edge_index, edge_features

    def _get_micronutrient_names(self, database: Optional[Dict] = None) -> List[str]:
        """Get list of micronutrient names"""
        if database and 'micronutrients' in database:
            return database['micronutrients'][:self.num_micronutrients]

        # Default micronutrients (vitamins, minerals, amino acids)
        micronutrients = [
            # Vitamins
            "Vitamin_A", "Vitamin_B1", "Vitamin_B2", "Vitamin_B3", "Vitamin_B5",
            "Vitamin_B6", "Vitamin_B7", "Vitamin_B9", "Vitamin_B12", "Vitamin_C",
            "Vitamin_D", "Vitamin_E", "Vitamin_K",
            # Minerals
            "Calcium", "Iron", "Magnesium", "Phosphorus", "Potassium", "Sodium",
            "Zinc", "Copper", "Manganese", "Selenium", "Iodine", "Chromium",
            # Essential amino acids
            "Histidine", "Isoleucine", "Leucine", "Lysine", "Methionine",
            "Phenylalanine", "Threonine", "Tryptophan", "Valine",
        ]
        return micronutrients[:self.num_micronutrients]

    def _get_food_component_names(self, database: Optional[Dict] = None) -> List[str]:
        """Get list of food component names"""
        if database and 'food_components' in database:
            return database['food_components'][:self.num_food_components]

        # Default food components
        components = [
            "Protein", "Carbohydrate", "Fat", "Fiber", "Sugar",
            "Saturated_Fat", "Monounsaturated_Fat", "Polyunsaturated_Fat",
            "Omega_3", "Omega_6", "Cholesterol", "Starch",
        ]
        # Pad to required number
        while len(components) < self.num_food_components:
            components.append(f"Component_{len(components)}")
        return components[:self.num_food_components]

    def _get_health_outcome_names(self, database: Optional[Dict] = None) -> List[str]:
        """Get list of health outcome names"""
        if database and 'health_outcomes' in database:
            return database['health_outcomes'][:self.num_health_outcomes]

        # Default health outcomes (biomarkers and clinical indicators)
        outcomes = [
            "Blood_Glucose", "Cholesterol_Total", "HDL", "LDL", "Triglycerides",
            "Blood_Pressure_Systolic", "Blood_Pressure_Diastolic",
            "BMI", "Body_Fat_Percentage", "Bone_Density",
            "Hemoglobin", "White_Blood_Cell_Count", "Red_Blood_Cell_Count",
            "Inflammation_Marker", "Oxidative_Stress",
        ]
        # Pad to required number
        while len(outcomes) < self.num_health_outcomes:
            outcomes.append(f"Outcome_{len(outcomes)}")
        return outcomes[:self.num_health_outcomes]

    def _compute_node_features(
        self, name: str, node_type: NodeType, database: Optional[Dict] = None
    ) -> np.ndarray:
        """Compute feature vector for a node"""
        if database and name in database.get('features', {}):
            features = database['features'][name]
            if len(features) < self.feature_dim:
                # Pad features
                features = np.pad(features, (0, self.feature_dim - len(features)))
            return features[:self.feature_dim]

        # Initialize with random features (should be replaced with real data)
        np.random.seed(hash(name) % 2**32)
        features = np.random.randn(self.feature_dim).astype(np.float32)
        return features

    def _compute_edge_features(
        self, source: int, target: int, interaction_type: InteractionType, strength: float
    ) -> np.ndarray:
        """Compute edge feature vector"""
        # Edge features encode interaction type and strength
        edge_dim = 16  # Edge feature dimension
        features = np.zeros(edge_dim, dtype=np.float32)

        # Encode interaction type (one-hot)
        type_encoding = {
            InteractionType.SYNERGISTIC: 0,
            InteractionType.ANTAGONISTIC: 1,
            InteractionType.THRESHOLD_DEPENDENT: 2,
            InteractionType.NEUTRAL: 3,
        }
        features[type_encoding[interaction_type]] = 1.0

        # Add strength
        features[4] = strength

        # Add source and target node type information
        features[5] = self.nodes[source].level / 2.0
        features[6] = self.nodes[target].level / 2.0

        return features

    def _get_known_interactions(self, database: Optional[Dict] = None) -> List[Tuple]:
        """Get known nutrient interactions from database"""
        if database and 'interactions' in database:
            return database['interactions']

        # Example interactions (should come from nutritional science database)
        interactions = []

        # Vitamin C enhances iron absorption (synergistic)
        if self.num_micronutrients >= 10:
            vitamin_c_id = 9  # Vitamin C
            iron_id = 14  # Iron
            interactions.append((vitamin_c_id, iron_id, InteractionType.SYNERGISTIC, 0.8))

        # Calcium inhibits zinc absorption (antagonistic)
        if self.num_micronutrients >= 20:
            calcium_id = 13  # Calcium
            zinc_id = 19  # Zinc
            interactions.append((calcium_id, zinc_id, InteractionType.ANTAGONISTIC, 0.7))

        # More interactions can be added based on nutritional science
        return interactions

    def _find_relevant_components(self, micronutrient_id: int) -> List[int]:
        """Find food components relevant to a micronutrient"""
        # Simplified: connect to a subset of food components
        num_connections = min(5, self.num_food_components)
        start_idx = self.level_ranges[1][0]

        # Use hash for deterministic selection
        np.random.seed(micronutrient_id)
        indices = np.random.choice(self.num_food_components, num_connections, replace=False)
        return [start_idx + idx for idx in indices]

    def _find_relevant_outcomes(self, component_id: int) -> List[int]:
        """Find health outcomes relevant to a food component"""
        # Simplified: connect to a subset of health outcomes
        num_connections = min(3, self.num_health_outcomes)
        start_idx = self.level_ranges[2][0]

        # Use hash for deterministic selection
        np.random.seed(component_id)
        indices = np.random.choice(self.num_health_outcomes, num_connections, replace=False)
        return [start_idx + idx for idx in indices]

    def get_subgraph(self, node_ids: List[int]) -> 'HierarchicalNutrientGraph':
        """Extract subgraph containing specified nodes"""
        # Create new graph with subset of nodes
        subgraph = HierarchicalNutrientGraph(
            num_micronutrients=0,
            num_food_components=0,
            num_health_outcomes=0,
            feature_dim=self.feature_dim,
        )

        # Add nodes
        node_id_map = {}
        for new_id, old_id in enumerate(node_ids):
            if old_id < len(self.nodes):
                node = self.nodes[old_id]
                subgraph.nodes.append(node)
                node_id_map[old_id] = new_id

        # Add edges that connect nodes in subgraph
        for edge in self.edges:
            if edge.source in node_id_map and edge.target in node_id_map:
                new_edge = NutrientEdge(
                    source=node_id_map[edge.source],
                    target=node_id_map[edge.target],
                    interaction_type=edge.interaction_type,
                    strength=edge.strength,
                    features=edge.features.copy(),
                )
                subgraph.edges.append(new_edge)

        subgraph._to_tensors()
        return subgraph


def build_nutrient_graph(
    num_micronutrients: int = 50,
    num_food_components: int = 30,
    num_health_outcomes: int = 20,
    feature_dim: int = 64,
    nutrient_database: Optional[Dict] = None,
    interaction_database: Optional[Dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convenience function to build a nutrient graph.

    Args:
        num_micronutrients: Number of micronutrient nodes
        num_food_components: Number of food component nodes
        num_health_outcomes: Number of health outcome nodes
        feature_dim: Dimension of node features
        nutrient_database: Optional database of nutrients
        interaction_database: Optional database of interactions

    Returns:
        Tuple of (node_features, edge_index, edge_features)
    """
    graph = HierarchicalNutrientGraph(
        num_micronutrients=num_micronutrients,
        num_food_components=num_food_components,
        num_health_outcomes=num_health_outcomes,
        feature_dim=feature_dim,
    )
    return graph.build_graph(nutrient_database, interaction_database)
