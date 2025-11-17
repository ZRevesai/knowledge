"""
Nutrient Interaction Network for KGNN.

This module implements the nutrient-nutrient interaction network
based on biochemical pathways and clinical research.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set
import json


class NutrientInteractionNetwork:
    """
    Manages nutrient-nutrient interactions based on biochemical knowledge.

    This class maintains known synergistic and antagonistic relationships
    between nutrients that affect absorption, metabolism, and function.
    """

    def __init__(self):
        """Initialize the nutrient interaction network."""
        self.graph = nx.Graph()
        self.interactions = {}
        self._build_interaction_network()

    def _build_interaction_network(self):
        """
        Build the nutrient interaction network based on biochemical research.

        Interaction types:
        - synergistic: Nutrients that enhance each other's effects
        - antagonistic: Nutrients that interfere with each other
        - metabolic: Nutrients involved in same metabolic pathways
        - absorption: Nutrients affecting each other's absorption
        """

        # Define known nutrient interactions from literature
        interactions = [
            # Synergistic interactions
            ('Vitamin_D', 'Calcium', 'synergistic', 0.9),  # Vit D enhances Ca absorption
            ('Vitamin_C', 'Iron', 'synergistic', 0.85),    # Vit C enhances Fe absorption
            ('Vitamin_B12', 'Folate', 'synergistic', 0.8), # B12 and folate work together
            ('Vitamin_B6', 'Magnesium', 'synergistic', 0.75),
            ('Vitamin_E', 'Selenium', 'synergistic', 0.7),
            ('Vitamin_K', 'Vitamin_D', 'synergistic', 0.75),
            ('Vitamin_A', 'Zinc', 'synergistic', 0.8),
            ('Calcium', 'Magnesium', 'synergistic', 0.7),

            # Antagonistic interactions
            ('Calcium', 'Iron', 'antagonistic', -0.7),     # Ca inhibits Fe absorption
            ('Calcium', 'Zinc', 'antagonistic', -0.65),    # Ca inhibits Zn absorption
            ('Iron', 'Zinc', 'antagonistic', -0.6),        # Fe and Zn compete
            ('Zinc', 'Copper', 'antagonistic', -0.7),      # Zn inhibits Cu

            # Metabolic pathway relationships
            ('Vitamin_B6', 'Vitamin_B12', 'metabolic', 0.7),  # Both in homocysteine metabolism
            ('Vitamin_B12', 'Folate', 'metabolic', 0.8),      # Folate cycle
            ('Vitamin_B6', 'Folate', 'metabolic', 0.65),
            ('Iron', 'Vitamin_A', 'metabolic', 0.6),          # Both in red blood cell formation
            ('Selenium', 'Vitamin_E', 'metabolic', 0.7),      # Antioxidant pathways

            # Absorption interactions
            ('Vitamin_D', 'Magnesium', 'absorption', 0.65),
            ('Vitamin_A', 'Vitamin_E', 'absorption', 0.6),
        ]

        # Build the network
        for nutrient1, nutrient2, interaction_type, strength in interactions:
            self.add_interaction(nutrient1, nutrient2, interaction_type, strength)

    def add_interaction(self, nutrient1: str, nutrient2: str,
                       interaction_type: str, strength: float):
        """
        Add an interaction between two nutrients.

        Args:
            nutrient1: First nutrient name
            nutrient2: Second nutrient name
            interaction_type: Type of interaction (synergistic, antagonistic, metabolic, absorption)
            strength: Strength of interaction (-1 to 1)
        """
        self.graph.add_edge(nutrient1, nutrient2,
                           type=interaction_type,
                           strength=strength)

        key = tuple(sorted([nutrient1, nutrient2]))
        self.interactions[key] = {
            'type': interaction_type,
            'strength': strength
        }

    def get_interaction(self, nutrient1: str, nutrient2: str) -> Dict:
        """
        Get the interaction details between two nutrients.

        Args:
            nutrient1: First nutrient name
            nutrient2: Second nutrient name

        Returns:
            Dictionary with interaction type and strength, or None if no interaction
        """
        key = tuple(sorted([nutrient1, nutrient2]))
        return self.interactions.get(key, None)

    def get_interaction_strength(self, nutrient1: str, nutrient2: str) -> float:
        """
        Get the interaction strength between two nutrients.

        Args:
            nutrient1: First nutrient name
            nutrient2: Second nutrient name

        Returns:
            Interaction strength (-1 to 1), 0 if no known interaction
        """
        interaction = self.get_interaction(nutrient1, nutrient2)
        return interaction['strength'] if interaction else 0.0

    def is_synergistic(self, nutrient1: str, nutrient2: str) -> bool:
        """
        Check if two nutrients have a synergistic relationship.

        Args:
            nutrient1: First nutrient name
            nutrient2: Second nutrient name

        Returns:
            True if synergistic relationship exists
        """
        interaction = self.get_interaction(nutrient1, nutrient2)
        return interaction is not None and interaction['type'] == 'synergistic'

    def is_antagonistic(self, nutrient1: str, nutrient2: str) -> bool:
        """
        Check if two nutrients have an antagonistic relationship.

        Args:
            nutrient1: First nutrient name
            nutrient2: Second nutrient name

        Returns:
            True if antagonistic relationship exists
        """
        interaction = self.get_interaction(nutrient1, nutrient2)
        return interaction is not None and interaction['type'] == 'antagonistic'

    def get_related_nutrients(self, nutrient: str, interaction_type: str = None) -> List[str]:
        """
        Get all nutrients that interact with the given nutrient.

        Args:
            nutrient: Nutrient name
            interaction_type: Optional filter by interaction type

        Returns:
            List of related nutrient names
        """
        if nutrient not in self.graph:
            return []

        related = []
        for neighbor in self.graph.neighbors(nutrient):
            edge_data = self.graph[nutrient][neighbor]
            if interaction_type is None or edge_data['type'] == interaction_type:
                related.append(neighbor)

        return related

    def get_interaction_matrix(self, nutrients: List[str]) -> np.ndarray:
        """
        Create an interaction matrix for a list of nutrients.

        Args:
            nutrients: List of nutrient names

        Returns:
            Matrix where element [i,j] is the interaction strength between nutrients i and j
        """
        n = len(nutrients)
        matrix = np.zeros((n, n))

        for i, nutrient1 in enumerate(nutrients):
            for j, nutrient2 in enumerate(nutrients):
                if i == j:
                    matrix[i, j] = 1.0  # Self-interaction
                else:
                    matrix[i, j] = self.get_interaction_strength(nutrient1, nutrient2)

        return matrix

    def get_all_interactions(self) -> List[Tuple[str, str, str, float]]:
        """
        Get all interactions in the network.

        Returns:
            List of tuples (nutrient1, nutrient2, type, strength)
        """
        interactions_list = []
        for (n1, n2), data in self.interactions.items():
            interactions_list.append((n1, n2, data['type'], data['strength']))
        return interactions_list

    def save_to_file(self, filepath: str):
        """
        Save interaction network to JSON file.

        Args:
            filepath: Path to save the network
        """
        data = {
            'interactions': [
                {
                    'nutrient1': n1,
                    'nutrient2': n2,
                    'type': info['type'],
                    'strength': info['strength']
                }
                for (n1, n2), info in self.interactions.items()
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filepath: str):
        """
        Load interaction network from JSON file.

        Args:
            filepath: Path to load the network from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.graph.clear()
        self.interactions.clear()

        for interaction in data['interactions']:
            self.add_interaction(
                interaction['nutrient1'],
                interaction['nutrient2'],
                interaction['type'],
                interaction['strength']
            )

    def visualize_network(self, save_path: str = None):
        """
        Create a visualization of the interaction network.

        Args:
            save_path: Optional path to save the visualization
        """
        try:
            import matplotlib.pyplot as plt

            pos = nx.spring_layout(self.graph, k=1, iterations=50)

            # Separate edges by type
            synergistic_edges = [(u, v) for u, v, d in self.graph.edges(data=True)
                                if d['type'] == 'synergistic']
            antagonistic_edges = [(u, v) for u, v, d in self.graph.edges(data=True)
                                 if d['type'] == 'antagonistic']
            metabolic_edges = [(u, v) for u, v, d in self.graph.edges(data=True)
                              if d['type'] == 'metabolic']
            absorption_edges = [(u, v) for u, v, d in self.graph.edges(data=True)
                               if d['type'] == 'absorption']

            plt.figure(figsize=(15, 10))

            # Draw nodes
            nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue',
                                  node_size=1500, alpha=0.9)
            nx.draw_networkx_labels(self.graph, pos, font_size=8, font_weight='bold')

            # Draw edges by type
            nx.draw_networkx_edges(self.graph, pos, edgelist=synergistic_edges,
                                  edge_color='green', width=2, alpha=0.7,
                                  label='Synergistic')
            nx.draw_networkx_edges(self.graph, pos, edgelist=antagonistic_edges,
                                  edge_color='red', width=2, alpha=0.7,
                                  label='Antagonistic')
            nx.draw_networkx_edges(self.graph, pos, edgelist=metabolic_edges,
                                  edge_color='blue', width=1.5, alpha=0.5,
                                  style='dashed', label='Metabolic')
            nx.draw_networkx_edges(self.graph, pos, edgelist=absorption_edges,
                                  edge_color='orange', width=1.5, alpha=0.5,
                                  style='dotted', label='Absorption')

            plt.title("Nutrient Interaction Network", fontsize=16)
            plt.legend(loc='upper left')
            plt.axis('off')

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
            plt.close()
        except ImportError:
            print("Matplotlib not available for visualization")


if __name__ == "__main__":
    # Test the interaction network
    network = NutrientInteractionNetwork()
    print("All interactions:", len(network.get_all_interactions()))
    print("\nVitamin D synergies:", network.get_related_nutrients('Vitamin_D', 'synergistic'))
    print("\nCalcium antagonists:", network.get_related_nutrients('Calcium', 'antagonistic'))
    print("\nInteraction (Vit D, Calcium):", network.get_interaction('Vitamin_D', 'Calcium'))
    print("\nInteraction (Calcium, Iron):", network.get_interaction('Calcium', 'Iron'))
