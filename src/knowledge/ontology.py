"""
Nutritional Ontology Handler for KGNN.

This module implements the FoodOn-based nutritional ontology
for categorizing and relating nutrients based on domain knowledge.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set
import json


class NutritionalOntology:
    """
    Manages nutritional domain knowledge through an ontology structure.

    This class organizes nutrients into hierarchical categories and
    maintains relationships based on nutritional science principles.
    """

    def __init__(self):
        """Initialize the nutritional ontology."""
        self.graph = nx.DiGraph()
        self.nutrient_categories = {}
        self.feature_to_idx = {}
        self.idx_to_feature = {}
        self._build_ontology()

    def _build_ontology(self):
        """
        Build the nutritional ontology graph based on FoodOn principles.

        Organizes nutrients into categories:
        - Vitamins (Water-soluble, Fat-soluble)
        - Minerals (Macro, Trace)
        - Biomarkers
        - Anthropometric measures
        - Functional assessments
        """
        # Root categories
        categories = {
            'vitamins': {
                'water_soluble': ['Vitamin_B12', 'Vitamin_B6', 'Folate', 'Vitamin_C'],
                'fat_soluble': ['Vitamin_D', 'Vitamin_E', 'Vitamin_A']
            },
            'minerals': {
                'macro': ['Calcium', 'Magnesium'],
                'trace': ['Iron', 'Zinc', 'Selenium']
            }
        }

        # Build hierarchical structure
        self.graph.add_node('root', type='root')

        # Add vitamin hierarchy
        self.graph.add_node('vitamins', type='category')
        self.graph.add_edge('root', 'vitamins')

        self.graph.add_node('water_soluble', type='subcategory')
        self.graph.add_edge('vitamins', 'water_soluble')
        for nutrient in categories['vitamins']['water_soluble']:
            self.graph.add_node(nutrient, type='nutrient', category='water_soluble')
            self.graph.add_edge('water_soluble', nutrient)
            self.nutrient_categories[nutrient] = 'water_soluble'

        self.graph.add_node('fat_soluble', type='subcategory')
        self.graph.add_edge('vitamins', 'fat_soluble')
        for nutrient in categories['vitamins']['fat_soluble']:
            self.graph.add_node(nutrient, type='nutrient', category='fat_soluble')
            self.graph.add_edge('fat_soluble', nutrient)
            self.nutrient_categories[nutrient] = 'fat_soluble'

        # Add mineral hierarchy
        self.graph.add_node('minerals', type='category')
        self.graph.add_edge('root', 'minerals')

        self.graph.add_node('macro_minerals', type='subcategory')
        self.graph.add_edge('minerals', 'macro_minerals')
        for nutrient in categories['minerals']['macro']:
            self.graph.add_node(nutrient, type='nutrient', category='macro_minerals')
            self.graph.add_edge('macro_minerals', nutrient)
            self.nutrient_categories[nutrient] = 'macro_minerals'

        self.graph.add_node('trace_minerals', type='subcategory')
        self.graph.add_edge('minerals', 'trace_minerals')
        for nutrient in categories['minerals']['trace']:
            self.graph.add_node(nutrient, type='nutrient', category='trace_minerals')
            self.graph.add_edge('trace_minerals', nutrient)
            self.nutrient_categories[nutrient] = 'trace_minerals'

    def get_nutrients_in_category(self, category: str) -> List[str]:
        """
        Get all nutrients belonging to a specific category.

        Args:
            category: Category name (e.g., 'water_soluble', 'trace_minerals')

        Returns:
            List of nutrient names in the category
        """
        return [n for n, cat in self.nutrient_categories.items() if cat == category]

    def are_in_same_category(self, nutrient1: str, nutrient2: str) -> bool:
        """
        Check if two nutrients belong to the same category.

        Args:
            nutrient1: First nutrient name
            nutrient2: Second nutrient name

        Returns:
            True if nutrients are in the same category
        """
        if nutrient1 not in self.nutrient_categories or nutrient2 not in self.nutrient_categories:
            return False
        return self.nutrient_categories[nutrient1] == self.nutrient_categories[nutrient2]

    def get_category_similarity(self, nutrient1: str, nutrient2: str) -> float:
        """
        Calculate semantic similarity between two nutrients based on ontology.

        Args:
            nutrient1: First nutrient name
            nutrient2: Second nutrient name

        Returns:
            Similarity score between 0 and 1
        """
        if nutrient1 not in self.graph or nutrient2 not in self.graph:
            return 0.0

        # Same nutrient
        if nutrient1 == nutrient2:
            return 1.0

        # Same category gets high similarity
        if self.are_in_same_category(nutrient1, nutrient2):
            return 0.8

        # Different categories but same parent (e.g., both vitamins)
        try:
            path1 = nx.shortest_path(self.graph, 'root', nutrient1)
            path2 = nx.shortest_path(self.graph, 'root', nutrient2)

            # Find common ancestors
            common = set(path1) & set(path2)
            if 'vitamins' in common or 'minerals' in common:
                return 0.4
        except nx.NetworkXNoPath:
            pass

        return 0.1

    def get_all_nutrients(self) -> List[str]:
        """
        Get list of all nutrients in the ontology.

        Returns:
            List of all nutrient names
        """
        return list(self.nutrient_categories.keys())

    def save_to_file(self, filepath: str):
        """
        Save ontology structure to JSON file.

        Args:
            filepath: Path to save the ontology
        """
        data = {
            'categories': self.nutrient_categories,
            'edges': list(self.graph.edges())
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filepath: str):
        """
        Load ontology structure from JSON file.

        Args:
            filepath: Path to load the ontology from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.nutrient_categories = data['categories']
        # Rebuild graph from edges
        self.graph.clear()
        for source, target in data['edges']:
            self.graph.add_edge(source, target)

    def visualize_ontology(self, save_path: str = None):
        """
        Create a visualization of the ontology structure.

        Args:
            save_path: Optional path to save the visualization
        """
        try:
            import matplotlib.pyplot as plt

            pos = nx.spring_layout(self.graph, k=2, iterations=50)

            # Color nodes by type
            node_colors = []
            for node in self.graph.nodes():
                node_type = self.graph.nodes[node].get('type', 'other')
                if node_type == 'root':
                    node_colors.append('red')
                elif node_type == 'category':
                    node_colors.append('orange')
                elif node_type == 'subcategory':
                    node_colors.append('yellow')
                elif node_type == 'nutrient':
                    node_colors.append('lightblue')
                else:
                    node_colors.append('gray')

            plt.figure(figsize=(15, 10))
            nx.draw(self.graph, pos, node_color=node_colors,
                   with_labels=True, node_size=1000,
                   font_size=8, font_weight='bold',
                   arrows=True, edge_color='gray')
            plt.title("Nutritional Ontology Structure", fontsize=16)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
            plt.close()
        except ImportError:
            print("Matplotlib not available for visualization")


if __name__ == "__main__":
    # Test the ontology
    ontology = NutritionalOntology()
    print("All nutrients:", ontology.get_all_nutrients())
    print("\nWater-soluble vitamins:", ontology.get_nutrients_in_category('water_soluble'))
    print("\nTrace minerals:", ontology.get_nutrients_in_category('trace_minerals'))
    print("\nSimilarity (B12, B6):", ontology.get_category_similarity('Vitamin_B12', 'Vitamin_B6'))
    print("Similarity (B12, Iron):", ontology.get_category_similarity('Vitamin_B12', 'Iron'))
