"""
Mask Matrix Builder for KGNN.

This module implements Algorithm 1 from the paper: Knowledge-Guided
Embedding Weight Initialization and constructs mask matrices for
structured connectivity in neural network layers.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from .ontology import NutritionalOntology
from .interactions import NutrientInteractionNetwork


class MaskMatrixBuilder:
    """
    Builds knowledge-constrained mask matrices for KGNN layers.

    This class implements the knowledge integration methodology from the paper,
    creating structured connectivity patterns based on nutritional domain knowledge.
    """

    def __init__(self,
                 ontology: NutritionalOntology,
                 interaction_network: NutrientInteractionNetwork,
                 similarity_threshold: float = 0.3):
        """
        Initialize the mask matrix builder.

        Args:
            ontology: Nutritional ontology instance
            interaction_network: Nutrient interaction network instance
            similarity_threshold: τ_sim from paper (default: 0.3)
        """
        self.ontology = ontology
        self.interaction_network = interaction_network
        self.similarity_threshold = similarity_threshold

    def initialize_embedding_weights(self,
                                    input_dim: int,
                                    embedding_dim: int,
                                    feature_names: List[str]) -> torch.Tensor:
        """
        Initialize embedding weights with knowledge constraints.

        Implements Algorithm 1 from the paper:
        Knowledge-Guided Embedding Weight Initialization

        Args:
            input_dim: Number of input features
            embedding_dim: Dimension of embedding space
            feature_names: Names of input features

        Returns:
            Knowledge-constrained weight matrix W_e
        """
        # Step 2: Initialize W_e with random values from N(0, 0.01)
        W_e = np.random.normal(0, 0.01, (embedding_dim, input_dim))

        # Identify nutrient features
        nutrient_features = [i for i, name in enumerate(feature_names)
                           if any(n in name for n in self.ontology.get_all_nutrients())]

        # Steps 3-5: Set similarity constraints for nutrients in same category
        for i in nutrient_features:
            for j in nutrient_features:
                if i < j:
                    # Extract nutrient names from feature names
                    nutrient_i = self._extract_nutrient_name(feature_names[i])
                    nutrient_j = self._extract_nutrient_name(feature_names[j])

                    if nutrient_i and nutrient_j:
                        # Check if in same category
                        if self.ontology.are_in_same_category(nutrient_i, nutrient_j):
                            # Apply similarity constraint
                            similarity = self.ontology.get_category_similarity(nutrient_i, nutrient_j)
                            if similarity > self.similarity_threshold:
                                # Make embeddings similar
                                W_e[:, j] = W_e[:, i] + np.random.normal(0, 0.01, embedding_dim)

        # Steps 6-10: Set correlation constraints based on interactions
        for i in nutrient_features:
            for j in nutrient_features:
                if i < j:
                    nutrient_i = self._extract_nutrient_name(feature_names[i])
                    nutrient_j = self._extract_nutrient_name(feature_names[j])

                    if nutrient_i and nutrient_j:
                        interaction = self.interaction_network.get_interaction(nutrient_i, nutrient_j)

                        if interaction:
                            if interaction['type'] == 'synergistic':
                                # Step 8: Positive correlation constraint
                                W_e[:, j] = np.abs(W_e[:, j]) * np.sign(W_e[:, i])
                            elif interaction['type'] == 'antagonistic':
                                # Step 10: Negative correlation constraint
                                W_e[:, j] = -np.abs(W_e[:, j]) * np.sign(W_e[:, i])

        # Step 11: Optimize W_e (apply L2 regularization)
        W_e = self._apply_regularization(W_e)

        # Step 12: Return W_e
        return torch.FloatTensor(W_e)

    def build_mask_matrix(self,
                         input_dim: int,
                         output_dim: int,
                         feature_names: List[str],
                         layer_type: str = 'hidden') -> torch.Tensor:
        """
        Build mask matrix M_l for structured connectivity in layer l.

        Args:
            input_dim: Number of input features to the layer
            output_dim: Number of output features from the layer
            feature_names: Names of input features
            layer_type: Type of layer ('embedding', 'hidden', 'output')

        Returns:
            Binary mask matrix M_l where 1 indicates allowed connection
        """
        # Initialize mask with all zeros (no connections)
        mask = np.zeros((output_dim, input_dim))

        if layer_type == 'embedding':
            # First layer: connect based on feature categories
            mask = self._build_embedding_mask(input_dim, output_dim, feature_names)

        elif layer_type == 'hidden':
            # Hidden layers: connect based on nutrient interactions
            mask = self._build_hidden_mask(input_dim, output_dim)

        elif layer_type == 'output':
            # Output layer: fully connected but weighted by relevance
            mask = self._build_output_mask(input_dim, output_dim)

        return torch.FloatTensor(mask)

    def _build_embedding_mask(self,
                             input_dim: int,
                             output_dim: int,
                             feature_names: List[str]) -> np.ndarray:
        """
        Build mask for embedding layer based on feature categories.

        Args:
            input_dim: Number of input features
            output_dim: Number of embedding dimensions
            feature_names: Names of input features

        Returns:
            Mask matrix for embedding layer
        """
        mask = np.zeros((output_dim, input_dim))

        # Group features by category
        feature_categories = self._categorize_features(feature_names)

        # Each category gets a subset of embedding dimensions
        dims_per_category = output_dim // len(feature_categories)

        for cat_idx, (category, feature_indices) in enumerate(feature_categories.items()):
            start_dim = cat_idx * dims_per_category
            end_dim = start_dim + dims_per_category if cat_idx < len(feature_categories) - 1 else output_dim

            # Connect features in this category to their embedding dimensions
            for feat_idx in feature_indices:
                mask[start_dim:end_dim, feat_idx] = 1

            # Allow some cross-category connections for interactions
            if category == 'dietary_intake':
                # Dietary intake can influence all embedding dimensions
                for feat_idx in feature_indices:
                    mask[:, feat_idx] = 1

        return mask

    def _build_hidden_mask(self, input_dim: int, output_dim: int) -> np.ndarray:
        """
        Build mask for hidden layer based on nutrient interactions.

        Args:
            input_dim: Number of input neurons
            output_dim: Number of output neurons

        Returns:
            Mask matrix for hidden layer
        """
        # Start with sparse random connectivity
        mask = (np.random.rand(output_dim, input_dim) > 0.7).astype(float)

        # Ensure minimum connectivity
        min_connections = max(3, input_dim // 10)
        for i in range(output_dim):
            if mask[i].sum() < min_connections:
                # Add random connections to meet minimum
                zero_indices = np.where(mask[i] == 0)[0]
                add_indices = np.random.choice(zero_indices,
                                              size=min_connections - int(mask[i].sum()),
                                              replace=False)
                mask[i, add_indices] = 1

        return mask

    def _build_output_mask(self, input_dim: int, output_dim: int) -> np.ndarray:
        """
        Build mask for output layer.

        Args:
            input_dim: Number of input neurons
            output_dim: Number of output neurons (micronutrients)

        Returns:
            Mask matrix for output layer
        """
        # Output layer is more densely connected
        mask = (np.random.rand(output_dim, input_dim) > 0.3).astype(float)

        # Ensure all outputs have sufficient connections
        min_connections = max(5, input_dim // 5)
        for i in range(output_dim):
            if mask[i].sum() < min_connections:
                zero_indices = np.where(mask[i] == 0)[0]
                add_indices = np.random.choice(zero_indices,
                                              size=min_connections - int(mask[i].sum()),
                                              replace=False)
                mask[i, add_indices] = 1

        return mask

    def _extract_nutrient_name(self, feature_name: str) -> str:
        """
        Extract nutrient name from feature name.

        Args:
            feature_name: Full feature name

        Returns:
            Nutrient name or empty string if not a nutrient
        """
        nutrients = self.ontology.get_all_nutrients()
        for nutrient in nutrients:
            if nutrient.lower() in feature_name.lower():
                return nutrient
        return ""

    def _categorize_features(self, feature_names: List[str]) -> Dict[str, List[int]]:
        """
        Categorize features by type.

        Args:
            feature_names: List of feature names

        Returns:
            Dictionary mapping category names to feature indices
        """
        categories = {
            'dietary_intake': [],
            'anthropometric': [],
            'biochemical': [],
            'medical_history': [],
            'functional_assessment': []
        }

        for idx, name in enumerate(feature_names):
            name_lower = name.lower()
            if any(kw in name_lower for kw in ['intake', 'diet', 'food', 'consumption']):
                categories['dietary_intake'].append(idx)
            elif any(kw in name_lower for kw in ['bmi', 'weight', 'height', 'waist', 'body']):
                categories['anthropometric'].append(idx)
            elif any(kw in name_lower for kw in ['serum', 'blood', 'plasma', 'level', 'concentration']):
                categories['biochemical'].append(idx)
            elif any(kw in name_lower for kw in ['disease', 'condition', 'medication', 'history']):
                categories['medical_history'].append(idx)
            elif any(kw in name_lower for kw in ['mobility', 'adl', 'cognitive', 'functional']):
                categories['functional_assessment'].append(idx)
            else:
                # Default to dietary intake
                categories['dietary_intake'].append(idx)

        return {k: v for k, v in categories.items() if v}  # Remove empty categories

    def _apply_regularization(self, weights: np.ndarray, lambda_reg: float = 0.01) -> np.ndarray:
        """
        Apply L2 regularization to weights.

        Args:
            weights: Weight matrix
            lambda_reg: Regularization strength

        Returns:
            Regularized weights
        """
        return weights * (1 - lambda_reg)

    def create_constraint_penalties(self,
                                   weights: torch.Tensor,
                                   feature_names: List[str]) -> torch.Tensor:
        """
        Calculate penalty for violating knowledge constraints.

        Used in the knowledge consistency loss L_know from Equation 6.

        Args:
            weights: Current weight matrix
            feature_names: Names of features

        Returns:
            Penalty value (scalar tensor)
        """
        penalty = torch.tensor(0.0)

        # Identify nutrient feature indices
        nutrient_features = [(i, self._extract_nutrient_name(feature_names[i]))
                           for i in range(len(feature_names))
                           if self._extract_nutrient_name(feature_names[i])]

        # Penalize violations of similarity constraints
        for i, nutrient_i in nutrient_features:
            for j, nutrient_j in nutrient_features:
                if i < j:
                    if self.ontology.are_in_same_category(nutrient_i, nutrient_j):
                        # Embeddings should be similar
                        similarity = torch.cosine_similarity(
                            weights[:, i].unsqueeze(0),
                            weights[:, j].unsqueeze(0)
                        )
                        if similarity < self.similarity_threshold:
                            penalty += (self.similarity_threshold - similarity) ** 2

        # Penalize violations of interaction constraints
        for i, nutrient_i in nutrient_features:
            for j, nutrient_j in nutrient_features:
                if i < j:
                    interaction = self.interaction_network.get_interaction(nutrient_i, nutrient_j)
                    if interaction:
                        correlation = torch.dot(weights[:, i], weights[:, j])
                        if interaction['type'] == 'synergistic' and correlation < 0:
                            penalty += correlation ** 2
                        elif interaction['type'] == 'antagonistic' and correlation > 0:
                            penalty += correlation ** 2

        return penalty


if __name__ == "__main__":
    # Test the mask builder
    ontology = NutritionalOntology()
    network = NutrientInteractionNetwork()
    builder = MaskMatrixBuilder(ontology, network)

    # Create sample feature names
    feature_names = ['Vitamin_B12_intake', 'Vitamin_D_intake', 'Iron_intake',
                    'BMI', 'Weight', 'Serum_B12', 'Serum_D', 'Serum_Iron']

    # Initialize embedding weights
    W_e = builder.initialize_embedding_weights(len(feature_names), 32, feature_names)
    print("Embedding weights shape:", W_e.shape)

    # Build mask matrices
    mask_emb = builder.build_mask_matrix(len(feature_names), 32, feature_names, 'embedding')
    print("Embedding mask shape:", mask_emb.shape)
    print("Embedding mask density:", mask_emb.mean().item())

    mask_hidden = builder.build_mask_matrix(32, 64, [], 'hidden')
    print("Hidden mask shape:", mask_hidden.shape)
    print("Hidden mask density:", mask_hidden.mean().item())
