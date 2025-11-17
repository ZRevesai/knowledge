"""
NUTRINET: A Computationally Efficient Graph Neural Model
for Interpretable Nutrient Interaction Analysis

This package implements the NUTRINET model as described in:
Revesai, Z. and Kogeda, O. P. (2026). NUTRINET: A Computationally Efficient
Graph Neural Model for Interpretable Nutrient Interaction Analysis.
SAICSIT 2025, CCIS 2583, pp. 189-205.
"""

__version__ = "1.0.0"
__author__ = "Zvinodashe Revesai and Okuthe P. Kogeda"

from .model.nutrinet import NUTRINET
from .model.graph_construction import HierarchicalNutrientGraph
from .model.message_passing import EdgeConditionedMessagePassing
from .model.attention import SparseAttention
from .model.transparent_prediction import TransparentPredictionModule

__all__ = [
    "NUTRINET",
    "HierarchicalNutrientGraph",
    "EdgeConditionedMessagePassing",
    "SparseAttention",
    "TransparentPredictionModule",
]
