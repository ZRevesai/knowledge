"""
NUTRINET Model Components

This module contains all the core components of the NUTRINET architecture:
- Hierarchical graph representation
- Edge-conditioned message passing
- Sparse attention mechanism
- Transparent prediction module
- Computational efficiency optimizations
"""

from .nutrinet import NUTRINET
from .graph_construction import HierarchicalNutrientGraph, build_nutrient_graph
from .message_passing import EdgeConditionedMessagePassing
from .attention import SparseAttention, ImportanceEstimator
from .transparent_prediction import TransparentPredictionModule
from .optimization import (
    QuantizedRepresentation,
    LazyEvaluation,
    AdaptiveComputation,
)

__all__ = [
    "NUTRINET",
    "HierarchicalNutrientGraph",
    "build_nutrient_graph",
    "EdgeConditionedMessagePassing",
    "SparseAttention",
    "ImportanceEstimator",
    "TransparentPredictionModule",
    "QuantizedRepresentation",
    "LazyEvaluation",
    "AdaptiveComputation",
]
