"""
Knowledge integration components for KGNN.
"""

from .ontology import NutritionalOntology
from .interactions import NutrientInteractionNetwork
from .mask_builder import MaskMatrixBuilder

__all__ = [
    "NutritionalOntology",
    "NutrientInteractionNetwork",
    "MaskMatrixBuilder",
]
