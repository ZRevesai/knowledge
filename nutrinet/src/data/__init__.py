"""
Data loading and preprocessing for NUTRINET.

This module provides dataset loaders for:
- USDA FoodData Central
- NHANES (National Health and Nutrition Examination Survey)
- Framingham Heart Study

All datasets support the hierarchical graph structure required by NUTRINET.
"""

from .dataset import (
    NutrientDataset,
    USDAFoodDataset,
    NHANESDataset,
    FraminghamDataset,
)
from .preprocessing import (
    NutrientPreprocessor,
    create_train_val_test_split,
    normalize_features,
)

__all__ = [
    "NutrientDataset",
    "USDAFoodDataset",
    "NHANESDataset",
    "FraminghamDataset",
    "NutrientPreprocessor",
    "create_train_val_test_split",
    "normalize_features",
]
