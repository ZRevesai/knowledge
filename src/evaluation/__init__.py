"""
Evaluation metrics and interpretability measures for KGNN.
"""

from .metrics import (
    calculate_performance_metrics,
    calculate_interpretability_metrics,
    evaluate_model
)
from .interpretability import (
    InterpretabilityEvaluator,
    explanation_completeness,
    explanation_consistency,
    feature_concentration
)

__all__ = [
    "calculate_performance_metrics",
    "calculate_interpretability_metrics",
    "evaluate_model",
    "InterpretabilityEvaluator",
    "explanation_completeness",
    "explanation_consistency",
    "feature_concentration",
]
