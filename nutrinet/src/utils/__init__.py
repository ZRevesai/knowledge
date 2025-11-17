"""
Utility modules for NUTRINET.

Provides:
- Performance metrics (MAE, AUC, F1-score)
- Energy consumption monitoring
- Visualization tools
- Logging and tracking
"""

from .metrics import (
    compute_nutrient_prediction_metrics,
    compute_deficiency_risk_metrics,
    compute_recommendation_metrics,
    compute_interpretability_metrics,
)
from .energy_monitor import (
    EnergyMonitor,
    measure_energy_consumption,
)

__all__ = [
    "compute_nutrient_prediction_metrics",
    "compute_deficiency_risk_metrics",
    "compute_recommendation_metrics",
    "compute_interpretability_metrics",
    "EnergyMonitor",
    "measure_energy_consumption",
]
