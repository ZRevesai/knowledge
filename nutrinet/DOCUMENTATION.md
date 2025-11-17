# NUTRINET Implementation Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture Details](#architecture-details)
3. [Module Reference](#module-reference)
4. [Implementation Details](#implementation-details)
5. [Training Procedure](#training-procedure)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Energy Monitoring](#energy-monitoring)
8. [Reproducibility](#reproducibility)

---

## Overview

This document provides comprehensive technical documentation for the NUTRINET implementation, covering all modules, algorithms, and experimental procedures described in the paper:

> **Revesai, Z. and Kogeda, O. P. (2026)**. NUTRINET: A Computationally Efficient Graph Neural Model for Interpretable Nutrient Interaction Analysis. *SAICSIT 2025, CCIS 2583, pp. 189-205.*

---

## Architecture Details

### 1. Hierarchical Nutrient Graph Representation

**Module**: `src/model/graph_construction.py`

**Implementation**: `HierarchicalNutrientGraph`

The hierarchical graph G = (V, E, X, R) is constructed as follows:

```python
class HierarchicalNutrientGraph:
    """
    Three-level hierarchy:
    - Level 0: Micronutrients (vitamins, minerals, amino acids)
    - Level 1: Food components (proteins, carbohydrates, fats)
    - Level 2: Health outcomes (biomarkers, clinical indicators)
    """
```

**Key Features**:
- Automatic graph construction from nutritional databases
- Support for custom interaction types (synergistic, antagonistic, threshold-dependent)
- Efficient tensor representation for PyTorch Geometric

**Algorithm 1 Implementation**:
```python
def build_graph(self, nutrient_database, interaction_database):
    self._initialize_nodes(nutrient_database)      # Step 1: Create nodes
    self._build_edges(interaction_database)         # Step 2: Create edges
    return self._to_tensors()                       # Step 3: Convert to tensors
```

---

### 2. Edge-Conditioned Message Passing

**Module**: `src/model/message_passing.py`

**Implementation**: `EdgeConditionedMessagePassing`

**Equation (2)**: Message function
```python
def message(self, x_i, x_j, edge_features, context):
    edge_transformed = self.edge_transform(edge_features)
    source_component = self.W1(x_i)
    target_component = self.W2(x_j) * edge_transformed
    message = source_component + target_component
    if context is not None:
        message = message + self.W3(context)
    return F.relu(message)
```

**Equation (3)**: Node update
```python
def forward(self, x, edge_index, edge_features, context):
    out = self.propagate(edge_index, x=x, edge_features=edge_features, context=context)
    x_updated = x + self.alpha * out
    return self.layer_norm(x_updated)
```

**Features**:
- Heterogeneous interaction modeling
- Context-aware message passing
- Residual connections for stable training
- Layer normalization

---

### 3. Sparse Attention Mechanism

**Module**: `src/model/attention.py`

**Implementation**: `SparseAttention`

**Equation (11)**: Sparse attention
```python
def forward(self, x, edge_index, edge_features):
    # Compute importance scores
    importance = self.importance_estimator(h_i, h_j, edge_features)

    # Gating function (Eq. 12)
    active_edges = (importance > self.tau).float()

    # Compute attention scores
    attention_scores = self._compute_attention_scores(h_i, h_j, edge_features)

    # Apply gating
    attention_scores = attention_scores * active_edges

    # Softmax normalization
    attention_weights = self._softmax_per_node(attention_scores, row, num_nodes)

    return attended_features
```

**Complexity Reduction**:
- Standard attention: O(n²)
- Sparse attention: O(k·n) where k << n
- Average activation: 18.7% of edges

---

### 4. Transparent Prediction Module

**Module**: `src/model/transparent_prediction.py`

**Implementation**: `TransparentPredictionModule`

**Three-Step Process**:

1. **Subgraph Extraction (Eq. 8)**:
```python
def extract_subgraph(self, prediction, node_features, edge_features):
    # Compute gradients
    grad_nodes = torch.autograd.grad(prediction, node_features)
    grad_edges = torch.autograd.grad(prediction, edge_features)

    # Identify significant nodes/edges
    significant_nodes = (torch.norm(grad_nodes, dim=-1) > self.threshold)
    significant_edges = (torch.norm(grad_edges, dim=-1) > self.threshold)

    return significant_nodes, significant_edges
```

2. **Path Analysis (Eq. 9)**:
```python
def find_critical_paths(self, source_nodes, target_nodes):
    # Build NetworkX graph
    G = self._build_networkx_graph(...)

    # Find all simple paths
    paths = []
    for source in source_nodes:
        for target in target_nodes:
            paths.extend(nx.all_simple_paths(G, source, target, cutoff=max_length))

    # Rank by importance
    return sorted(paths, key=lambda p: self._compute_path_score(p))[:top_k]
```

3. **Explanation Generation (Eq. 10)**:
```python
def generate_explanation(self, paths, ...):
    explanation = {
        'summary': self._generate_summary(...),
        'critical_paths': [self._explain_path(p) for p in paths],
        'key_nutrients': self._extract_key_nutrients(paths),
    }
    return explanation
```

---

### 5. Computational Efficiency Optimizations

**Module**: `src/model/optimization.py`

**Four Optimization Techniques**:

1. **Quantized Representation (Eq. 4)**:
```python
class QuantizedRepresentation:
    def quantize(self, x):
        scale = (x.max() - x.min()) / (self.num_levels - 1)
        x_quantized = torch.round(x / scale) * scale
        return x_quantized, scale
```

2. **Lazy Evaluation (Eq. 5)**:
```python
class LazyEvaluation:
    def forward(self, messages, active_edges):
        masked_messages = messages * active_edges.unsqueeze(-1)
        return masked_messages
```

3. **Early Stopping (Eq. 6)**:
```python
class EarlyStopping:
    def check_convergence(self, h_current):
        if self.prev_h is not None:
            delta = torch.norm(h_current - self.prev_h, p=2)
            if delta < self.convergence_threshold:
                return True
        return False
```

4. **Adaptive Computation (Eq. 7)**:
```python
class AdaptiveComputation:
    def compute_dimension(self, x, edge_index):
        complexity = self.estimate_complexity(x, edge_index)
        dim = int(self.d_min + complexity * (self.d_max - self.d_min))
        return max(self.d_min, min(self.d_max, dim))
```

---

## Module Reference

### Core Model (`src/model/`)

| Module | Description | Key Classes |
|--------|-------------|-------------|
| `nutrinet.py` | Main NUTRINET model | `NUTRINET`, `create_nutrinet_model` |
| `graph_construction.py` | Hierarchical graph builder | `HierarchicalNutrientGraph` |
| `message_passing.py` | Edge-conditioned MP | `EdgeConditionedMessagePassing` |
| `attention.py` | Sparse attention | `SparseAttention`, `ImportanceEstimator` |
| `transparent_prediction.py` | Explanation generation | `TransparentPredictionModule` |
| `optimization.py` | Efficiency techniques | `QuantizedRepresentation`, `AdaptiveComputation` |

### Data (`src/data/`)

| Module | Description | Key Classes |
|--------|-------------|-------------|
| `dataset.py` | Dataset loaders | `USDAFoodDataset`, `NHANESDataset`, `FraminghamDataset` |
| `preprocessing.py` | Data preprocessing | `NutrientPreprocessor` |

### Utilities (`src/utils/`)

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `metrics.py` | Evaluation metrics | `compute_nutrient_prediction_metrics`, `compute_deficiency_risk_metrics` |
| `energy_monitor.py` | Energy tracking | `EnergyMonitor`, `measure_energy_consumption` |

---

## Implementation Details

### Hyperparameters (Section 3.6)

As specified in the paper:

```yaml
Model:
  - Node embedding dimension: 64 (adaptive: 32-128)
  - Message passing steps: 3 (adaptive: 2-5)
  - Attention heads: 4
  - Quantization: 8-bit (node), 4-bit (edge)

Training:
  - Optimizer: Adam
  - Learning rate: 0.001
  - Weight decay: 1e-5
  - Batch size: 32
  - Epochs: 100 (with early stopping)
```

### Population-Specific Adaptations (Section 3.7)

```python
# Context vector encoding (Equation 2)
context = torch.tensor([
    age,                    # Age-related absorption changes
    gender,                 # Sex-specific requirements
    bmi,                    # Body composition
    medication_flag,        # Drug-nutrient interactions
    pregnancy_trimester,    # Pregnancy status
    chronic_condition,      # Disease state
    absorption_factor,      # Physiological state
    mobility_score,         # Activity level
    dietary_diversity,      # Food access
    supplement_use,         # Supplementation
])
```

---

## Training Procedure

### Standard Training Loop

```python
# Create model
model, graph = create_nutrinet_model(...)

# Setup training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.MSELoss()

# Training loop
trainer = NUTRINETTrainer(model, optimizer, criterion)
history = trainer.train(train_loader, val_loader, num_epochs=100)
```

### 5-Fold Cross-Validation

```python
from src.data.preprocessing import create_cross_validation_folds

# Create folds
folds = create_cross_validation_folds(X, y, n_folds=5)

# Train on each fold
for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(folds):
    model = create_nutrinet_model(...)
    # Train on this fold
    ...
```

---

## Evaluation Metrics

### Primary Metrics (Table 1)

```python
from src.utils.metrics import compute_all_metrics

# Nutrient Prediction Task
metrics = compute_all_metrics(
    task_type='nutrient_prediction',
    y_true=y_true,
    y_pred=y_pred,
)
# Returns: {'mae': 0.09, 'rmse': ..., 'r2': ...}

# Deficiency Risk Assessment
metrics = compute_all_metrics(
    task_type='deficiency_risk',
    y_true=y_true,
    y_pred=y_pred,
    y_pred_proba=y_pred_proba,
)
# Returns: {'auc': 0.91, 'precision': ..., 'recall': ...}

# Recommendation Quality
metrics = compute_all_metrics(
    task_type='recommendation',
    y_true=y_true,
    y_pred=y_pred,
)
# Returns: {'f1': 0.85, 'precision': ..., 'recall': ...}
```

### Interpretability Metrics (Table 3)

```python
from src.utils.metrics import compute_interpretability_metrics

# Generate explanations
explanations = [
    model.explain_prediction(x_i, edge_index, edge_features)
    for x_i in test_samples
]

# Compute metrics
interp_metrics = compute_interpretability_metrics(explanations)
# Returns: {
#     'explanation_fidelity': 0.89,
#     'explanation_conciseness': 7.2,
#     'clinical_relevance': 0.87
# }
```

---

## Energy Monitoring

### Methodology (Section 3.6)

Following Li et al. [37], energy consumption is tracked during both training and inference:

```python
from src.utils.energy_monitor import EnergyMonitor, measure_energy_consumption

# Method 1: Context manager
with measure_energy_consumption() as measurement:
    # Train or run inference
    model.train()

print(f"Energy: {measurement.total_energy_kwh} kWh")
print(f"CO2: {measurement.co2_emissions_kg} kg")

# Method 2: Manual monitoring
monitor = EnergyMonitor()
monitor.start_tracking()
# ... training code ...
measurement = monitor.stop_tracking()
```

### Baseline Comparisons (Table 2)

```python
# Compare with baselines
comparison = monitor.compare_with_baseline(
    baseline_energy_kwh=59.2,  # GAT energy from paper
    baseline_name="GAT"
)

print(f"Energy reduction: {comparison['energy_reduction_percent']:.1f}%")
# Expected output: "Energy reduction: 73.0%"
```

---

## Reproducibility

### Random Seeds

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # As used in paper
```

### Dataset Splits

```python
from src.data.preprocessing import create_train_val_test_split

# 80:20 train:test split (as per paper)
splits = create_train_val_test_split(
    X, y,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_state=42,
)
```

### Expected Results

Based on Tables 1-3 from the paper:

| Metric | Expected Value | Tolerance |
|--------|---------------|-----------|
| MAE (nutrient prediction) | 0.09 | ±0.01 |
| AUC (deficiency risk) | 0.91 | ±0.02 |
| F1-score (recommendations) | 0.85 | ±0.02 |
| Energy consumption (kWh) | 18.5 | ±2.0 |
| Energy reduction vs GAT | 73% | ±5% |
| Explanation fidelity | 0.89 | ±0.05 |

---

## API Reference

### Main Model

```python
from src.model.nutrinet import NUTRINET, create_nutrinet_model

# Create model
model, graph = create_nutrinet_model(
    num_micronutrients=50,
    num_food_components=30,
    num_health_outcomes=20,
    feature_dim=64,
    hidden_dim=64,
    output_dim=1,
    context_dim=10,
)

# Forward pass
output = model(
    x=node_features,              # [num_nodes, feature_dim]
    edge_index=edge_index,        # [2, num_edges]
    edge_features=edge_features,  # [num_edges, edge_dim]
    context=context,              # [batch_size, context_dim]
    return_attention=True,        # Return attention weights
    return_explanation=True,      # Generate explanation
)

# Output dictionary
{
    'predictions': tensor,         # [batch_size, output_dim]
    'node_embeddings': tensor,     # [num_nodes, hidden_dim]
    'graph_embedding': tensor,     # [batch_size, hidden_dim]
    'attention_weights': tensor,   # [num_edges] (optional)
    'active_edges': tensor,        # [num_edges] (optional)
    'explanation': dict,           # (optional)
    'efficiency_metrics': dict,    # Always included
}
```

### Explanation

```python
# Generate explanation
explanation = model.explain_prediction(
    x=node_features,
    edge_index=edge_index,
    edge_features=edge_features,
    context=context,
    source_nodes=[0, 1, 2],  # Input nutrients
    target_nodes=[35, 36],    # Output biomarkers
)

# Explanation dictionary
{
    'summary': str,                    # Natural language summary
    'critical_paths': List[dict],      # Identified pathways
    'key_nutrients': List[dict],       # Important nutrients
    'significant_nodes': List[int],    # Node IDs in subgraph
    'significant_edges': List[int],    # Edge IDs in subgraph
    'node_importance': List[float],    # Importance scores
    'edge_importance': List[float],    # Importance scores
}
```

---

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use smaller hidden dimensions

2. **Slow Training**
   - Enable mixed precision (`use_amp=True`)
   - Increase number of workers for data loading
   - Use sparse attention with lower threshold

3. **Poor Convergence**
   - Check learning rate
   - Verify data normalization
   - Increase model capacity

4. **Energy Tracking Not Working**
   - Install codecarbon: `pip install codecarbon`
   - Install pynvml for GPU: `pip install pynvml`
   - Check permissions for power monitoring

---

## Citation

```bibtex
@inproceedings{revesai2026nutrinet,
  title={NUTRINET: A Computationally Efficient Graph Neural Model for Interpretable Nutrient Interaction Analysis},
  author={Revesai, Zvinodashe and Kogeda, Okuthe P.},
  booktitle={SAICSIT 2025},
  series={CCIS},
  volume={2583},
  pages={189--205},
  year={2026},
  publisher={Springer Nature Switzerland AG},
  doi={10.1007/978-3-031-96262-2_13}
}
```

---

**Last Updated**: 2025-01-17
**Implementation Version**: 1.0.0
