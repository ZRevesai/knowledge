# NUTRINET: A Computationally Efficient Graph Neural Model for Interpretable Nutrient Interaction Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.3+-3C8DBC.svg)](https://pytorch-geometric.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

NUTRINET is a novel graph neural network that combines **hierarchical nutrient graph representation**, **edge-conditioned message passing**, **sparse attention mechanisms**, and **transparent prediction modules** to achieve superior performance in nutrient analysis while reducing energy consumption by up to **73%** compared to Graph Attention Networks.

### Key Features

- 🎯 **Superior Predictive Performance**
  - MAE: 0.09 for nutrient prediction (vs 0.12 for GAT)
  - AUC: 0.91 for deficiency risk assessment (vs 0.89 for GAT)
  - F1-score: 0.85 for personalized recommendations (vs 0.81 for GAT)

- ⚡ **Computational Efficiency**
  - 73% energy reduction compared to Graph Attention Networks
  - Reduces complexity from O(n²) to O(k·n) where k << n
  - Adaptive computation based on input complexity

- 🔍 **Interpretability**
  - Built-in explanation generation through subgraph extraction
  - Critical pathway analysis
  - Human-readable summaries
  - 78% intervention change rate in clinical settings

- 🌍 **Population-Specific Adaptations**
  - Support for elderly populations
  - Pregnancy-specific modeling
  - Chronic disease considerations
  - Cultural food pattern integration

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- CUDA (optional, for GPU acceleration)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/ZRevesai/nutrinet.git
cd nutrinet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install NUTRINET
pip install -e .
```

### Install from source

```bash
git clone https://github.com/ZRevesai/nutrinet.git
cd nutrinet
pip install -r requirements.txt
python setup.py install
```

## Quick Start

### Basic Usage

```python
import torch
from src.model.nutrinet import create_nutrinet_model

# Create NUTRINET model with hierarchical graph
model, graph = create_nutrinet_model(
    num_micronutrients=50,
    num_food_components=30,
    num_health_outcomes=20,
    feature_dim=64,
    hidden_dim=64,
    output_dim=1,
)

# Get graph data
x = graph.node_features
edge_index = graph.edge_index
edge_features = graph.edge_features

# Create context (age, pregnancy status, medications, etc.)
context = torch.randn(1, 10)

# Forward pass with explanation
output = model(
    x=x,
    edge_index=edge_index,
    edge_features=edge_features,
    context=context,
    return_attention=True,
    return_explanation=True,
)

# Access results
predictions = output['predictions']
explanation = output['explanation']
efficiency_metrics = output['efficiency_metrics']

print(f"Prediction: {predictions.item():.4f}")
print(f"Explanation: {explanation['summary']}")
print(f"Energy efficiency: {efficiency_metrics['active_edge_ratio']:.2%} edges activated")
```

### Training

```bash
# Train on NHANES dataset
python train.py \
    --dataset nhanes \
    --data-dir ./data \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.001 \
    --device cuda

# Train with custom config
python train.py --config configs/custom_config.yaml
```

### Evaluation

```bash
# Evaluate on test set
python evaluate.py \
    --model-path checkpoints/best_model.pt \
    --dataset nhanes \
    --data-dir ./data
```

## Architecture

NUTRINET consists of four main components:

### 1. Hierarchical Graph Representation

Represents nutritional data as a multi-level graph:

```
G = (V, E, X, R)
```

where:
- **V**: Nodes (nutrients, biomarkers, health indicators)
- **E**: Edges (established interactions)
- **X**: Node features
- **R**: Edge features

**Three hierarchical levels:**
1. Micronutrient level (vitamins, minerals, amino acids)
2. Food component level (proteins, carbohydrates, fats)
3. Health outcome level (biomarkers, clinical indicators)

### 2. Edge-Conditioned Message Passing

Implements heterogeneous nutrient interactions:

```
M(i,j) = σ(W₁·hᵢ + W₂·hⱼ ⊙ φ(rᵢⱼ) + W₃·c)
hᵢ⁽ᵗ⁺¹⁾ = hᵢ⁽ᵗ⁾ + α·Σⱼ∈N(i) M(i,j)
```

Captures:
- Synergistic effects (e.g., vitamin C enhancing iron absorption)
- Antagonistic interactions (e.g., calcium inhibiting zinc absorption)
- Threshold-dependent effects (e.g., toxicity at high doses)

### 3. Sparse Attention Mechanism

Reduces computational complexity:

```
A(i,j) = softmax_j(S(hᵢ, hⱼ, rᵢⱼ)) · δ(hᵢ, hⱼ, rᵢⱼ)
```

where gating function:

```
δ(hᵢ, hⱼ, rᵢⱼ) = 1  if I(hᵢ, hⱼ, rᵢⱼ) > τ
                  0  otherwise
```

**Benefits:**
- Complexity: O(k·n) instead of O(n²)
- Only 18.7% of edges activated on average
- 73% energy reduction

### 4. Transparent Prediction Module

Provides interpretability through:

**Subgraph extraction (Eq. 8):**
```
Gₛ = {vᵢ, eᵢⱼ | ∂y/∂hᵢ > ε ∨ ∂y/∂rᵢⱼ > ε}
```

**Path analysis (Eq. 9):**
```
P = {p₁, p₂, ..., pₘ}  where  pₖ = (vₖ₁, vₖ₂, ..., vₖₜ)
```

**Explanation generation (Eq. 10):**
```
E(pₖ) = T(vₖ₁, vₖ₂, ..., vₖₜ, rₖ₁ₖ₂, ..., rₖ₍ₜ₋₁₎ₖₜ)
```

## Datasets

NUTRINET has been evaluated on three public datasets:

### 1. USDA FoodData Central
- **Size**: 50,000 food items with complete nutrient profiles
- **Use**: Nutrient composition prediction and food category analysis
- **URL**: https://fdc.nal.usda.gov/

### 2. NHANES
- **Size**: 8,500 participants with complete dietary and biomarker data
- **Use**: Biomarker-nutrient correlation and deficiency risk assessment
- **Features**: 200+ biomarkers across diverse U.S. populations
- **URL**: https://www.cdc.gov/nchs/nhanes/

### 3. Framingham Heart Study
- **Size**: 12,000 participants with 5+ years longitudinal data
- **Use**: Long-term health outcome prediction and causal inference
- **URL**: https://framinghamheartstudy.org/

## Results

### Performance Comparison (Table 1 from paper)

| Model | Nutrient Prediction (MAE↓) | Deficiency Risk (AUC↑) | Recommendation Quality (F1↑) |
|-------|---------------------------|------------------------|------------------------------|
| FeedForward | 0.16 | 0.82 | 0.74 |
| NutriCNN | 0.14 | 0.85 | 0.77 |
| GCN | 0.13 | 0.87 | 0.80 |
| GAT | 0.12 | 0.89 | 0.81 |
| RGCN | 0.11 | 0.89 | 0.83 |
| **NUTRINET** | **0.09** | **0.91** | **0.85** |

### Computational Efficiency (Table 2 from paper)

| Model | Training Time (h) | Inference Time (ms) | Memory (MB) | Energy (kWh) |
|-------|------------------|---------------------|-------------|--------------|
| FeedForward | 4.2 | 3.5 | 82 | 15.8 |
| NutriCNN | 12.7 | 8.2 | 245 | 47.3 |
| GCN | 7.8 | 5.3 | 124 | 28.4 |
| GAT | 16.5 | 12.7 | 286 | 59.2 |
| RGCN | 19.2 | 14.8 | 312 | 68.7 |
| **NUTRINET** | **5.3** | **4.1** | **98** | **18.5** |

**Key Findings:**
- **73% energy reduction** compared to GAT
- **68% energy reduction** compared to RGCN
- **42% reduction** in message passing iterations
- Minimal performance variance (≤2.6%) across population groups

### Interpretability Assessment (Table 3 from paper)

| Model | Explanation Fidelity | Explanation Conciseness (Words) | Clinical Relevance |
|-------|---------------------|--------------------------------|-------------------|
| LIME + FeedForward | 0.62 | 15.3 | 0.52 |
| LIME + NutriCNN | 0.67 | 13.8 | 0.58 |
| GNNExplainer + GCN | 0.71 | 11.2 | 0.63 |
| GNNExplainer + GAT | 0.75 | 10.5 | 0.67 |
| **NUTRINET** | **0.89** | **7.2** | **0.87** |

## Examples

### Example 1: Elderly Nutrition Assessment

```python
from src.model.nutrinet import create_nutrinet_model
from src.data.preprocessing import engineer_population_features
import torch

# Create model
model, graph = create_nutrinet_model()

# Prepare elderly-specific features
x = graph.node_features
age = torch.tensor([75.0])  # 75 years old
x_elderly = engineer_population_features(
    x.numpy(),
    population_type='elderly',
    age=age.numpy()
)

# Add context (elderly-specific)
context = torch.tensor([[
    75.0,      # age
    0.0,       # gender (male)
    26.5,      # BMI
    1.0,       # medication_flag
    0.8,       # absorption_factor
    # ... other context features
]])

# Get prediction with explanation
output = model(
    x=torch.FloatTensor(x_elderly),
    edge_index=graph.edge_index,
    edge_features=graph.edge_features,
    context=context,
    return_explanation=True,
)

print(output['explanation']['summary'])
# Output: "The model predicts a high deficiency risk (score: 0.76).
# Key contributing factors include: Calcium, Vitamin_D, Vitamin_B12.
# This prediction is supported by 8 metabolic pathways."
```

### Example 2: Energy Monitoring

```python
from src.utils.energy_monitor import EnergyMonitor, measure_energy_consumption

# Create energy monitor
monitor = EnergyMonitor()

# Measure energy during training
with measure_energy_consumption(monitor) as measurement:
    # Train model
    trainer.train(train_loader, val_loader, num_epochs=100)

# Print summary
monitor.print_summary()

# Compare with baseline
comparison = monitor.compare_with_baseline(
    baseline_energy_kwh=59.2,  # GAT energy from paper
    baseline_name="GAT"
)

print(f"Energy reduction: {comparison['energy_reduction_percent']:.1f}%")
# Output: Energy reduction: 73.0%
```

### Example 3: Batch Inference with Explanation

```python
from src.data.dataset import NHANESDataset
import torch

# Load dataset
dataset = NHANESDataset(data_dir='./data', split='test')

# Get batch
batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=10)))

# Inference with explanations
explanations = []
for i in range(len(batch['features'])):
    output = model(
        x=batch['features'][i:i+1],
        edge_index=graph.edge_index,
        edge_features=graph.edge_features,
        context=batch.get('demographics', None),
        return_explanation=True,
    )
    explanations.append(output['explanation'])

# Analyze explanations
from src.utils.metrics import compute_interpretability_metrics

interp_metrics = compute_interpretability_metrics(explanations)
print(f"Explanation fidelity: {interp_metrics['explanation_fidelity']:.2f}")
print(f"Clinical relevance: {interp_metrics['clinical_relevance']:.2f}")
```

## Project Structure

```
nutrinet/
├── src/
│   ├── model/
│   │   ├── nutrinet.py              # Main NUTRINET model
│   │   ├── graph_construction.py    # Hierarchical graph representation
│   │   ├── message_passing.py       # Edge-conditioned message passing
│   │   ├── attention.py            # Sparse attention mechanism
│   │   ├── transparent_prediction.py # Transparent prediction module
│   │   └── optimization.py         # Efficiency optimizations
│   ├── data/
│   │   ├── dataset.py             # Dataset classes
│   │   └── preprocessing.py        # Data preprocessing
│   └── utils/
│       ├── metrics.py             # Evaluation metrics
│       └── energy_monitor.py       # Energy consumption monitoring
├── examples/
│   ├── basic_usage.py            # Basic usage examples
│   └── advanced_usage.py          # Advanced examples
├── tests/
│   └── test_model.py             # Unit tests
├── configs/
│   └── default_config.yaml        # Default configuration
├── train.py                       # Training script
├── evaluate.py                    # Evaluation script
├── requirements.txt               # Python dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

## Configuration

Edit `configs/default_config.yaml` to customize model parameters:

```yaml
model:
  # Graph structure
  num_micronutrients: 50
  num_food_components: 30
  num_health_outcomes: 20

  # Model dimensions
  feature_dim: 64
  hidden_dim: 64
  edge_dim: 16
  context_dim: 10
  output_dim: 1

  # Attention
  num_attention_heads: 4
  sparse_attention_threshold: 0.5

  # Optimization
  enable_quantization: true
  num_quantization_bits: 8
  enable_adaptive_dim: true
  convergence_threshold: 0.01

training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.00001
  early_stopping_patience: 15

data:
  dataset: "nhanes"
  data_dir: "./data"
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
```

## Citation

If you use NUTRINET in your research, please cite:

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Authors

- **Zvinodashe Revesai** - *Initial work* - University of KwaZulu-Natal
- **Okuthe P. Kogeda** - *Supervisor* - University of KwaZulu-Natal

## Acknowledgments

- USDA FoodData Central for nutrient composition data
- NHANES dataset (1988-2018) for nutritional and biomarker data
- Framingham Heart Study for longitudinal cardiovascular data
- Research collaborators at geriatric care centers
- PyTorch Geometric team for graph neural network tools

## Contact

For questions or feedback, please open an issue on GitHub or contact:

- Zvinodashe Revesai: 224195689@stu.ukzn.ac.za
- Okuthe P. Kogeda: kogedao@ukzn.ac.za

---

**Keywords:** Graph Neural Networks, Computational Efficiency, Explainable AI, Nutritional Informatics, Green AI, Sustainable Computing, Vulnerable Populations
