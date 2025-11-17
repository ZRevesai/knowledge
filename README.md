# Knowledge-Guided Neural Networks (KGNN) for Elderly Nutrition

A deep learning framework for micronutrient deficiency detection in elderly populations using interpretable AI with integrated nutritional domain knowledge.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This repository contains the complete implementation of **Knowledge-Guided Neural Networks (KGNNs)** for detecting micronutrient deficiencies in elderly populations. KGNN integrates nutritional domain knowledge directly into deep learning architectures to achieve:

- **93.7% deficiency detection accuracy** (vs 88.2% for standard neural networks)
- **41% better interpretability** than post-hoc methods (SHAP, LIME)
- Transparent explanations via concept activation vectors, attention visualization, and counterfactual recommendations
- Support for 12 key micronutrients including vitamins and minerals

## Key Features

### 🧠 Knowledge Integration
- Nutritional ontology based on FoodOn principles
- Biochemical interaction networks for nutrient relationships
- Structured connectivity through knowledge-guided mask matrices

### 📊 Model Architecture
- Knowledge Embedding Layer with domain-constrained initialization
- Knowledge-Guided Interaction Layers with structured connections
- Multi-head attention mechanism for interpretability
- Interpretable output layer with per-micronutrient predictions

### 🎯 Multi-Objective Training
- Prediction loss (L_pred) for accuracy
- Knowledge consistency loss (L_know) for domain coherence
- Explanation quality loss (L_expl) for interpretability
- Early stopping and learning rate scheduling

### 📈 Comprehensive Evaluation
- Performance metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Interpretability metrics: Explanation Completeness, Consistency, Feature Concentration
- Comparison with baseline models (FFNN, XGBoost, Transformer, LIME, SHAP)

### 🎨 Visualization Tools
- Training history and loss curves
- Attention heatmaps and feature importance
- ROC curves and confusion matrices
- Performance vs interpretability trade-off charts

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+
- NumPy, Pandas, Scikit-learn
- XGBoost, SHAP, LIME (for baselines)
- Matplotlib, Seaborn (for visualization)

### Install via pip

```bash
# Clone the repository
git clone https://github.com/ZRevesai/knowledge.git
cd knowledge

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### 1. Generate Synthetic Data

```python
from src.data.preprocessing import NutritionalDataPreprocessor

# Initialize preprocessor
preprocessor = NutritionalDataPreprocessor(random_state=42)

# Generate synthetic NHANES-style data
features_df, labels_df = preprocessor.generate_synthetic_data(n_samples=5000)

# Preprocess and split
X, y = preprocessor.fit_transform(features_df, labels_df)
data_splits = preprocessor.split_data(X, y)
```

### 2. Build Knowledge Components

```python
from src.knowledge import (
    NutritionalOntology,
    NutrientInteractionNetwork,
    MaskMatrixBuilder
)

# Create ontology and interaction network
ontology = NutritionalOntology()
interaction_network = NutrientInteractionNetwork()

# Build mask matrices
mask_builder = MaskMatrixBuilder(ontology, interaction_network)
embedding_weights = mask_builder.initialize_embedding_weights(
    input_dim=X.shape[1],
    embedding_dim=128,
    feature_names=preprocessor.feature_names
)
```

### 3. Train KGNN Model

```python
from src.models import KGNN
from src.training import KGNNTrainer, MultiObjectiveLoss
from src.data import create_dataloaders
import torch

# Create model
model = KGNN(
    input_dim=X.shape[1],
    embedding_dim=128,
    hidden_dims=[256, 128, 64],
    num_micronutrients=12,
    embedding_weights=embedding_weights
)

# Create dataloaders
dataloaders = create_dataloaders(
    data_splits,
    batch_size=64,
    micronutrient_names=preprocessor.micronutrient_names
)

# Setup training
criterion = MultiObjectiveLoss(
    lambda_k=0.3,
    lambda_e=0.2,
    mask_builder=mask_builder,
    feature_names=preprocessor.feature_names
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train model
trainer = KGNNTrainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

history = trainer.train(
    train_loader=dataloaders['train'],
    val_loader=dataloaders['val'],
    num_epochs=100
)
```

### 4. Evaluate and Visualize

```python
from src.evaluation import evaluate_model, print_evaluation_results
from src.visualization import (
    plot_training_history,
    plot_attention_heatmap,
    visualize_explanation
)

# Evaluate model
results = evaluate_model(
    model=model,
    data_loader=dataloaders['test'],
    device='cpu',
    return_predictions=True
)

# Print results
print_evaluation_results(
    results,
    model_name="KGNN",
    micronutrient_names=preprocessor.micronutrient_names
)

# Visualize training
plot_training_history(history, save_path='figures/training_history.png')

# Visualize attention
plot_attention_heatmap(
    results['predictions']['attention_weights'],
    feature_names=preprocessor.feature_names,
    save_path='figures/attention_heatmap.png'
)

# Get explanation for a single sample
sample = torch.FloatTensor(X[0:1])
explanation = model.explain_prediction(sample, preprocessor.feature_names)
visualize_explanation(explanation, save_path='figures/explanation.png')
```

## Project Structure

```
knowledge/
├── src/
│   ├── knowledge/          # Knowledge integration components
│   │   ├── ontology.py     # Nutritional ontology
│   │   ├── interactions.py # Nutrient interaction network
│   │   └── mask_builder.py # Mask matrix construction
│   ├── models/             # Model architectures
│   │   ├── kgnn.py         # KGNN main model
│   │   ├── layers.py       # Custom layers
│   │   └── baselines.py    # Baseline models
│   ├── data/               # Data preprocessing
│   │   ├── preprocessing.py # Data preprocessor
│   │   └── dataset.py      # PyTorch datasets
│   ├── training/           # Training components
│   │   ├── losses.py       # Loss functions
│   │   └── trainer.py      # Training loop
│   ├── evaluation/         # Evaluation metrics
│   │   ├── metrics.py      # Performance metrics
│   │   └── interpretability.py # Interpretability metrics
│   └── visualization/      # Visualization tools
│       ├── plots.py        # Performance plots
│       └── attention_viz.py # Attention visualization
├── scripts/                # Executable scripts
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   └── compare_models.py  # Model comparison
├── notebooks/             # Jupyter notebooks
│   └── demo.ipynb        # Demo notebook
├── configs/              # Configuration files
│   └── config.yaml       # Default config
├── data/                 # Data directory
├── results/              # Results directory
├── requirements.txt      # Dependencies
├── setup.py             # Package setup
└── README.md            # This file
```

## Model Architecture

KGNN consists of four main components:

### 1. Knowledge Embedding Layer
Transforms input features into semantically meaningful representations:
```
φ(x) = σ(W_e × x + b_e)
```
where `W_e` is initialized with knowledge constraints from nutritional ontology.

### 2. Knowledge-Guided Interaction Layers
Structures connections based on known nutritional principles:
```
h_l = σ(M_l ⊙ (W_l × h_{l-1}) + b_l)
```
where `M_l` is a mask matrix encoding nutritional interactions.

### 3. Attention Mechanism
Highlights influential factors for interpretability:
```
α = softmax(v^T × tanh(W_a × h_l + b_a))
z = Σ_i α_i × h_{l,i}
```

### 4. Interpretable Output Layer
Produces per-micronutrient deficiency probabilities:
```
p(d_m | x) = σ(w_m^T × z + b_m)
```

## Results

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Interpretability |
|-------|----------|-----------|--------|----------|------------------|
| **KGNN** | **93.7%** | **92.4%** | **91.8%** | **92.1%** | **0.90** |
| Transformer | 89.1% | 87.3% | 88.2% | 87.7% | 0.48 |
| FFNN | 88.2% | 86.1% | 85.7% | 85.9% | 0.28 |
| XGBoost | 85.4% | 83.2% | 84.8% | 84.0% | 0.62 |
| FFNN+SHAP | 88.2% | 86.1% | 85.7% | 85.9% | 0.66 |
| LIME-based | 81.3% | 80.5% | 79.8% | 80.1% | 0.67 |

### Key Findings

- **5.5% accuracy improvement** over standard neural networks
- **41% interpretability improvement** over best baseline (FFNN+SHAP)
- Particularly strong performance on nutrients with complex interactions (e.g., B12: 95.2% vs 87.6%)
- Clinically meaningful explanations validated by domain experts

## Configuration

Edit `configs/config.yaml` to customize model parameters:

```yaml
model:
  embedding_dim: 128
  hidden_dims: [256, 128, 64]
  dropout_rate: 0.3

knowledge:
  similarity_threshold: 0.3  # τ_sim
  knowledge_weight: 0.3      # λ_k
  explanation_weight: 0.2    # λ_e

training:
  batch_size: 64
  num_epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 15
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{kgnn2025,
  title={Knowledge-Guided Neural Networks for Micronutrient Deficiency Detection in Elderly Populations},
  author={[Authors]},
  journal={[Conference/Journal]},
  year={2025}
}
```

## Paper Abstract

Micronutrient deficiencies significantly impact elderly health outcomes, increasing morbidity, cognitive decline, and reducing quality of life. This paper presents Knowledge-Guided Neural Networks (KGNNs), integrating nutritional domain knowledge into deep learning through ontologies, biochemical pathways, and clinical guidelines. Using 5,000 elderly nutritional profiles, KGNNs achieve 93.7% deficiency detection accuracy while providing explanations via concept activation vectors, attention visualisation, and counterfactual recommendations. KGNNs outperform black-box models (88.2%) and conventional machine learning (85.4%), with 41% better interpretability than post-hoc methods.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue on GitHub.

## Acknowledgments

- NHANES dataset (1988-2018) for nutritional data
- FoodOn ontology for nutritional categorization
- Research collaborators at geriatric care centers

---

**Keywords:** Deep learning, interpretability, nutritional assessment, vulnerable populations, micronutrient deficiency
