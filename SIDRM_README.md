# Smart Interpretable Dietary Recommender Model (SIDRM) for Vulnerable Populations

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

This repository contains the complete implementation of the **Smart Interpretable Dietary Recommender Model (SIDRM)**, a novel deep learning architecture for generating transparent, clinically meaningful dietary recommendations for vulnerable populations.

### Paper Reference
**"Smart Interpretable Dietary Recommender Model for Vulnerable Populations"**
*Zvinodashe Revesai and Okuthe P. Kogeda (2025)*
School of Mathematics, Statistics and Computer Science, University of KwaZulu-Natal

### Key Features

- **Population-Specific Processing**: Dedicated encoders for pregnant women, elderly individuals, children, and chronic disease patients
- **Cross-Population Integration**: Learns shared patterns across vulnerable groups
- **Multi-Head Attention**: Transparent feature attribution for clinical interpretability
- **SHAP-Based Explanations**: Quantitative interpretability metrics (SHAP stability: 0.91, Attention consistency: 0.93)
- **Mobile Deployment**: Optimized configurations for resource-constrained environments (81% size reduction)

### Performance Highlights

| Metric | SIDRM | Baseline Transformer |
|--------|-------|---------------------|
| **Overall Accuracy** | 91.0% | 87.0% |
| **F1-Score** | 0.89 | 0.85 |
| **Parameters** | 56.2M | 127.3M (64% reduction) |
| **Training Time** | 14.2h | 18.7h (24% faster) |
| **SHAP Stability** | 0.91 | 0.62 |
| **Attention Consistency** | 0.93 | N/A |

## Architecture

SIDRM implements a multi-branch transformer structure with five hierarchical layers:

```
Input (NHANES Features) →
├─ Population-Specific Encoders (P, E, C, CD) →
├─ Cross-Population Integration →
├─ 5× Transformer Blocks (8-head attention) →
└─ Interpretable Output Layer → Dietary Recommendations
```

### Mathematical Foundation

#### 1. Data Integration (Equation 1)
```
I(i,j,k,l) = Σᵘ Σᵛ K(u,v;i,j) · D(i+u-1, j+v-1, k+w-1, l)
```

#### 2. Population-Specific Processing (Equation 3)
```
O_P(i,j,k,c) = Σ K_P(u,v,w,c) · I(i+u-1, j+v-1, k+w-1, c)
```

#### 3. Cross-Population Integration (Equation 5)
```
O_C(i,j,k,n) = Σ K_C(l,n) · O_P(i,j,k,l)
```

#### 4. Multi-Head Attention (Equation 8)
```
Attention(Q,K,V) = softmax(Q·K^T / √d_k) · V
```

#### 5. Multi-Objective Loss (Equation 13)
```
L_total = 0.4·L_accuracy + 0.3·L_interpret + 0.2·L_clinical + 0.1·L_safety
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/ZRevesai/knowledge.git
cd knowledge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### 1. Data Preparation

```python
from src.data.nhanes_dataset import NHANESPreprocessor, create_nhanes_dataloaders

# Initialize preprocessor
preprocessor = NHANESPreprocessor(random_state=42)

# Generate/load NHANES data (2015-2018 cycles)
# For real data, replace with actual NHANES loading
data_splits = preprocessor.generate_synthetic_nhanes_data(
    n_samples=15560,  # As per paper
    train_ratio=0.7,
    val_ratio=0.15
)

# Create dataloaders
dataloaders = create_nhanes_dataloaders(
    data_splits,
    preprocessor,
    batch_size=32,
    num_workers=4
)
```

### 2. Model Creation

```python
from src.models.sidrm import SIDRM

# Full SIDRM model (as per paper specifications)
model = SIDRM(
    input_dim=105,  # NHANES feature dimension
    num_populations=4,  # P, E, C, CD
    population_names=['Pregnant', 'Elderly', 'Children', 'Chronic_Disease'],
    hidden_dim=128,
    num_layers=5,  # 5 hierarchical layers
    num_attention_heads=8,
    num_nutrients=50,
    dropout_rate=0.3,
    use_population_specific=True
)

print(model.get_model_summary())
```

### 3. Training

```python
from src.training.sidrm_trainer import SIDRMTrainer
from src.training.sidrm_losses import SIDRMMultiObjectiveLoss

# Multi-objective loss function (Equation 13)
criterion = SIDRMMultiObjectiveLoss(
    num_nutrients=50,
    alpha=0.4,  # Accuracy
    beta=0.3,   # Interpretability
    gamma=0.2,  # Clinical
    delta=0.1   # Safety
)

# AdamW optimizer (as per Section 3.5)
trainer = SIDRMTrainer(
    model=model,
    criterion=criterion,
    train_loader=dataloaders['train'],
    val_loader=dataloaders['val'],
    learning_rate=1e-4,
    weight_decay=1e-5,
    patience=15,  # LR reduction patience
    decay_factor=0.7,
    device='cuda'
)

# Train for 200 epochs
history = trainer.train(
    num_epochs=200,
    early_stopping=True,
    early_stopping_patience=30,
    save_best=True
)
```

### 4. Evaluation

```python
from src.evaluation.sidrm_metrics import SIDRMEvaluator

# Create evaluator
evaluator = SIDRMEvaluator(
    model=model,
    device='cuda',
    population_names=['Pregnant', 'Elderly', 'Children', 'Chronic_Disease'],
    nutrient_names=preprocessor.nutrient_names
)

# Comprehensive evaluation
results = evaluator.evaluate(
    dataloaders['test'],
    return_predictions=True,
    compute_per_nutrient=True,
    compute_per_population=True
)

# Print results
evaluator.print_evaluation_results(results, detailed=True)
```

### 5. Interpretability Analysis

```python
from src.evaluation.sidrm_interpretability import SIDRMInterpretabilityEvaluator

# Initialize interpretability evaluator
interp_evaluator = SIDRMInterpretabilityEvaluator(
    model=model,
    background_data=train_features,
    feature_names=preprocessor.feature_names,
    num_bootstrap_samples=100
)

# Evaluate interpretability (Equations 10-12)
interp_results = interp_evaluator.evaluate_interpretability(
    test_features,
    population_indices,
    compute_shap=True,
    shap_nsamples=100
)

print(f"SHAP Stability: {interp_results['shap_stability']:.4f}")
print(f"Attention Consistency: {interp_results['attention_consistency']:.4f}")
print(f"Overall Interpretability: {interp_results['overall_interpretability']:.4f}")
```

## Mobile Deployment

### Mobile-Optimized Configuration

```python
from src.models.sidrm_mobile import SIDRMMobileOptimized

# 81% size reduction: 174.9 MB → 32.8 MB
mobile_model = SIDRMMobileOptimized(
    input_dim=105,
    num_populations=4,
    hidden_dim=64,  # Reduced from 128
    num_layers=3,   # Reduced from 5
    num_attention_heads=4,  # Reduced from 8
    num_nutrients=50,
    dropout_rate=0.2
)

# Performance: 88.7% accuracy, 0.83 interpretability
size_info = mobile_model.get_model_size()
print(f"Model size: {size_info['size_mb']:.2f} MB")
```

### Edge-Only Configuration

```python
from src.models.sidrm_mobile import SIDRMEdgeOnly

# Ultra-compact: 28.1 MB
edge_model = SIDRMEdgeOnly(
    input_dim=105,
    hidden_dim=48,
    num_layers=2,
    num_nutrients=50
)

# Performance: 87.2% accuracy, 0.79 interpretability
# Inference time: 6.4ms
```

## Datasets

### NHANES (2015-2018)

The model is trained on the National Health and Nutrition Examination Survey dataset:

- **Total Participants**: 15,560
- **Vulnerable Populations**:
  - Pregnant Women: 15% (2,334 samples)
  - Elderly (65+): 30% (4,668 samples)
  - Children (2-17): 25% (3,890 samples)
  - Chronic Disease: 30% (4,668 samples)

#### Data Splits
- Training: 70% (10,892 samples)
- Validation: 15% (2,334 samples)
- Testing: 15% (2,334 samples)

#### Features (105 dimensions)
- **Demographics**: Age, gender, race/ethnicity, population category
- **Anthropometrics**: Height, weight, BMI
- **Clinical Biomarkers**: Glucose, cholesterol, HDL, LDL, triglycerides, HbA1c, blood pressure
- **Serum Nutrients**: B12, folate, iron, ferritin, calcium, vitamin D
- **Dietary Intake**: Energy, macronutrients, 20 micronutrients (24-hour recall)
- **Supplements**: Multivitamin, vitamin D, calcium, iron usage
- **Lifestyle**: Physical activity, smoking, alcohol consumption
- **Chronic Conditions**: Diabetes, hypertension, heart disease, kidney disease

## Results Reproduction

### Paper Results (Tables 1-6)

#### Table 1: Computational Requirements
```python
# Run comparative analysis
python scripts/compare_models.py --config configs/sidrm_config.yaml

# Expected results:
# SIDRM: 56.2M params, 14.2h training, 18.3ms inference
# Baseline: 127.3M params, 18.7h training, 24.6ms inference
```

#### Table 2: Population-Specific Performance
```python
# Evaluate per-population metrics
python scripts/evaluate_sidrm.py --checkpoint checkpoints/best_model.pth --per-population

# Expected accuracy by population:
# Pregnant: 91.2%, Elderly: 89.8%, Children: 92.4%, Chronic Disease: 90.6%
```

#### Table 3: Model Comparison
```python
# Compare with state-of-the-art
python scripts/compare_models.py --baselines all

# Expected results:
# SIDRM: 91.0% accuracy, 0.89 F1-score, 0.89 interpretability
# Transformer: 87.0% accuracy, 0.85 F1-score, 0.41 interpretability
```

#### Table 4: Mobile Deployment
```python
# Test mobile configurations
python scripts/evaluate_mobile.py

# Expected sizes and accuracy:
# Full SIDRM: 174.9 MB, 91.0% accuracy
# Mobile-Optimized: 32.8 MB, 88.7% accuracy
# Edge-Only: 28.1 MB, 87.2% accuracy
```

## Project Structure

```
knowledge/
├── src/
│   ├── models/
│   │   ├── sidrm.py                    # Main SIDRM model
│   │   ├── sidrm_mobile.py             # Mobile-optimized versions
│   │   ├── layers.py                   # Custom layers
│   │   └── baselines.py                # Baseline models
│   ├── data/
│   │   ├── nhanes_dataset.py           # NHANES preprocessing & loading
│   │   ├── preprocessing.py            # Data preprocessing
│   │   └── dataset.py                  # PyTorch datasets
│   ├── training/
│   │   ├── sidrm_trainer.py            # SIDRM training pipeline
│   │   ├── sidrm_losses.py             # Multi-objective loss
│   │   ├── losses.py                   # Other loss functions
│   │   └── trainer.py                  # General trainer
│   ├── evaluation/
│   │   ├── sidrm_metrics.py            # Performance metrics
│   │   ├── sidrm_interpretability.py   # Interpretability evaluation
│   │   ├── metrics.py                  # General metrics
│   │   └── interpretability.py         # Interpretability tools
│   ├── visualization/
│   │   ├── plots.py                    # Performance plots
│   │   └── attention_viz.py            # Attention visualization
│   └── knowledge/                      # KGNN components (separate)
├── scripts/
│   ├── train_sidrm.py                  # Training script
│   ├── evaluate_sidrm.py               # Evaluation script
│   ├── compare_models.py               # Model comparison
│   └── evaluate_mobile.py              # Mobile evaluation
├── configs/
│   ├── sidrm_config.yaml               # SIDRM configuration
│   └── sidrm_mobile_config.yaml        # Mobile configuration
├── notebooks/
│   ├── sidrm_demo.ipynb                # Demo notebook
│   └── interpretability_analysis.ipynb # Interpretability analysis
├── data/                               # Data directory
├── checkpoints/                        # Model checkpoints
├── results/                            # Results directory
├── requirements.txt                    # Dependencies
├── setup.py                            # Package setup
├── README.md                           # Main README (KGNN)
├── SIDRM_README.md                     # This file
└── LICENSE                             # MIT License
```

## Configuration

Example `configs/sidrm_config.yaml`:

```yaml
model:
  input_dim: 105
  num_populations: 4
  hidden_dim: 128
  num_layers: 5
  num_attention_heads: 8
  num_nutrients: 50
  dropout_rate: 0.3
  use_population_specific: true

training:
  batch_size: 32
  num_epochs: 200
  learning_rate: 1.0e-4
  weight_decay: 1.0e-5
  patience: 15
  decay_factor: 0.7
  gradient_clip: 1.0
  early_stopping_patience: 30

loss:
  alpha: 0.4  # Accuracy weight
  beta: 0.3   # Interpretability weight
  gamma: 0.2  # Clinical weight
  delta: 0.1  # Safety weight

data:
  n_samples: 15560
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  num_workers: 4
```

## Command-Line Scripts

### Training

```bash
# Train full SIDRM model
python scripts/train_sidrm.py --config configs/sidrm_config.yaml --device cuda

# Train mobile-optimized model
python scripts/train_sidrm.py --config configs/sidrm_mobile_config.yaml --mobile

# Resume from checkpoint
python scripts/train_sidrm.py --resume checkpoints/checkpoint_epoch_50.pth
```

### Evaluation

```bash
# Evaluate on test set
python scripts/evaluate_sidrm.py --checkpoint checkpoints/best_model.pth --test

# Evaluate interpretability
python scripts/evaluate_sidrm.py --checkpoint checkpoints/best_model.pth --interpretability

# Generate predictions
python scripts/evaluate_sidrm.py --checkpoint checkpoints/best_model.pth --predict --output predictions.csv
```

### Model Comparison

```bash
# Compare SIDRM with baselines
python scripts/compare_models.py --models sidrm transformer cnn --config configs/sidrm_config.yaml

# Mobile performance comparison
python scripts/evaluate_mobile.py --checkpoint checkpoints/best_model.pth
```

## Advanced Usage

### Custom Population Processing

```python
# Add custom population encoder
from src.models.sidrm import PopulationSpecificEncoder

custom_encoder = PopulationSpecificEncoder(
    input_dim=128,
    hidden_dim=128,
    filter_size=3,
    population_name="Custom_Population",
    dropout_rate=0.3
)
```

### Custom Loss Components

```python
from src.training.sidrm_losses import (
    AccuracyLoss, InterpretabilityLoss,
    ClinicalLoss, SafetyLoss
)

# Create custom multi-objective loss
custom_criterion = SIDRMMultiObjectiveLoss(
    num_nutrients=50,
    alpha=0.5,  # Prioritize accuracy
    beta=0.2,
    gamma=0.2,
    delta=0.1,
    guideline_bounds={...},  # Custom clinical guidelines
    upper_limits={...}        # Custom safety limits
)
```

### Visualization

```python
from src.visualization.plots import plot_training_history
from src.visualization.attention_viz import plot_attention_heatmap

# Plot training history
plot_training_history(history, save_path='figures/training.png')

# Visualize attention weights
plot_attention_heatmap(
    attention_weights,
    feature_names=preprocessor.feature_names,
    save_path='figures/attention.png'
)
```

## Clinical Deployment

### Safety Considerations

1. **Model Validation**: All recommendations should be validated by registered dietitians
2. **Toxicity Prevention**: Safety loss component prevents excessive nutrient recommendations
3. **Population Constraints**: Clinical loss enforces population-specific guidelines
4. **Interpretability**: SHAP values and attention weights provide transparent explanations

### Recommended Workflow

```python
# 1. Load validated model
model = SIDRM(...)
model.load_state_dict(torch.load('checkpoints/validated_model.pth'))
model.eval()

# 2. Prepare patient data
patient_features = preprocess_patient_data(patient_nhanes_profile)

# 3. Generate recommendations
with torch.no_grad():
    outputs = model(patient_features, population_indices=patient_population)
    recommendations = outputs['nutrient_recommendations']
    deficiency_probs = outputs['deficiency_probabilities']

# 4. Get interpretability scores
interp_scores = model.get_interpretability_scores(patient_features)

# 5. Generate clinical report
clinical_report = generate_report(
    recommendations,
    deficiency_probs,
    interp_scores,
    patient_demographics
)

# 6. Review by dietitian (required)
```

## Citation

If you use SIDRM in your research, please cite:

```bibtex
@article{revesai2025sidrm,
  title={Smart Interpretable Dietary Recommender Model for Vulnerable Populations},
  author={Revesai, Zvinodashe and Kogeda, Okuthe P.},
  journal={[Conference/Journal]},
  year={2025},
  institution={University of KwaZulu-Natal}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NHANES dataset (2015-2018) from the Centers for Disease Control and Prevention (CDC)
- National Institutes of Health (NIH) for dietary reference intakes
- Vulnerable population health research collaborators
- University of KwaZulu-Natal for research support

## Contact

For questions, issues, or collaborations:

- **Zvinodashe Revesai**: 224195689@stu.ukzn.ac.za
- **Dr. Okuthe P. Kogeda**: kogedao@ukzn.ac.za
- **GitHub Issues**: [https://github.com/ZRevesai/knowledge/issues](https://github.com/ZRevesai/knowledge/issues)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**Keywords**: Deep Learning, Interpretable AI, Nutrition Recommendation, Vulnerable Populations, Healthcare AI, NHANES, SHAP, Attention Mechanisms, Mobile Health, Dietary Assessment
