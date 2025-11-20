# SIDRM Quick Start Guide

Complete quick start guide for the Smart Interpretable Dietary Recommender Model (SIDRM).

## Installation

```bash
# Clone the repository (if not already done)
git clone https://github.com/ZRevesai/knowledge.git
cd knowledge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Verify Installation

```bash
# Check code structure (no dependencies needed)
python verify_code_structure.py

# Test installation (requires dependencies)
python test_sidrm_installation.py
```

## Quick Demo (Recommended First Step)

```bash
# Run quick demo (500 samples, 3 epochs, ~2 minutes)
python demo_sidrm.py --quick

# Standard demo (2,000 samples, 10 epochs)
python demo_sidrm.py

# Full demo (15,560 samples, 200 epochs - as per paper)
python demo_sidrm.py --full
```

## Training SIDRM

### Basic Training

```bash
# Train with default configuration
python scripts/train_sidrm.py

# Train with custom config
python scripts/train_sidrm.py --config configs/sidrm_config.yaml

# Train on GPU
python scripts/train_sidrm.py --device cuda

# Train mobile-optimized model
python scripts/train_sidrm.py --mobile --epochs 100
```

### Advanced Training

```bash
# Resume from checkpoint
python scripts/train_sidrm.py --resume checkpoints/sidrm/checkpoint_epoch_50.pth

# Train edge-only model
python scripts/train_sidrm.py --edge --epochs 50

# Custom number of epochs
python scripts/train_sidrm.py --epochs 200
```

## Evaluation

### Basic Evaluation

```bash
# Evaluate on test set
python scripts/evaluate_sidrm.py --checkpoint checkpoints/sidrm/best_model.pth --test

# Evaluate per-population metrics
python scripts/evaluate_sidrm.py --checkpoint checkpoints/sidrm/best_model.pth --per-population

# Evaluate per-nutrient metrics
python scripts/evaluate_sidrm.py --checkpoint checkpoints/sidrm/best_model.pth --per-nutrient
```

### Interpretability Evaluation

```bash
# Basic interpretability (attention-based, fast)
python scripts/evaluate_sidrm.py --checkpoint checkpoints/sidrm/best_model.pth --interpretability

# Full interpretability with SHAP (slow but comprehensive)
python scripts/evaluate_sidrm.py --checkpoint checkpoints/sidrm/best_model.pth --interpretability --compute-shap

# Save results to file
python scripts/evaluate_sidrm.py --checkpoint checkpoints/sidrm/best_model.pth --test --output results/evaluation.json
```

## Python API Usage

### 1. Data Preparation

```python
from src.data.nhanes_dataset import NHANESPreprocessor, create_nhanes_dataloaders

# Initialize preprocessor
preprocessor = NHANESPreprocessor(random_state=42)

# Generate/load data
data_splits = preprocessor.generate_synthetic_nhanes_data(
    n_samples=15560,
    train_ratio=0.7,
    val_ratio=0.15
)

# Create dataloaders
dataloaders = create_nhanes_dataloaders(
    data_splits,
    preprocessor,
    batch_size=32
)
```

### 2. Model Creation

```python
from src.models.sidrm import SIDRM

# Create full SIDRM model
model = SIDRM(
    input_dim=105,
    num_populations=4,
    hidden_dim=128,
    num_layers=5,
    num_attention_heads=8,
    num_nutrients=50,
    dropout_rate=0.3
)

# Create mobile model
from src.models.sidrm_mobile import SIDRMMobileOptimized

mobile_model = SIDRMMobileOptimized(
    input_dim=105,
    num_populations=4,
    hidden_dim=64,
    num_layers=3,
    num_nutrients=50
)
```

### 3. Training

```python
from src.training.sidrm_trainer import SIDRMTrainer
from src.training.sidrm_losses import SIDRMMultiObjectiveLoss

# Create loss function
criterion = SIDRMMultiObjectiveLoss(
    num_nutrients=50,
    alpha=0.4,  # Accuracy
    beta=0.3,   # Interpretability
    gamma=0.2,  # Clinical
    delta=0.1   # Safety
)

# Create trainer
trainer = SIDRMTrainer(
    model=model,
    criterion=criterion,
    train_loader=dataloaders['train'],
    val_loader=dataloaders['val'],
    learning_rate=1e-4,
    weight_decay=1e-5,
    device='cuda'
)

# Train
history = trainer.train(
    num_epochs=200,
    early_stopping=True,
    save_best=True
)
```

### 4. Evaluation

```python
from src.evaluation.sidrm_metrics import SIDRMEvaluator

# Create evaluator
evaluator = SIDRMEvaluator(
    model=model,
    device='cuda'
)

# Evaluate
results = evaluator.evaluate(
    dataloaders['test'],
    compute_per_nutrient=True,
    compute_per_population=True
)

# Print results
evaluator.print_evaluation_results(results)
```

### 5. Interpretability Analysis

```python
from src.evaluation.sidrm_interpretability import SIDRMInterpretabilityEvaluator

# Get background data
train_batch = next(iter(dataloaders['train']))
background_data = train_batch['features'][:100]

# Create interpretability evaluator
interp_evaluator = SIDRMInterpretabilityEvaluator(
    model=model,
    background_data=background_data,
    feature_names=preprocessor.feature_names
)

# Evaluate interpretability
test_batch = next(iter(dataloaders['test']))
interp_results = interp_evaluator.evaluate_interpretability(
    test_batch['features'][:50],
    test_batch['population'][:50],
    compute_shap=True
)

print(f"SHAP Stability: {interp_results['shap_stability']:.4f}")
print(f"Attention Consistency: {interp_results['attention_consistency']:.4f}")
```

## Configuration

Edit `configs/sidrm_config.yaml` to customize:

```yaml
model:
  hidden_dim: 128      # Change model size
  num_layers: 5        # Change depth
  num_attention_heads: 8

training:
  learning_rate: 1.0e-4
  batch_size: 32
  num_epochs: 200

loss:
  alpha: 0.4  # Accuracy weight
  beta: 0.3   # Interpretability weight
  gamma: 0.2  # Clinical weight
  delta: 0.1  # Safety weight
```

## Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Open demo notebook
# notebooks/sidrm_quickstart.ipynb
```

## Common Commands

```bash
# Quick test (fastest)
python demo_sidrm.py --quick --no-display

# Train and evaluate
python scripts/train_sidrm.py --config configs/sidrm_config.yaml
python scripts/evaluate_sidrm.py --checkpoint checkpoints/sidrm/best_model.pth --test --per-population

# Mobile deployment test
python demo_sidrm.py --mobile --skip-training

# Get model summary
python -c "from src.models.sidrm import SIDRM; m = SIDRM(105, 4, 128, 5, 8, 50); print(m.get_model_summary())"
```

## Expected Results

Based on the paper (Revesai & Kogeda, 2025):

| Metric | Target Value |
|--------|--------------|
| Overall Accuracy | 91.0% |
| F1-Score | 0.89 |
| SHAP Stability | 0.91 |
| Attention Consistency | 0.93 |
| Model Parameters | 56.2M |
| Training Time | 14.2h (full dataset) |

### Per-Population Accuracy
- Pregnant Women: 91.2%
- Elderly: 89.8%
- Children: 92.4%
- Chronic Disease: 90.6%

### Mobile Deployment
- Full SIDRM: 174.9 MB
- Mobile-Optimized: 32.8 MB (81% reduction, 88.7% accuracy)
- Edge-Only: 28.1 MB (87.2% accuracy)

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python scripts/train_sidrm.py --config configs/sidrm_config.yaml
# Then edit configs/sidrm_config.yaml: batch_size: 16

# Use mobile model
python demo_sidrm.py --mobile
```

### Slow Training
```bash
# Use GPU
python scripts/train_sidrm.py --device cuda

# Reduce epochs for testing
python demo_sidrm.py --quick

# Use mobile model
python scripts/train_sidrm.py --mobile --epochs 50
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Verify installation
python verify_code_structure.py
python test_sidrm_installation.py
```

## Next Steps

1. **Quick Test**: `python demo_sidrm.py --quick`
2. **Read Documentation**: `cat SIDRM_README.md`
3. **Train Model**: `python scripts/train_sidrm.py`
4. **Evaluate**: `python scripts/evaluate_sidrm.py --checkpoint <path>`
5. **Deploy Mobile**: Test mobile configurations

## Resources

- **Main Documentation**: `SIDRM_README.md`
- **Paper Reference**: Revesai & Kogeda (2025)
- **Configuration**: `configs/sidrm_config.yaml`
- **Demo Notebook**: `notebooks/sidrm_quickstart.ipynb`
- **Examples**: `demo_sidrm.py`

## Support

For issues or questions:
- Check `SIDRM_README.md`
- Run `python verify_code_structure.py`
- Review `notebooks/sidrm_quickstart.ipynb`

---

**SIDRM**: Smart Interpretable Dietary Recommender Model for Vulnerable Populations
**University of KwaZulu-Natal** | School of Mathematics, Statistics and Computer Science
