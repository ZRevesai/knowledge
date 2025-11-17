# Quick Start Guide

Get started with the Nutrient Analysis Model in 5 minutes!

## Prerequisites

```bash
# Python 3.8 or higher
python --version

# CUDA (optional, for GPU training)
nvidia-smi
```

## Installation

```bash
# Clone repository
cd knowledge

# Install dependencies
pip install -r requirements.txt
```

## Quick Demo

### 1. Test Model Architecture

```python
from src.models.nutrient_model import NutrientAnalysisModel

# Create model
model = NutrientAnalysisModel(num_classes=500)

# Model info
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")  # ~3-4M parameters
print(f"Size: {total_params * 4 / (1024**2):.2f} MB")  # ~11 MB
```

### 2. Prepare Sample Data

```bash
# Create sample nutritional database
python scripts/prepare_data.py --output ./data/food101 --create_nutritional_db
```

### 3. Train Model (Quick Test)

```bash
# Note: Download Food-101 dataset first
# https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/

python scripts/train_model.py \
    --config config/config.yaml \
    --data_dir ./data/food101
```

### 4. Optimize for Mobile

```bash
python scripts/optimize_mobile.py \
    --model checkpoints/best_model.pth \
    --output mobile_models/ \
    --quantization static \
    --tflite
```

### 5. Evaluate Model

```bash
python scripts/evaluate_model.py \
    --model checkpoints/best_model.pth \
    --data_dir ./data/food101 \
    --split test
```

## Interactive Demo

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/demo.ipynb
```

## Expected Results

Based on the research paper:

- **Accuracy**: 97.1% (98.0% with cross-validation)
- **MAE**: 7.2% for nutrient estimation
- **Model Size**: 11.0 MB (after quantization: 9.4 MB)
- **Inference Time**: 150 ms (lab), 240-310 ms (real devices)
- **Energy**: 180 mJ per inference

### Food Security Categories

- Staple Foods: 94.1% accuracy
- Affordable Proteins: 93.2% accuracy
- Accessible Produce: 92.8% accuracy

## Common Issues

### CUDA Out of Memory
```bash
# Reduce batch size in config.yaml
training:
  batch_size: 16  # Instead of 32
```

### Dataset Not Found
```bash
# Ensure correct directory structure:
data/food101/
├── images/
│   ├── apple_pie/
│   └── ...
└── nutritional_database.json
```

### TFLite Conversion Fails
```bash
# Install dependencies:
pip install onnx onnx-tf tensorflow
```

## Next Steps

1. **Custom Dataset**: Modify `src/data/dataset.py` for your data
2. **Fine-tuning**: Adjust hyperparameters in `config/config.yaml`
3. **Deploy**: Use TFLite model for Android/iOS apps
4. **Interpretability**: Explore Grad-CAM, LIME, and CAVs

## Support

- Documentation: See `README.md`
- Issues: Check existing implementation
- Paper: Read the full research article

## Quick Reference

```bash
# Training
python scripts/train_model.py --config config/config.yaml

# Optimization
python scripts/optimize_mobile.py --model <path> --output <dir>

# Evaluation
python scripts/evaluate_model.py --model <path> --data_dir <dir>

# Data Prep
python scripts/prepare_data.py --output <dir>
```
