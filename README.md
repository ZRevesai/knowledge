# Lightweight Interpretable Deep Learning Model for Nutrient Analysis

Implementation of the research paper: "Lightweight Interpretable Deep Learning Model for Nutrient Analysis in Mobile Health Applications" by Zvinodashe Revesai and Okuthe P. Kogeda.

## Overview

This project implements a lightweight, interpretable deep learning model for nutrient analysis that achieves:
- **97.1% food recognition accuracy** (98.0% with cross-validation)
- **7.2% mean absolute error** in nutrient estimation
- **11 MB model footprint**
- **150 ms inference time** on mobile devices
- Integrated interpretability with Grad-CAM, LIME, and CAVs

## Key Features

1. **Efficient Architecture**
   - MobileNetV3-based backbone
   - Depthwise separable convolutions
   - Shuffle Attention mechanisms
   - Squeeze-and-Excitation blocks

2. **Knowledge Distillation**
   - 62% model size reduction
   - 36% energy consumption reduction
   - 2.2% accuracy improvement

3. **Interpretability**
   - Grad-CAM visualizations
   - LIME explanations
   - Concept Activation Vectors (CAVs)

4. **Mobile Optimization**
   - 8-bit quantization
   - TensorFlow Lite conversion
   - On-device inference

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd knowledge

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
├── src/
│   ├── models/          # Model architectures
│   ├── data/            # Dataset preparation
│   ├── training/        # Training pipeline
│   ├── interpretability/ # Explainability features
│   ├── optimization/    # Mobile optimization
│   ├── evaluation/      # Metrics and evaluation
│   └── utils/           # Helper functions
├── scripts/             # Executable scripts
├── config/              # Configuration files
└── notebooks/           # Jupyter notebooks
```

## Quick Start

### 1. Prepare Dataset

```bash
python scripts/prepare_data.py --data_dir /path/to/food101 --output_dir ./data/processed
```

### 2. Train Model

```bash
python scripts/train_model.py --config config/config.yaml
```

### 3. Optimize for Mobile

```bash
python scripts/optimize_mobile.py --model_path ./checkpoints/best_model.pth --output_dir ./mobile_models
```

### 4. Evaluate

```bash
python scripts/evaluate_model.py --model_path ./checkpoints/best_model.pth --test_data ./data/processed/test
```

## Usage Example

```python
from src.models.nutrient_model import NutrientAnalysisModel
from src.interpretability.gradcam import GradCAM
import torch
from PIL import Image

# Load model
model = NutrientAnalysisModel.from_pretrained('checkpoints/best_model.pth')
model.eval()

# Load and preprocess image
image = Image.open('food_image.jpg')
input_tensor = preprocess(image)

# Inference
with torch.no_grad():
    food_pred, portion, nutrients = model(input_tensor)

# Generate interpretability visualizations
gradcam = GradCAM(model)
heatmap = gradcam.generate_heatmap(input_tensor, target_class=food_pred.argmax())
```

## Results

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 97.1% |
| Top-5 Accuracy | 98.0% |
| MAE (Nutrients) | 7.2% |
| Model Size | 11.0 MB |
| Inference Time (Lab) | 150 ms |
| Inference Time (Real) | 240-310 ms |
| Energy Consumption | 180 mJ |

### Food Security Categories

| Category | Accuracy | MAE |
|----------|----------|-----|
| Staple Foods | 94.1% | 6.8% |
| Affordable Proteins | 93.2% | 7.0% |
| Accessible Produce | 92.8% | 7.1% |

## Citation

```bibtex
@article{revesai2025lightweight,
  title={Lightweight Interpretable Deep Learning Model for Nutrient Analysis in Mobile Health Applications},
  author={Revesai, Zvinodashe and Kogeda, Okuthe P.},
  journal={Digital},
  volume={5},
  number={2},
  pages={23},
  year={2025},
  publisher={MDPI}
}
```

## License

This project is licensed under the Creative Commons Attribution (CC BY) license.

## Authors

- Zvinodashe Revesai (224195689@stu.ukzn.ac.za)
- Okuthe P. Kogeda (kogedao@ukzn.ac.za)

University of KwaZulu-Natal, School of Mathematics, Statistics and Computer Science
