#!/usr/bin/env python3
"""
Quick Installation Test for SIDRM

This script verifies that the SIDRM implementation is correctly installed
and all components are working.

Usage:
    python test_sidrm_installation.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("SIDRM Installation Test")
print("=" * 70)

# Test 1: Import core modules
print("\n[1/8] Testing core imports...")
try:
    import torch
    import numpy as np
    import pandas as pd
    import sklearn
    print("✓ Core dependencies OK")
except ImportError as e:
    print(f"✗ Missing core dependency: {e}")
    sys.exit(1)

# Test 2: Import SIDRM model
print("\n[2/8] Testing SIDRM model import...")
try:
    from src.models.sidrm import SIDRM, PopulationSpecificEncoder, CrossPopulationIntegration
    print("✓ SIDRM model imports OK")
except ImportError as e:
    print(f"✗ Failed to import SIDRM: {e}")
    sys.exit(1)

# Test 3: Import mobile models
print("\n[3/8] Testing mobile models import...")
try:
    from src.models.sidrm_mobile import SIDRMMobileOptimized, SIDRMEdgeOnly
    print("✓ Mobile models imports OK")
except ImportError as e:
    print(f"✗ Failed to import mobile models: {e}")
    sys.exit(1)

# Test 4: Import training components
print("\n[4/8] Testing training components...")
try:
    from src.training.sidrm_trainer import SIDRMTrainer
    from src.training.sidrm_losses import SIDRMMultiObjectiveLoss
    print("✓ Training components OK")
except ImportError as e:
    print(f"✗ Failed to import training components: {e}")
    sys.exit(1)

# Test 5: Import data processing
print("\n[5/8] Testing data processing...")
try:
    from src.data.nhanes_dataset import NHANESPreprocessor, create_nhanes_dataloaders
    print("✓ Data processing components OK")
except ImportError as e:
    print(f"✗ Failed to import data components: {e}")
    sys.exit(1)

# Test 6: Import evaluation components
print("\n[6/8] Testing evaluation components...")
try:
    from src.evaluation.sidrm_metrics import SIDRMEvaluator
    from src.evaluation.sidrm_interpretability import SIDRMInterpretabilityEvaluator
    print("✓ Evaluation components OK")
except ImportError as e:
    print(f"✗ Failed to import evaluation components: {e}")
    sys.exit(1)

# Test 7: Create a model instance
print("\n[7/8] Creating test model instance...")
try:
    model = SIDRM(
        input_dim=105,
        num_populations=4,
        hidden_dim=64,  # Smaller for quick test
        num_layers=2,   # Fewer layers for quick test
        num_attention_heads=4,
        num_nutrients=20,  # Fewer nutrients for quick test
        dropout_rate=0.3
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created successfully")
    print(f"  Parameters: {total_params:,}")

except Exception as e:
    print(f"✗ Failed to create model: {e}")
    sys.exit(1)

# Test 8: Test forward pass
print("\n[8/8] Testing forward pass...")
try:
    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 105)
    population_indices = torch.randint(0, 4, (batch_size,))

    # Forward pass
    with torch.no_grad():
        outputs = model(
            x,
            population_indices=population_indices,
            return_attention=True,
            return_population_scores=True
        )

    # Verify outputs
    assert 'nutrient_recommendations' in outputs
    assert 'deficiency_probabilities' in outputs
    assert 'attention_weights' in outputs
    assert 'population_scores' in outputs

    # Check shapes
    assert outputs['nutrient_recommendations'].shape == (batch_size, 20)
    assert outputs['deficiency_probabilities'].shape == (batch_size, 20)
    assert len(outputs['attention_weights']) == 2  # num_layers

    print("✓ Forward pass successful")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {outputs['deficiency_probabilities'].shape}")
    print(f"  Attention layers: {len(outputs['attention_weights'])}")

except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Success!
print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED")
print("=" * 70)
print("\nSIDRM is correctly installed and ready to use!")
print("\nNext steps:")
print("  1. Run quick demo:  python demo_sidrm.py --quick")
print("  2. Train model:     python scripts/train_sidrm.py")
print("  3. See README:      cat SIDRM_README.md")
print("\n" + "=" * 70)
