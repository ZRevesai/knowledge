#!/usr/bin/env python3
"""
End-to-End SIDRM Demo Script

This script demonstrates the complete SIDRM pipeline:
1. Data generation/loading
2. Model creation
3. Training
4. Evaluation
5. Interpretability analysis
6. Mobile deployment

Usage:
    python demo_sidrm.py --quick      # Quick demo (small dataset, few epochs)
    python demo_sidrm.py --full       # Full demo (realistic settings)
    python demo_sidrm.py --mobile     # Test mobile models
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.sidrm import SIDRM
from src.models.sidrm_mobile import SIDRMMobileOptimized, SIDRMEdgeOnly
from src.data.nhanes_dataset import NHANESPreprocessor, create_nhanes_dataloaders
from src.training.sidrm_trainer import SIDRMTrainer
from src.training.sidrm_losses import SIDRMMultiObjectiveLoss
from src.evaluation.sidrm_metrics import SIDRMEvaluator


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def demo_data_preparation(args):
    """Demonstrate data preparation."""
    print_section("STEP 1: DATA PREPARATION")

    preprocessor = NHANESPreprocessor(random_state=42)

    if args.quick:
        n_samples = 500
        print("Quick mode: Using 500 samples")
    elif args.full:
        n_samples = 15560  # Full dataset as per paper
        print("Full mode: Using 15,560 samples (NHANES 2015-2018)")
    else:
        n_samples = 2000
        print("Standard mode: Using 2,000 samples")

    data_splits = preprocessor.generate_synthetic_nhanes_data(
        n_samples=n_samples,
        train_ratio=0.7,
        val_ratio=0.15
    )

    dataloaders = create_nhanes_dataloaders(
        data_splits,
        preprocessor,
        batch_size=32,
        num_workers=0
    )

    print(f"\n✓ Data splits created:")
    print(f"  Training:   {len(dataloaders['train'].dataset):,} samples")
    print(f"  Validation: {len(dataloaders['val'].dataset):,} samples")
    print(f"  Test:       {len(dataloaders['test'].dataset):,} samples")

    # Show data sample
    batch = next(iter(dataloaders['train']))
    print(f"\n✓ Data dimensions:")
    print(f"  Input features: {batch['features'].shape[1]}")
    print(f"  Nutrients:      {batch['labels'].shape[1]}")
    print(f"  Populations:    4 (Pregnant, Elderly, Children, Chronic Disease)")

    return preprocessor, dataloaders, batch


def demo_model_creation(batch, args):
    """Demonstrate model creation."""
    print_section("STEP 2: MODEL CREATION")

    input_dim = batch['features'].shape[1]
    num_nutrients = batch['labels'].shape[1]

    if args.mobile:
        print("Creating Mobile-Optimized SIDRM...")
        model = SIDRMMobileOptimized(
            input_dim=input_dim,
            num_populations=4,
            hidden_dim=64,
            num_layers=3,
            num_attention_heads=4,
            num_nutrients=num_nutrients,
            dropout_rate=0.2
        )
        model_type = "mobile"
    else:
        print("Creating Full SIDRM...")
        model = SIDRM(
            input_dim=input_dim,
            num_populations=4,
            population_names=['Pregnant', 'Elderly', 'Children', 'Chronic_Disease'],
            hidden_dim=128,
            num_layers=5,
            num_attention_heads=8,
            num_nutrients=num_nutrients,
            dropout_rate=0.3,
            use_population_specific=True
        )
        model_type = "full"

    print("\n" + model.get_model_summary())

    # Test forward pass
    print("\n✓ Testing forward pass...")
    x = batch['features'][:4]
    pop = batch['population'][:4]

    with torch.no_grad():
        outputs = model(x, pop, return_attention=True)

    print(f"  Input shape:        {x.shape}")
    print(f"  Output shape:       {outputs['deficiency_probabilities'].shape}")
    if 'attention_weights' in outputs:
        print(f"  Attention layers:   {len(outputs['attention_weights'])}")

    return model, model_type, input_dim, num_nutrients


def demo_training(model, dataloaders, num_nutrients, args):
    """Demonstrate model training."""
    print_section("STEP 3: TRAINING")

    # Create loss function
    criterion = SIDRMMultiObjectiveLoss(
        num_nutrients=num_nutrients,
        alpha=0.4,  # Accuracy
        beta=0.3,   # Interpretability
        gamma=0.2,  # Clinical
        delta=0.1   # Safety
    )

    print("\n✓ Multi-Objective Loss Function:")
    print(f"  α (Accuracy):        0.4")
    print(f"  β (Interpretability): 0.3")
    print(f"  γ (Clinical):        0.2")
    print(f"  δ (Safety):          0.1")

    # Create trainer
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'

    trainer = SIDRMTrainer(
        model=model,
        criterion=criterion,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        learning_rate=1e-4,
        weight_decay=1e-5,
        device=device,
        patience=15,
        decay_factor=0.7,
        save_dir='demo_checkpoints'
    )

    print(f"\n✓ Trainer Configuration:")
    print(f"  Device:        {device}")
    print(f"  Optimizer:     AdamW")
    print(f"  Learning rate: 1e-4")
    print(f"  Weight decay:  1e-5")

    # Train
    if args.quick:
        num_epochs = 3
        print(f"\nQuick mode: Training for {num_epochs} epochs...")
    elif args.full:
        num_epochs = 200
        print(f"\nFull mode: Training for {num_epochs} epochs...")
    else:
        num_epochs = 10
        print(f"\nStandard mode: Training for {num_epochs} epochs...")

    start_time = time.time()

    history = trainer.train(
        num_epochs=num_epochs,
        early_stopping=True,
        early_stopping_patience=5 if args.quick else 30,
        save_best=True,
        verbose=not args.quiet
    )

    training_time = time.time() - start_time

    print(f"\n✓ Training completed in {training_time:.2f}s")
    print(f"  Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"  Best epoch:           {trainer.best_epoch}")
    print(f"  Final train accuracy: {history['train_accuracy'][-1]:.4f}")
    print(f"  Final val accuracy:   {history['val_accuracy'][-1]:.4f}")

    return trainer, history


def demo_evaluation(model, dataloaders, preprocessor, args):
    """Demonstrate model evaluation."""
    print_section("STEP 4: EVALUATION")

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'

    evaluator = SIDRMEvaluator(
        model=model,
        device=device,
        population_names=['Pregnant', 'Elderly', 'Children', 'Chronic_Disease'],
        nutrient_names=preprocessor.nutrient_names
    )

    print("\n✓ Evaluating on test set...")

    results = evaluator.evaluate(
        dataloaders['test'],
        return_predictions=True,
        compute_per_nutrient=True,
        compute_per_population=True
    )

    print("\n" + "-" * 80)
    print("PERFORMANCE METRICS:")
    print("-" * 80)
    print(f"Overall Accuracy:  {results['accuracy']*100:.2f}%")
    print(f"Precision:         {results['precision']*100:.2f}%")
    print(f"Recall:            {results['recall']*100:.2f}%")
    print(f"F1-Score:          {results['f1_score']:.4f}")
    if 'auc_roc' in results:
        print(f"AUC-ROC:           {results['auc_roc']:.4f}")

    # Per-population results
    if 'per_population' in results:
        print("\n" + "-" * 80)
        print("PER-POPULATION ACCURACY:")
        print("-" * 80)
        for pop_name, metrics in results['per_population'].items():
            print(f"{pop_name:20s}: {metrics['accuracy']*100:6.2f}%  "
                  f"(F1: {metrics['f1_score']:.4f}, n={metrics['num_samples']})")

    # Top nutrients
    if 'per_nutrient' in results:
        print("\n" + "-" * 80)
        print("TOP 5 NUTRIENTS (by F1-Score):")
        print("-" * 80)

        nutrient_f1 = {name: metrics['f1_score']
                      for name, metrics in results['per_nutrient'].items()}
        sorted_nutrients = sorted(nutrient_f1.items(), key=lambda x: x[1], reverse=True)

        for name, f1 in sorted_nutrients[:5]:
            acc = results['per_nutrient'][name]['accuracy']
            print(f"{name:30s}: F1={f1:.4f}, Acc={acc*100:.2f}%")

    return results


def demo_mobile_models(input_dim, num_nutrients):
    """Demonstrate mobile-optimized models."""
    print_section("STEP 5: MOBILE DEPLOYMENT")

    print("\n✓ Creating mobile-optimized models...\n")

    # Full SIDRM
    full_model = SIDRM(
        input_dim=input_dim,
        num_populations=4,
        hidden_dim=128,
        num_layers=5,
        num_attention_heads=8,
        num_nutrients=num_nutrients
    )

    full_params = sum(p.numel() for p in full_model.parameters())
    full_size = sum(p.numel() * p.element_size() for p in full_model.parameters()) / (1024**2)

    print(f"Full SIDRM:")
    print(f"  Parameters: {full_params:,}")
    print(f"  Size:       {full_size:.2f} MB")
    print(f"  Target:     174.9 MB (paper)\n")

    # Mobile-Optimized
    mobile_model = SIDRMMobileOptimized(
        input_dim=input_dim,
        num_populations=4,
        hidden_dim=64,
        num_layers=3,
        num_attention_heads=4,
        num_nutrients=num_nutrients
    )

    mobile_size = mobile_model.get_model_size()
    size_reduction = (1 - mobile_size['size_mb'] / full_size) * 100

    print(f"Mobile-Optimized SIDRM:")
    print(f"  Parameters: {mobile_size['parameters']:,}")
    print(f"  Size:       {mobile_size['size_mb']:.2f} MB")
    print(f"  Reduction:  {size_reduction:.1f}%")
    print(f"  Target:     32.8 MB, 81% reduction (paper)\n")

    # Edge-Only
    edge_model = SIDRMEdgeOnly(
        input_dim=input_dim,
        hidden_dim=48,
        num_layers=2,
        num_nutrients=num_nutrients
    )

    edge_size = edge_model.get_model_size()
    edge_reduction = (1 - edge_size['size_mb'] / full_size) * 100

    print(f"Edge-Only SIDRM:")
    print(f"  Parameters: {edge_size['parameters']:,}")
    print(f"  Size:       {edge_size['size_mb']:.2f} MB")
    print(f"  Reduction:  {edge_reduction:.1f}%")
    print(f"  Target:     28.1 MB (paper)\n")

    # Inference time test
    print("✓ Testing inference time...")
    x = torch.randn(16, input_dim)

    with torch.no_grad():
        # Full model
        start = time.time()
        for _ in range(100):
            _ = full_model(x, torch.randint(0, 4, (16,)))
        full_time = (time.time() - start) / 100 * 1000

        # Mobile model
        start = time.time()
        for _ in range(100):
            _ = mobile_model(x, torch.randint(0, 4, (16,)))
        mobile_time = (time.time() - start) / 100 * 1000

        # Edge model
        start = time.time()
        for _ in range(100):
            _ = edge_model(x)
        edge_time = (time.time() - start) / 100 * 1000

    print(f"\nInference Time (batch=16):")
    print(f"  Full:   {full_time:.2f} ms  (Target: 18.3 ms)")
    print(f"  Mobile: {mobile_time:.2f} ms  (Target: 8.9 ms)")
    print(f"  Edge:   {edge_time:.2f} ms  (Target: 6.4 ms)")


def visualize_results(history, results):
    """Visualize training and evaluation results."""
    print_section("STEP 6: VISUALIZATION")

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Training loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history['train_loss'], label='Train', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Training accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history['train_accuracy'], label='Train', linewidth=2)
    ax2.plot(history['val_accuracy'], label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Accuracy', fontsize=10)
    ax2.set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Learning rate
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(history['learning_rates'], linewidth=2, color='green')
    ax3.set_xlabel('Epoch', fontsize=10)
    ax3.set_ylabel('Learning Rate', fontsize=10)
    ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    # Per-population accuracy
    if 'per_population' in results:
        ax4 = fig.add_subplot(gs[1, 0])
        pop_names = list(results['per_population'].keys())
        accuracies = [results['per_population'][pop]['accuracy'] * 100 for pop in pop_names]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        bars = ax4.bar(pop_names, accuracies, color=colors)
        ax4.set_ylabel('Accuracy (%)', fontsize=10)
        ax4.set_title('Per-Population Accuracy', fontsize=12, fontweight='bold')
        ax4.set_ylim([0, 100])
        ax4.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    # Metrics comparison
    ax5 = fig.add_subplot(gs[1, 1])
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        results['accuracy'] * 100,
        results['precision'] * 100,
        results['recall'] * 100,
        results['f1_score'] * 100
    ]
    bars = ax5.barh(metrics, values, color='#5DADE2')
    ax5.set_xlabel('Score (%)', fontsize=10)
    ax5.set_title('Overall Performance Metrics', fontsize=12, fontweight='bold')
    ax5.set_xlim([0, 100])
    ax5.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (metric, value) in enumerate(zip(metrics, values)):
        ax5.text(value + 1, i, f'{value:.1f}%', va='center', fontsize=9)

    # Model comparison (from paper)
    ax6 = fig.add_subplot(gs[1, 2])
    models = ['SIDRM\n(Ours)', 'Transformer\nBaseline', 'CNN-Based', 'Rule-Based']
    accuracy = [91.0, 87.0, 85.3, 76.0]
    interpretability = [0.89, 0.41, 0.31, 0.93]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax6.bar(x - width/2, accuracy, width, label='Accuracy', color='#4ECDC4')
    bars2 = ax6.bar(x + width/2, [i*100 for i in interpretability], width,
                    label='Interpretability (×100)', color='#FFA07A')

    ax6.set_ylabel('Score', fontsize=10)
    ax6.set_title('Model Comparison (Paper Results)', fontsize=12, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(models, fontsize=8)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')

    plt.suptitle('SIDRM: Smart Interpretable Dietary Recommender Model',
                 fontsize=16, fontweight='bold', y=0.98)

    # Save figure
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'sidrm_demo_results.png', dpi=150, bbox_inches='tight')

    print(f"\n✓ Results visualized and saved to: figures/sidrm_demo_results.png")

    if not args.no_display:
        plt.show()
    plt.close()


def main(args):
    """Main demo function."""
    print("\n" + "=" * 80)
    print("  SIDRM: Smart Interpretable Dietary Recommender Model")
    print("  Complete End-to-End Demonstration")
    print("=" * 80)
    print("\nReference: Revesai & Kogeda (2025)")
    print("'Smart Interpretable Dietary Recommender Model for Vulnerable Populations'")

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Step 1: Data Preparation
    preprocessor, dataloaders, batch = demo_data_preparation(args)

    # Step 2: Model Creation
    model, model_type, input_dim, num_nutrients = demo_model_creation(batch, args)

    # Step 3: Training
    if not args.skip_training:
        trainer, history = demo_training(model, dataloaders, num_nutrients, args)
    else:
        print_section("STEP 3: TRAINING")
        print("\n⊘ Training skipped (--skip-training flag)")
        history = None

    # Step 4: Evaluation
    results = demo_evaluation(model, dataloaders, preprocessor, args)

    # Step 5: Mobile Models
    if args.mobile or args.test_all:
        demo_mobile_models(input_dim, num_nutrients)

    # Step 6: Visualization
    if history is not None and not args.no_plot:
        visualize_results(history, results)

    # Summary
    print_section("DEMONSTRATION COMPLETE")
    print("\n✓ Successfully demonstrated:")
    print("  1. NHANES data preparation and preprocessing")
    print("  2. SIDRM model creation and architecture")
    if not args.skip_training:
        print("  3. Multi-objective training with AdamW optimizer")
    print("  4. Comprehensive evaluation (overall, per-population, per-nutrient)")
    if args.mobile or args.test_all:
        print("  5. Mobile-optimized model configurations")
    if history is not None and not args.no_plot:
        print("  6. Results visualization")

    print("\n✓ Key Achievements:")
    print(f"  - Overall Accuracy: {results['accuracy']*100:.2f}%")
    print(f"  - F1-Score: {results['f1_score']:.4f}")
    if 'per_population' in results:
        avg_pop_acc = np.mean([m['accuracy'] for m in results['per_population'].values()])
        print(f"  - Avg Population Accuracy: {avg_pop_acc*100:.2f}%")

    print("\n✓ Paper Targets:")
    print("  - Overall Accuracy: 91.0%")
    print("  - F1-Score: 0.89")
    print("  - SHAP Stability: 0.91")
    print("  - Attention Consistency: 0.93")

    print("\n" + "=" * 80)
    print("For more details, see SIDRM_README.md")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SIDRM End-to-End Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick demo (3 epochs, 500 samples)
  python demo_sidrm.py --quick

  # Full demo (200 epochs, 15,560 samples)
  python demo_sidrm.py --full

  # Mobile models only
  python demo_sidrm.py --mobile --skip-training

  # Standard demo with CPU
  python demo_sidrm.py --cpu
        """
    )

    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: 500 samples, 3 epochs')
    parser.add_argument('--full', action='store_true',
                       help='Full mode: 15,560 samples, 200 epochs')
    parser.add_argument('--mobile', action='store_true',
                       help='Use mobile-optimized model')
    parser.add_argument('--test-all', action='store_true',
                       help='Test all model configurations')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training step')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting')
    parser.add_argument('--no-display', action='store_true',
                       help='Save plots but do not display')

    args = parser.parse_args()

    main(args)
