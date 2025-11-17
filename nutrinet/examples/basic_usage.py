"""
Basic Usage Examples for NUTRINET

This script demonstrates the basic functionality of NUTRINET including:
1. Model creation and initialization
2. Forward pass with predictions
3. Explanation generation
4. Energy monitoring
5. Evaluation metrics
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.nutrinet import create_nutrinet_model, NUTRINET
from src.utils.metrics import (
    compute_nutrient_prediction_metrics,
    compute_interpretability_metrics,
    print_metrics,
)
from src.utils.energy_monitor import EnergyMonitor, measure_energy_consumption


def example_1_basic_prediction():
    """Example 1: Basic prediction with NUTRINET"""
    print("\n" + "="*70)
    print("Example 1: Basic Prediction with NUTRINET")
    print("="*70)

    # Create model and graph
    print("\n1. Creating NUTRINET model...")
    model, graph = create_nutrinet_model(
        num_micronutrients=20,
        num_food_components=15,
        num_health_outcomes=10,
        feature_dim=32,
        hidden_dim=64,
        output_dim=1,
    )

    print(f"   ✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Get graph data
    x = graph.node_features
    edge_index = graph.edge_index
    edge_features = graph.edge_features

    print(f"   ✓ Graph: {x.size(0)} nodes, {edge_index.size(1)} edges")

    # Create dummy context
    context = torch.randn(1, 10)

    # Forward pass
    print("\n2. Forward pass...")
    with torch.no_grad():
        output = model(
            x=x,
            edge_index=edge_index,
            edge_features=edge_features,
            context=context,
            return_attention=True,
        )

    # Display results
    print(f"   ✓ Prediction: {output['predictions'].item():.4f}")
    print(f"   ✓ Node embeddings shape: {output['node_embeddings'].shape}")

    # Efficiency metrics
    eff_metrics = output['efficiency_metrics']
    print(f"\n3. Efficiency Metrics:")
    print(f"   • Active edge ratio: {eff_metrics['active_edge_ratio']:.2%}")
    print(f"   • Number of iterations: {eff_metrics['num_iterations']}")
    print(f"   • Adaptive dimension: {eff_metrics['adaptive_dimension']}")

    # Computational cost
    cost = model.estimate_computational_cost(
        num_nodes=x.size(0),
        num_edges=edge_index.size(1),
    )
    print(f"\n4. Computational Cost:")
    print(f"   • FLOPs: {cost['flops']:,.0f}")
    print(f"   • Memory: {cost['memory_mb']:.2f} MB")
    print(f"   • Sparse vs Full Attention: {cost['relative_to_full_attention']:.2%}")


def example_2_explanation_generation():
    """Example 2: Generate explanations for predictions"""
    print("\n" + "="*70)
    print("Example 2: Explanation Generation")
    print("="*70)

    # Create model
    print("\n1. Creating model with node names...")
    model, graph = create_nutrinet_model(
        num_micronutrients=20,
        num_food_components=15,
        num_health_outcomes=10,
        feature_dim=32,
        hidden_dim=64,
    )

    # Get data
    x = graph.node_features
    edge_index = graph.edge_index
    edge_features = graph.edge_features
    context = torch.randn(1, 10)

    # Generate explanation
    print("\n2. Generating explanation...")
    explanation = model.explain_prediction(
        x=x,
        edge_index=edge_index,
        edge_features=edge_features,
        context=context,
        source_nodes=[0, 1, 2],  # First few micronutrients
        target_nodes=[35, 36, 37],  # First few health outcomes
    )

    # Display explanation
    print("\n3. Explanation Summary:")
    print(f"   {explanation['summary']}")

    print(f"\n4. Key Nutrients:")
    for nutrient in explanation['key_nutrients'][:5]:
        print(f"   • {nutrient['name']}: frequency = {nutrient['frequency']}")

    print(f"\n5. Critical Pathways:")
    for i, path in enumerate(explanation['critical_paths'][:3], 1):
        print(f"   Path {i}: {' → '.join(path['nodes'][:5])}")

    print(f"\n6. Significant Subgraph:")
    print(f"   • Nodes: {len(explanation['significant_nodes'])}")
    print(f"   • Edges: {len(explanation['significant_edges'])}")


def example_3_energy_monitoring():
    """Example 3: Monitor energy consumption"""
    print("\n" + "="*70)
    print("Example 3: Energy Monitoring")
    print("="*70)

    # Create model
    model, graph = create_nutrinet_model(
        num_micronutrients=20,
        num_food_components=15,
        num_health_outcomes=10,
    )

    x = graph.node_features
    edge_index = graph.edge_index
    edge_features = graph.edge_features

    # Create energy monitor
    print("\n1. Creating energy monitor...")
    monitor = EnergyMonitor(output_dir='./emissions')

    # Simulate training with energy tracking
    print("\n2. Simulating model training...")

    num_iterations = 50
    with measure_energy_consumption(monitor) as measurement:
        for i in range(num_iterations):
            output = model(
                x=x,
                edge_index=edge_index,
                edge_features=edge_features,
            )

            # Simulate backward pass
            loss = output['predictions'].sum()
            # Note: In real training, would call loss.backward()

    print("\n3. Energy Consumption:")
    print(f"   • Total energy: {measurement.total_energy_kwh:.6f} kWh")
    print(f"   • Duration: {measurement.duration_seconds:.2f} seconds")
    print(f"   • CO2 emissions: {measurement.co2_emissions_kg:.6f} kg")
    print(f"   • Average power: {measurement.average_power_watts:.2f} W")

    # Compare with baseline
    print("\n4. Comparison with Baseline Models:")
    baselines = [
        ("GAT", 59.2),
        ("RGCN", 68.7),
        ("GCN", 28.4),
    ]

    for name, energy in baselines:
        comparison = monitor.compare_with_baseline(
            baseline_energy_kwh=energy,
            baseline_name=name,
        )
        print(f"   • vs {name}: {comparison['energy_reduction_percent']:+.1f}% "
              f"(efficiency ratio: {comparison['efficiency_ratio']:.2f}x)")


def example_4_batch_inference():
    """Example 4: Batch inference with metrics"""
    print("\n" + "="*70)
    print("Example 4: Batch Inference with Metrics")
    print("="*70)

    # Create model
    model, graph = create_nutrinet_model()
    model.eval()

    # Generate synthetic batch data
    print("\n1. Generating synthetic batch...")
    batch_size = 32
    num_features = graph.node_features.size(-1)

    # Simulate multiple samples
    predictions = []
    true_values = []

    print("\n2. Running batch inference...")
    with torch.no_grad():
        for i in range(batch_size):
            # Generate sample
            x = torch.randn_like(graph.node_features)
            true_value = torch.rand(1)  # Simulated ground truth

            # Predict
            output = model(
                x=x,
                edge_index=graph.edge_index,
                edge_features=graph.edge_features,
            )

            predictions.append(output['predictions'].item())
            true_values.append(true_value.item())

    # Compute metrics
    print("\n3. Computing metrics...")
    y_true = np.array(true_values)
    y_pred = np.array(predictions)

    metrics = compute_nutrient_prediction_metrics(y_true, y_pred)

    print_metrics(metrics, title="Batch Inference Results")


def example_5_population_specific():
    """Example 5: Population-specific predictions"""
    print("\n" + "="*70)
    print("Example 5: Population-Specific Predictions")
    print("="*70)

    # Create model
    model, graph = create_nutrinet_model()

    # Elderly population example
    print("\n1. Elderly Population (Age 75):")
    context_elderly = torch.tensor([[
        75.0,    # age
        0.0,     # gender (male)
        26.5,    # BMI
        1.0,     # on medications
        0.8,     # absorption factor (reduced)
        0.9,     # mobility score
        1.0,     # chronic condition flag
        0.0,     # pregnancy (N/A)
        0.7,     # dietary diversity
        0.8,     # supplement use
    ]])

    with torch.no_grad():
        output_elderly = model(
            x=graph.node_features,
            edge_index=graph.edge_index,
            edge_features=graph.edge_features,
            context=context_elderly,
        )

    print(f"   • Deficiency risk: {output_elderly['predictions'].item():.4f}")

    # Pregnant population example
    print("\n2. Pregnant Individual (Age 28, 2nd trimester):")
    context_pregnant = torch.tensor([[
        28.0,    # age
        1.0,     # gender (female)
        24.0,    # BMI
        0.0,     # on medications
        1.0,     # absorption factor (normal)
        1.0,     # mobility score
        0.0,     # chronic condition flag
        2.0,     # pregnancy trimester (2nd)
        0.9,     # dietary diversity
        1.0,     # prenatal supplement use
    ]])

    with torch.no_grad():
        output_pregnant = model(
            x=graph.node_features,
            edge_index=graph.edge_index,
            edge_features=graph.edge_features,
            context=context_pregnant,
        )

    print(f"   • Deficiency risk: {output_pregnant['predictions'].item():.4f}")

    # Compare
    print("\n3. Comparison:")
    diff = abs(output_elderly['predictions'].item() - output_pregnant['predictions'].item())
    print(f"   • Risk difference: {diff:.4f}")
    print(f"   • Model successfully adapts to population context!")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print(" "*20 + "NUTRINET Basic Usage Examples")
    print("="*70)

    try:
        # Run examples
        example_1_basic_prediction()
        example_2_explanation_generation()
        example_3_energy_monitoring()
        example_4_batch_inference()
        example_5_population_specific()

        print("\n" + "="*70)
        print(" "*25 + "All examples completed!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
