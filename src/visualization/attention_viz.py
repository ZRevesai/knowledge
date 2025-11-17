"""
Attention and interpretability visualization for KGNN.

Creates visualizations for:
- Attention heatmaps
- Feature importance
- Counterfactual explanations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import torch


def plot_attention_heatmap(attention_weights: np.ndarray,
                          feature_names: Optional[List[str]] = None,
                          sample_names: Optional[List[str]] = None,
                          title: str = 'Attention Heatmap',
                          save_path: Optional[str] = None,
                          show: bool = True):
    """
    Plot attention weights as a heatmap.

    Args:
        attention_weights: Attention weights (n_samples, n_features)
        feature_names: Names of features
        sample_names: Names of samples
        title: Plot title
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    plt.figure(figsize=(14, 8))

    # Limit to reasonable number of features/samples for visualization
    max_samples = 20
    max_features = 30

    if attention_weights.shape[0] > max_samples:
        attention_weights = attention_weights[:max_samples]
        if sample_names:
            sample_names = sample_names[:max_samples]

    if attention_weights.shape[1] > max_features:
        # Select top features by average attention
        avg_attention = np.mean(np.abs(attention_weights), axis=0)
        top_indices = np.argsort(avg_attention)[-max_features:]
        attention_weights = attention_weights[:, top_indices]
        if feature_names:
            feature_names = [feature_names[i] for i in top_indices]

    # Create heatmap
    sns.heatmap(attention_weights,
               cmap='YlOrRd',
               xticklabels=feature_names if feature_names else range(attention_weights.shape[1]),
               yticklabels=sample_names if sample_names else range(attention_weights.shape[0]),
               cbar_kws={'label': 'Attention Weight'},
               annot=False)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Features', fontsize=12, fontweight='bold')
    plt.ylabel('Samples', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention heatmap to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_feature_importance(importance_scores: np.ndarray,
                           feature_names: Optional[List[str]] = None,
                           top_k: int = 20,
                           title: str = 'Feature Importance',
                           save_path: Optional[str] = None,
                           show: bool = True):
    """
    Plot feature importance scores.

    Args:
        importance_scores: Feature importance scores (n_features,)
        feature_names: Names of features
        top_k: Number of top features to display
        title: Plot title
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    # Get top k features
    top_indices = np.argsort(np.abs(importance_scores))[-top_k:][::-1]
    top_scores = importance_scores[top_indices]

    if feature_names:
        top_names = [feature_names[i] for i in top_indices]
    else:
        top_names = [f'Feature {i}' for i in top_indices]

    # Create colors based on positive/negative
    colors = ['green' if score > 0 else 'red' for score in top_scores]

    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(top_scores)), top_scores, color=colors, alpha=0.7, edgecolor='black')

    plt.yticks(range(len(top_scores)), top_names)
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, top_scores)):
        plt.text(score, i, f' {score:.3f}', va='center',
                ha='left' if score > 0 else 'right', fontsize=9)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_explanation(explanation: Dict,
                         title: str = 'Prediction Explanation',
                         save_path: Optional[str] = None,
                         show: bool = True):
    """
    Visualize comprehensive explanation for a prediction.

    Args:
        explanation: Explanation dictionary from model
        title: Plot title
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Prediction probabilities
    if 'predictions' in explanation:
        predictions = explanation['predictions']
        micronutrients = list(predictions.keys())
        probs = [predictions[m]['probability'] for m in micronutrients]

        colors = ['red' if p > 0.5 else 'green' for p in probs]

        axes[0].barh(range(len(micronutrients)), probs, color=colors, alpha=0.7, edgecolor='black')
        axes[0].set_yticks(range(len(micronutrients)))
        axes[0].set_yticklabels(micronutrients)
        axes[0].set_xlabel('Deficiency Probability', fontsize=11, fontweight='bold')
        axes[0].set_title('Predicted Deficiencies', fontsize=12, fontweight='bold')
        axes[0].axvline(x=0.5, color='black', linestyle='--', linewidth=1)
        axes[0].set_xlim([0, 1])
        axes[0].grid(axis='x', alpha=0.3)

        # Add risk labels
        for i, (m, p) in enumerate(zip(micronutrients, probs)):
            risk = predictions[m]['deficiency_risk']
            axes[0].text(p, i, f'  {risk}', va='center', fontsize=9)

    # Plot 2: Top contributing features
    if 'top_features' in explanation:
        top_features = explanation['top_features'][:10]  # Top 10

        feature_names = [f['name'] if 'name' in f else f'Feature {f["index"]}'
                        for f in top_features]
        importance = [f['importance'] for f in top_features]

        colors = ['blue' if imp > 0 else 'orange' for imp in importance]

        axes[1].barh(range(len(feature_names)), importance, color=colors, alpha=0.7, edgecolor='black')
        axes[1].set_yticks(range(len(feature_names)))
        axes[1].set_yticklabels(feature_names)
        axes[1].set_xlabel('Feature Importance', fontsize=11, fontweight='bold')
        axes[1].set_title('Top Contributing Features', fontsize=12, fontweight='bold')
        axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        axes[1].grid(axis='x', alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved explanation visualization to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_counterfactual_explanation(counterfactual: Dict,
                                   feature_names: Optional[List[str]] = None,
                                   save_path: Optional[str] = None,
                                   show: bool = True):
    """
    Visualize counterfactual explanation.

    Args:
        counterfactual: Counterfactual explanation dictionary
        feature_names: Names of all features
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    plt.figure(figsize=(12, 6))

    # Extract recommended changes
    changes = counterfactual['recommended_changes']
    current_pred = counterfactual['current_prediction']
    micronutrient = counterfactual['micronutrient']

    feature_indices = [c['feature_index'] for c in changes]
    importance = [c['importance'] for c in changes]
    recommendations = [c['recommendation'] for c in changes]

    # Get feature names
    if feature_names:
        names = [feature_names[i] for i in feature_indices]
    else:
        names = [f'Feature {i}' for i in feature_indices]

    # Create colors based on recommendation
    colors = ['green' if rec == 'increase' else 'red' for rec in recommendations]

    plt.barh(range(len(names)), importance, color=colors, alpha=0.7, edgecolor='black')
    plt.yticks(range(len(names)), names)
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.title(f'Counterfactual Explanation for {micronutrient}\n'
             f'Current Prediction: {current_pred:.2%}',
             fontsize=13, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)

    # Add recommendation labels
    for i, (imp, rec) in enumerate(zip(importance, recommendations)):
        plt.text(imp, i, f'  {rec.upper()}', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved counterfactual explanation to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_attention_distribution(attention_weights: np.ndarray,
                               title: str = 'Attention Distribution',
                               save_path: Optional[str] = None,
                               show: bool = True):
    """
    Plot distribution of attention weights.

    Args:
        attention_weights: Attention weights (n_samples, n_features)
        title: Plot title
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Histogram of attention weights
    axes[0].hist(attention_weights.flatten(), bins=50, alpha=0.7,
                color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Attention Weight', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0].set_title('Distribution of Attention Weights', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # Plot 2: Average attention per feature
    avg_attention = np.mean(attention_weights, axis=0)
    sorted_indices = np.argsort(avg_attention)[::-1][:20]  # Top 20
    sorted_attention = avg_attention[sorted_indices]

    axes[1].bar(range(len(sorted_attention)), sorted_attention,
               color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Feature Rank', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Average Attention', fontsize=11, fontweight='bold')
    axes[1].set_title('Top 20 Features by Average Attention', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention distribution plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    print("Testing Attention Visualization Functions...")

    # Test 1: Attention heatmap
    print("\n1. Testing attention heatmap...")
    attention = np.random.rand(10, 20)
    feature_names = [f'Feature_{i}' for i in range(20)]
    plot_attention_heatmap(attention, feature_names=feature_names, show=False)

    # Test 2: Feature importance
    print("\n2. Testing feature importance plot...")
    importance = np.random.randn(50)
    plot_feature_importance(importance, top_k=15, show=False)

    # Test 3: Explanation visualization
    print("\n3. Testing explanation visualization...")
    explanation = {
        'predictions': {
            'Vitamin_B12': {'probability': 0.85, 'deficiency_risk': 'High'},
            'Vitamin_D': {'probability': 0.65, 'deficiency_risk': 'Medium'},
            'Iron': {'probability': 0.25, 'deficiency_risk': 'Low'},
        },
        'top_features': [
            {'name': 'Serum_B12', 'importance': 0.85, 'value': 150},
            {'name': 'Dietary_B12', 'importance': 0.72, 'value': 2.1},
            {'name': 'Age', 'importance': 0.58, 'value': 75},
        ]
    }
    visualize_explanation(explanation, show=False)

    # Test 4: Counterfactual
    print("\n4. Testing counterfactual visualization...")
    counterfactual = {
        'current_prediction': 0.85,
        'micronutrient': 'Vitamin_B12',
        'recommended_changes': [
            {'feature_index': 5, 'importance': 0.85, 'recommendation': 'increase'},
            {'feature_index': 12, 'importance': 0.72, 'recommendation': 'increase'},
            {'feature_index': 20, 'importance': -0.45, 'recommendation': 'decrease'},
        ]
    }
    plot_counterfactual_explanation(counterfactual, show=False)

    print("\nAttention visualization test completed!")
