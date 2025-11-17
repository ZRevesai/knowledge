"""
Visualization functions for KGNN results.

Creates plots for:
- Training history
- Model performance comparison
- Confusion matrices
- ROC curves
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_training_history(history: Dict,
                          save_path: Optional[str] = None,
                          show: bool = True):
    """
    Plot training history including loss curves.

    Args:
        history: Training history dictionary
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Total Loss
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Prediction Loss
    axes[0, 1].plot(history['train_pred_loss'], label='Train', linewidth=2)
    axes[0, 1].plot(history['val_pred_loss'], label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Prediction Loss (L_pred)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Knowledge Loss
    axes[1, 0].plot(history['train_know_loss'], label='Train', linewidth=2)
    axes[1, 0].plot(history['val_know_loss'], label='Validation', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Knowledge Loss (L_know)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Explanation Loss
    axes[1, 1].plot(history['train_expl_loss'], label='Train', linewidth=2)
    axes[1, 1].plot(history['val_expl_loss'], label='Validation', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Explanation Loss (L_expl)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(cm: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         title: str = 'Confusion Matrix',
                         save_path: Optional[str] = None,
                         show: bool = True):
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: Names of classes
        title: Plot title
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    plt.figure(figsize=(10, 8))

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=class_names if class_names else range(len(cm)),
               yticklabels=class_names if class_names else range(len(cm)),
               cbar_kws={'label': 'Normalized Count'})

    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curves(y_true: np.ndarray,
                   y_prob: np.ndarray,
                   class_names: Optional[List[str]] = None,
                   title: str = 'ROC Curves',
                   save_path: Optional[str] = None,
                   show: bool = True):
    """
    Plot ROC curves for multi-label classification.

    Args:
        y_true: True labels (n_samples, n_classes)
        y_prob: Predicted probabilities (n_samples, n_classes)
        class_names: Names of classes
        title: Plot title
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    from sklearn.metrics import roc_curve, auc

    n_classes = y_true.shape[1]
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]

    plt.figure(figsize=(12, 8))

    # Plot ROC curve for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, linewidth=2, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curves to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_model_comparison(comparison_data: Dict,
                         metric: str = 'accuracy',
                         title: Optional[str] = None,
                         save_path: Optional[str] = None,
                         show: bool = True):
    """
    Plot bar chart comparing models.

    Args:
        comparison_data: Dictionary with model comparison results
        metric: Metric to plot ('accuracy', 'f1_score', etc.)
        title: Plot title
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    if 'summary' in comparison_data:
        data = comparison_data['summary']['performance']
    else:
        data = comparison_data

    models = list(data.keys())
    values = [data[model].get(metric, 0.0) for model in models]

    # Create color palette
    colors = sns.color_palette("husl", len(models))

    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, values, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    if title is None:
        title = f'Model Comparison: {metric.replace("_", " ").title()}'

    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel(f'{metric.replace("_", " ").title()} (%)', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved model comparison to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_performance_vs_interpretability(comparison_data: Dict,
                                        save_path: Optional[str] = None,
                                        show: bool = True):
    """
    Plot scatter plot of performance vs interpretability (Figure 3 from paper).

    Args:
        comparison_data: Dictionary with model comparison results
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    if 'summary' not in comparison_data:
        print("Warning: 'summary' not found in comparison data")
        return

    perf_data = comparison_data['summary']['performance']
    interp_data = comparison_data['summary']['interpretability']

    models = list(perf_data.keys())
    accuracy = [perf_data[m]['accuracy'] for m in models]
    interpretability = [interp_data[m]['overall_score'] * 100 for m in models]

    plt.figure(figsize=(10, 8))

    # Create scatter plot
    colors = sns.color_palette("husl", len(models))

    for i, model in enumerate(models):
        plt.scatter(interpretability[i], accuracy[i],
                   s=300, c=[colors[i]], alpha=0.7,
                   edgecolors='black', linewidth=2,
                   label=model)

        # Add model name annotation
        plt.annotate(model,
                    (interpretability[i], accuracy[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')

    plt.xlabel('Interpretability Score', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Performance vs Interpretability Trade-off',
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved performance vs interpretability plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_per_micronutrient_performance(results: Dict,
                                      micronutrient_names: List[str],
                                      metric: str = 'f1_score',
                                      save_path: Optional[str] = None,
                                      show: bool = True):
    """
    Plot performance across different micronutrients (Figure 2 from paper).

    Args:
        results: Results dictionary from evaluation
        micronutrient_names: Names of micronutrients
        metric: Metric to plot
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    per_class_data = results['performance'].get('per_class', {})

    if not per_class_data:
        print("Warning: No per-class data available")
        return

    values = []
    for i in range(len(micronutrient_names)):
        class_key = f'class_{i}'
        if class_key in per_class_data:
            values.append(per_class_data[class_key].get(metric, 0.0) * 100)
        else:
            values.append(0.0)

    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(micronutrient_names)), values,
                   color=sns.color_palette("viridis", len(micronutrient_names)),
                   alpha=0.8, edgecolor='black')

    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%',
                ha='center', va='bottom', fontsize=9)

    plt.xlabel('Micronutrient', fontsize=12, fontweight='bold')
    plt.ylabel(f'{metric.replace("_", " ").title()} (%)', fontsize=12, fontweight='bold')
    plt.title(f'Performance Across Individual Micronutrients',
             fontsize=14, fontweight='bold')
    plt.xticks(range(len(micronutrient_names)), micronutrient_names,
              rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved per-micronutrient performance to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    print("Testing Visualization Functions...")

    # Test 1: Training history
    print("\n1. Testing training history plot...")
    history = {
        'train_loss': np.random.randn(50).cumsum() * -0.01 + 1.0,
        'val_loss': np.random.randn(50).cumsum() * -0.01 + 1.0,
        'train_pred_loss': np.random.randn(50).cumsum() * -0.01 + 0.8,
        'val_pred_loss': np.random.randn(50).cumsum() * -0.01 + 0.8,
        'train_know_loss': np.random.rand(50) * 0.1,
        'val_know_loss': np.random.rand(50) * 0.1,
        'train_expl_loss': np.random.rand(50) * 0.1,
        'val_expl_loss': np.random.rand(50) * 0.1,
    }
    plot_training_history(history, show=False)

    # Test 2: Model comparison
    print("\n2. Testing model comparison plot...")
    comparison_data = {
        'KGNN': {'accuracy': 93.7, 'f1_score': 92.1},
        'FFNN': {'accuracy': 88.2, 'f1_score': 85.9},
        'Transformer': {'accuracy': 89.1, 'f1_score': 87.7},
        'XGBoost': {'accuracy': 85.4, 'f1_score': 84.0},
    }
    plot_model_comparison(comparison_data, metric='accuracy', show=False)

    print("\nVisualization functions test completed!")
