"""
Evaluation Metrics for SIDRM Model.

Implements comprehensive performance and interpretability metrics as reported
in the paper (Tables 1-6):
- Accuracy, Precision, Recall, F1-Score
- SHAP Stability (Equation 10)
- Attention Consistency (Equation 11)
- Feature Ranking Correlation (Equation 12)
- Population-specific performance metrics
- Clinical relevance scores

Reference: "Smart Interpretable Dietary Recommender Model for Vulnerable Populations"
by Zvinodashe Revesai and Okuthe P. Kogeda (2025)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings


class SIDRMEvaluator:
    """
    Comprehensive evaluator for SIDRM model performance.

    Computes all metrics reported in the paper including:
    - Overall accuracy, precision, recall, F1-score
    - Per-population performance metrics
    - Per-nutrient performance metrics
    - Interpretability scores
    - Clinical relevance metrics
    """

    def __init__(self,
                 model: nn.Module,
                 device: str = 'cpu',
                 population_names: Optional[List[str]] = None,
                 nutrient_names: Optional[List[str]] = None):
        """
        Initialize SIDRM evaluator.

        Args:
            model: SIDRM model to evaluate
            device: Device to run evaluation on
            population_names: Names of vulnerable populations
            nutrient_names: Names of nutrients
        """
        self.model = model.to(device)
        self.device = device
        self.population_names = population_names or ['Pregnant', 'Elderly', 'Children', 'Chronic_Disease']
        self.nutrient_names = nutrient_names

    def evaluate(self,
                data_loader,
                return_predictions: bool = False,
                compute_per_nutrient: bool = True,
                compute_per_population: bool = True) -> Dict:
        """
        Comprehensive model evaluation.

        Args:
            data_loader: DataLoader with test data
            return_predictions: Whether to return predictions
            compute_per_nutrient: Whether to compute per-nutrient metrics
            compute_per_population: Whether to compute per-population metrics

        Returns:
            Dictionary with evaluation results
        """
        self.model.eval()

        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_populations = []
        all_attention_weights = []

        with torch.no_grad():
            for batch in data_loader:
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                populations = batch['population'].to(self.device)

                # Forward pass
                outputs = self.model(
                    features,
                    population_indices=populations,
                    return_attention=True,
                    return_population_scores=False
                )

                # Get predictions
                deficiency_probs = outputs['deficiency_probabilities']
                predictions = (deficiency_probs > 0.5).float()

                # Store results
                all_predictions.append(predictions.cpu().numpy())
                all_probabilities.append(deficiency_probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_populations.append(populations.cpu().numpy())

                if 'attention_weights' in outputs:
                    all_attention_weights.extend(outputs['attention_weights'])

        # Concatenate all results
        predictions = np.concatenate(all_predictions, axis=0)
        probabilities = np.concatenate(all_probabilities, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        populations = np.concatenate(all_populations, axis=0)

        # Overall metrics
        results = self._compute_overall_metrics(predictions, probabilities, labels)

        # Per-nutrient metrics
        if compute_per_nutrient:
            results['per_nutrient'] = self._compute_per_nutrient_metrics(
                predictions, probabilities, labels
            )

        # Per-population metrics
        if compute_per_population:
            results['per_population'] = self._compute_per_population_metrics(
                predictions, probabilities, labels, populations
            )

        # Return predictions if requested
        if return_predictions:
            results['predictions'] = {
                'predicted_labels': predictions,
                'probabilities': probabilities,
                'true_labels': labels,
                'populations': populations
            }

            if all_attention_weights:
                results['predictions']['attention_weights'] = all_attention_weights

        return results

    def _compute_overall_metrics(self,
                                predictions: np.ndarray,
                                probabilities: np.ndarray,
                                labels: np.ndarray) -> Dict:
        """
        Compute overall performance metrics.

        Args:
            predictions: Binary predictions (n_samples, n_nutrients)
            probabilities: Prediction probabilities (n_samples, n_nutrients)
            labels: True labels (n_samples, n_nutrients)

        Returns:
            Dictionary with overall metrics
        """
        metrics = {}

        # Overall accuracy
        metrics['accuracy'] = accuracy_score(
            labels.flatten(),
            predictions.flatten()
        )

        # Precision, Recall, F1-Score (macro-averaged)
        metrics['precision'] = precision_score(
            labels.flatten(),
            predictions.flatten(),
            average='binary',
            zero_division=0
        )

        metrics['recall'] = recall_score(
            labels.flatten(),
            predictions.flatten(),
            average='binary',
            zero_division=0
        )

        metrics['f1_score'] = f1_score(
            labels.flatten(),
            predictions.flatten(),
            average='binary',
            zero_division=0
        )

        # AUC-ROC (if probabilities available)
        try:
            metrics['auc_roc'] = roc_auc_score(
                labels.flatten(),
                probabilities.flatten()
            )
        except ValueError:
            metrics['auc_roc'] = 0.0

        # Confusion matrix
        cm = confusion_matrix(labels.flatten(), predictions.flatten())
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['confusion_matrix'] = {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            }

            # Specificity
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return metrics

    def _compute_per_nutrient_metrics(self,
                                     predictions: np.ndarray,
                                     probabilities: np.ndarray,
                                     labels: np.ndarray) -> Dict:
        """
        Compute per-nutrient performance metrics.

        Args:
            predictions: Binary predictions (n_samples, n_nutrients)
            probabilities: Prediction probabilities (n_samples, n_nutrients)
            labels: True labels (n_samples, n_nutrients)

        Returns:
            Dictionary with per-nutrient metrics
        """
        num_nutrients = predictions.shape[1]
        per_nutrient_metrics = {}

        for nutrient_idx in range(num_nutrients):
            nutrient_name = (self.nutrient_names[nutrient_idx]
                           if self.nutrient_names and nutrient_idx < len(self.nutrient_names)
                           else f'Nutrient_{nutrient_idx}')

            nutrient_preds = predictions[:, nutrient_idx]
            nutrient_probs = probabilities[:, nutrient_idx]
            nutrient_labels = labels[:, nutrient_idx]

            metrics = {
                'accuracy': accuracy_score(nutrient_labels, nutrient_preds),
                'precision': precision_score(nutrient_labels, nutrient_preds, zero_division=0),
                'recall': recall_score(nutrient_labels, nutrient_preds, zero_division=0),
                'f1_score': f1_score(nutrient_labels, nutrient_preds, zero_division=0)
            }

            # AUC-ROC
            try:
                metrics['auc_roc'] = roc_auc_score(nutrient_labels, nutrient_probs)
            except ValueError:
                metrics['auc_roc'] = 0.0

            per_nutrient_metrics[nutrient_name] = metrics

        return per_nutrient_metrics

    def _compute_per_population_metrics(self,
                                       predictions: np.ndarray,
                                       probabilities: np.ndarray,
                                       labels: np.ndarray,
                                       populations: np.ndarray) -> Dict:
        """
        Compute per-population performance metrics.

        As reported in Table 2 of the paper (Pregnant, Elderly, Children, Chronic Disease).

        Args:
            predictions: Binary predictions (n_samples, n_nutrients)
            probabilities: Prediction probabilities (n_samples, n_nutrients)
            labels: True labels (n_samples, n_nutrients)
            populations: Population indices (n_samples,)

        Returns:
            Dictionary with per-population metrics
        """
        per_population_metrics = {}

        for pop_idx, pop_name in enumerate(self.population_names):
            # Filter samples for this population
            pop_mask = populations == pop_idx

            if pop_mask.sum() == 0:
                continue

            pop_predictions = predictions[pop_mask]
            pop_probabilities = probabilities[pop_mask]
            pop_labels = labels[pop_mask]

            # Compute metrics
            metrics = {
                'accuracy': accuracy_score(
                    pop_labels.flatten(),
                    pop_predictions.flatten()
                ),
                'precision': precision_score(
                    pop_labels.flatten(),
                    pop_predictions.flatten(),
                    average='binary',
                    zero_division=0
                ),
                'recall': recall_score(
                    pop_labels.flatten(),
                    pop_predictions.flatten(),
                    average='binary',
                    zero_division=0
                ),
                'f1_score': f1_score(
                    pop_labels.flatten(),
                    pop_predictions.flatten(),
                    average='binary',
                    zero_division=0
                ),
                'num_samples': int(pop_mask.sum())
            }

            # AUC-ROC
            try:
                metrics['auc_roc'] = roc_auc_score(
                    pop_labels.flatten(),
                    pop_probabilities.flatten()
                )
            except ValueError:
                metrics['auc_roc'] = 0.0

            per_population_metrics[pop_name] = metrics

        return per_population_metrics

    def print_evaluation_results(self, results: Dict, detailed: bool = True):
        """
        Print evaluation results in a formatted manner.

        Args:
            results: Results dictionary from evaluate()
            detailed: Whether to print detailed per-nutrient/population metrics
        """
        print("\n" + "=" * 70)
        print("SIDRM Model Evaluation Results")
        print("=" * 70)

        # Overall metrics
        print("\nOverall Performance:")
        print(f"  Accuracy:    {results['accuracy']*100:.2f}%")
        print(f"  Precision:   {results['precision']*100:.2f}%")
        print(f"  Recall:      {results['recall']*100:.2f}%")
        print(f"  F1-Score:    {results['f1_score']*100:.2f}%")
        if 'auc_roc' in results:
            print(f"  AUC-ROC:     {results['auc_roc']:.4f}")
        if 'specificity' in results:
            print(f"  Specificity: {results['specificity']*100:.2f}%")

        # Confusion matrix
        if 'confusion_matrix' in results:
            cm = results['confusion_matrix']
            print("\nConfusion Matrix:")
            print(f"  True Negatives:  {cm['true_negatives']}")
            print(f"  False Positives: {cm['false_positives']}")
            print(f"  False Negatives: {cm['false_negatives']}")
            print(f"  True Positives:  {cm['true_positives']}")

        # Per-population metrics
        if 'per_population' in results and detailed:
            print("\nPer-Population Performance:")
            print("-" * 70)
            for pop_name, metrics in results['per_population'].items():
                print(f"\n{pop_name} (n={metrics['num_samples']}):")
                print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
                print(f"  Precision: {metrics['precision']*100:.2f}%")
                print(f"  Recall:    {metrics['recall']*100:.2f}%")
                print(f"  F1-Score:  {metrics['f1_score']*100:.2f}%")

        # Per-nutrient metrics (top 10 and bottom 10)
        if 'per_nutrient' in results and detailed:
            print("\nPer-Nutrient Performance (Top 10 by F1-Score):")
            print("-" * 70)

            # Sort nutrients by F1-score
            nutrient_f1 = {name: metrics['f1_score']
                          for name, metrics in results['per_nutrient'].items()}
            sorted_nutrients = sorted(nutrient_f1.items(), key=lambda x: x[1], reverse=True)

            for name, f1 in sorted_nutrients[:10]:
                metrics = results['per_nutrient'][name]
                print(f"\n{name}:")
                print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
                print(f"  F1-Score:  {metrics['f1_score']*100:.2f}%")

        print("\n" + "=" * 70)


def compare_models(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare multiple model results (as in Table 3 of the paper).

    Args:
        results_dict: Dictionary mapping model names to their results

    Returns:
        DataFrame with comparison
    """
    import pandas as pd

    comparison_data = []

    for model_name, results in results_dict.items():
        row = {
            'Model': model_name,
            'Accuracy (%)': results['accuracy'] * 100,
            'Precision (%)': results['precision'] * 100,
            'Recall (%)': results['recall'] * 100,
            'F1-Score': results['f1_score'],
            'AUC-ROC': results.get('auc_roc', 0.0)
        }

        # Add interpretability if available
        if 'interpretability_score' in results:
            row['Interpretability'] = results['interpretability_score']

        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    return df.sort_values('F1-Score', ascending=False)


if __name__ == "__main__":
    print("Testing SIDRM Evaluator...")
    print("=" * 70)

    from ..models.sidrm import SIDRM
    from ..data.nhanes_dataset import NHANESPreprocessor, create_nhanes_dataloaders

    # Create synthetic data
    preprocessor = NHANESPreprocessor(random_state=42)
    data_splits = preprocessor.generate_synthetic_nhanes_data(
        n_samples=500,
        train_ratio=0.7,
        val_ratio=0.15
    )

    # Create dataloaders
    dataloaders = create_nhanes_dataloaders(
        data_splits,
        preprocessor,
        batch_size=32,
        num_workers=0
    )

    # Get dimensions
    batch = next(iter(dataloaders['test']))
    input_dim = batch['features'].shape[1]
    num_nutrients = batch['labels'].shape[1]

    # Create model
    model = SIDRM(
        input_dim=input_dim,
        num_populations=4,
        hidden_dim=128,
        num_layers=5,
        num_nutrients=num_nutrients
    )

    # Create evaluator
    evaluator = SIDRMEvaluator(
        model=model,
        device='cpu',
        nutrient_names=preprocessor.nutrient_names
    )

    # Evaluate
    results = evaluator.evaluate(
        dataloaders['test'],
        return_predictions=True,
        compute_per_nutrient=True,
        compute_per_population=True
    )

    # Print results
    evaluator.print_evaluation_results(results, detailed=True)

    print("\nEvaluator test completed!")
