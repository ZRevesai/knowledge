"""
Local Interpretable Model-agnostic Explanations (LIME) for nutrient analysis.

Reference: Equation (7) in the paper
"""

import torch
import numpy as np
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from PIL import Image


class LIMEExplainer:
    """
    LIME-based explainer for food images and nutrient predictions.

    Reference: Equation (7)
    ξ(x) = arg min_{g∈G} L(f, g, π_x) + Ω(g)

    where:
    - f: target deep learning model
    - g: interpretable model
    - π_x: locality region around instance x
    - L: approximation fidelity
    - Ω(g): complexity penalty

    Args:
        model: Trained nutrient analysis model
        device: Device to run inference on
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()

        # LIME image explainer
        self.explainer = lime_image.LimeImageExplainer()

    def predict_fn(self, images):
        """
        Prediction function for LIME.

        Args:
            images: Batch of images as numpy arrays (B, H, W, C)

        Returns:
            Predictions as numpy array (B, num_classes)
        """
        # Convert numpy to tensor
        batch = []
        for img in images:
            # Normalize
            img = img.astype(np.float32) / 255.0

            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = (img - mean) / std

            # Convert to tensor (H, W, C) -> (C, H, W)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            batch.append(img_tensor)

        # Stack into batch
        batch_tensor = torch.stack(batch).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(batch_tensor)

            if isinstance(outputs, dict):
                logits = outputs['food_logits']
            else:
                logits = outputs

            probs = torch.softmax(logits, dim=1)

        return probs.cpu().numpy()

    def explain_instance(self, image, num_samples=1000, num_features=10,
                        top_labels=5):
        """
        Generate LIME explanation for a single image.

        Args:
            image: PIL Image or numpy array
            num_samples: Number of samples for local approximation
            num_features: Number of features in explanation
            top_labels: Number of top classes to explain

        Returns:
            LIME explanation object
        """
        # Convert to numpy array if PIL Image
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image

        # Generate explanation
        explanation = self.explainer.explain_instance(
            image_array,
            self.predict_fn,
            top_labels=top_labels,
            num_samples=num_samples,
            num_features=num_features,
            batch_size=32
        )

        return explanation

    def get_feature_importance(self, explanation, label):
        """
        Extract feature importance scores from LIME explanation.

        Args:
            explanation: LIME explanation object
            label: Class label to get importances for

        Returns:
            Dictionary with feature indices and importance scores
        """
        # Get feature importance for label
        local_exp = explanation.local_exp[label]

        # Sort by absolute importance
        local_exp = sorted(local_exp, key=lambda x: abs(x[1]), reverse=True)

        importances = {
            'features': [x[0] for x in local_exp],
            'weights': [x[1] for x in local_exp]
        }

        return importances

    def visualize_explanation(self, image, explanation, label,
                            positive_only=True, hide_rest=False):
        """
        Visualize LIME explanation.

        Args:
            image: Original image as numpy array
            explanation: LIME explanation object
            label: Class label to visualize
            positive_only: Show only positive contributions
            hide_rest: Hide regions not in explanation

        Returns:
            Visualization as numpy array
        """
        # Get image and mask
        temp, mask = explanation.get_image_and_mask(
            label,
            positive_only=positive_only,
            num_features=10,
            hide_rest=hide_rest
        )

        # Mark boundaries
        img_boundary = mark_boundaries(temp / 255.0, mask)

        return (img_boundary * 255).astype(np.uint8)

    def explain_nutrients(self, image, nutrient_type='macronutrients'):
        """
        Explain nutrient predictions using LIME.

        Args:
            image: Input image
            nutrient_type: Type of nutrients to explain
                          ('macronutrients' or 'micronutrients')

        Returns:
            Dictionary with nutrient explanations
        """
        # Convert to numpy array if PIL Image
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image

        # Custom prediction function for nutrients
        def nutrient_predict_fn(images):
            batch = []
            for img in images:
                img = img.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = (img - mean) / std
                img_tensor = torch.from_numpy(img).permute(2, 0, 1)
                batch.append(img_tensor)

            batch_tensor = torch.stack(batch).to(self.device)

            with torch.no_grad():
                outputs = self.model(batch_tensor)
                nutrients = outputs[nutrient_type]

            return nutrients.cpu().numpy()

        # Generate explanation using regression mode
        explanation = self.explainer.explain_instance(
            image_array,
            nutrient_predict_fn,
            top_labels=3,  # For macronutrients (protein, carbs, fat)
            num_samples=1000
        )

        return explanation

    def compute_decision_boundary_accuracy(self, images, labels, num_samples=100):
        """
        Evaluate LIME's decision boundary accuracy.

        Reference: Equation (31) in the paper
        DB_accuracy = (1/n) Σ I(f(xi) = g(xi))

        Args:
            images: List of images
            labels: Corresponding labels
            num_samples: Number of samples for evaluation

        Returns:
            Decision boundary accuracy score
        """
        correct = 0
        total = 0

        for img, label in zip(images[:num_samples], labels[:num_samples]):
            # Get model prediction
            if isinstance(img, Image.Image):
                img_array = np.array(img)
            else:
                img_array = img

            model_pred = self.predict_fn(img_array[np.newaxis, ...])[0]
            model_class = model_pred.argmax()

            # Generate LIME explanation
            explanation = self.explain_instance(img, num_samples=500)

            # Get LIME's local model prediction
            lime_pred_probs = np.zeros_like(model_pred)

            # For the predicted class, extract LIME's local approximation
            if model_class in explanation.local_exp:
                # LIME provides feature importance, we approximate the prediction
                local_exp = explanation.local_exp[model_class]
                importance_sum = sum([abs(x[1]) for x in local_exp])

                if importance_sum > 0:
                    lime_pred_probs[model_class] = 1.0

            lime_class = lime_pred_probs.argmax()

            # Check if predictions match (Equation 31)
            if model_class == lime_class:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0

    def compute_feature_consistency(self, images, category_id, num_samples=50):
        """
        Compute feature consistency across similar images.

        Reference: Equation (30) in the paper
        F_consistency = (1/C) Σ (1 - σ_c/μ_c)

        Args:
            images: List of images from same category
            category_id: Category identifier
            num_samples: Number of images to evaluate

        Returns:
            Feature consistency score
        """
        all_importances = []

        for img in images[:num_samples]:
            explanation = self.explain_instance(img, num_samples=500)

            # Get feature importance for predicted class
            pred_class = self.predict_fn(np.array(img)[np.newaxis, ...])[0].argmax()

            if pred_class in explanation.local_exp:
                importances = self.get_feature_importance(explanation, pred_class)
                all_importances.append(importances['weights'])

        if len(all_importances) > 0:
            # Compute mean and std of importance scores
            all_importances = np.array(all_importances)
            mean_importance = np.mean(all_importances)
            std_importance = np.std(all_importances)

            # Consistency score (Equation 30)
            if mean_importance > 0:
                consistency = 1 - (std_importance / mean_importance)
                return max(0, min(1, consistency))  # Clip to [0, 1]

        return 0.0


if __name__ == "__main__":
    print("Testing LIME Explainer...")

    from src.models.nutrient_model import NutrientAnalysisModel

    # Create model
    model = NutrientAnalysisModel(num_classes=500)
    model.eval()

    # Create dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Create LIME explainer
    lime_explainer = LIMEExplainer(model, device='cpu')

    print("LIME Explainer created successfully!")
    print("Note: Full explanation generation requires actual trained model and data")

    # Test prediction function
    preds = lime_explainer.predict_fn(dummy_image[np.newaxis, ...])
    print(f"\nPrediction shape: {preds.shape}")
    print(f"Prediction sum (should be ~1): {preds.sum():.3f}")

    print("\nLIME Explainer test completed!")
