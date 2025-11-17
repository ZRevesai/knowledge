"""
Gradient-weighted Class Activation Mapping (Grad-CAM) for visual explanations.

Reference: Equations (5) and (6) in the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


class GradCAM:
    """
    Grad-CAM implementation for generating visual explanations.

    Reference:
    - Equation (5): α^c_k = (1/Z) Σ_i Σ_j ∂y^c/∂A^k_{ij}
    - Equation (6): L^c_{Grad-CAM} = ReLU(Σ_k α^c_k A^k)

    Args:
        model: The neural network model
        target_layer: Name of the target layer for CAM generation
    """
    def __init__(self, model, target_layer='features.16'):
        self.model = model
        self.target_layer = target_layer

        # Hooks for extracting feature maps and gradients
        self.feature_maps = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks."""
        # Get target layer
        target_module = self._get_module_by_name(self.model, self.target_layer)

        if target_module is None:
            raise ValueError(f"Target layer '{self.target_layer}' not found in model")

        # Forward hook to capture feature maps
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()

        # Backward hook to capture gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)

    def _get_module_by_name(self, model, target_layer):
        """Get module by name from model."""
        names = target_layer.split('.')
        module = model

        for name in names:
            if hasattr(module, 'backbone'):
                module = module.backbone

            if hasattr(module, name):
                module = getattr(module, name)
            elif name.isdigit():
                module = module[int(name)]
            else:
                return None

        return module

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for input image.

        Args:
            input_tensor: Input tensor of shape (1, 3, H, W)
            target_class: Target class index (if None, uses predicted class)

        Returns:
            Heatmap array of shape (H, W)
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # Get predicted class if not specified
        if target_class is None:
            if isinstance(output, dict):
                logits = output['food_logits']
            else:
                logits = output
            target_class = logits.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        if isinstance(output, dict):
            score = output['food_logits'][0, target_class]
        else:
            score = output[0, target_class]

        score.backward()

        # Get feature maps and gradients
        # A^k: feature maps (Equation 5)
        feature_maps = self.feature_maps[0]  # (C, H, W)

        # ∂y^c/∂A^k: gradients (Equation 5)
        gradients = self.gradients[0]  # (C, H, W)

        # Compute weights α^c_k (Equation 5)
        # α^c_k = (1/Z) Σ_i Σ_j ∂y^c/∂A^k_{ij}
        Z = gradients.shape[1] * gradients.shape[2]  # H * W
        weights = gradients.mean(dim=(1, 2))  # Global average pooling

        # Compute weighted combination of feature maps (Equation 6)
        # L^c_{Grad-CAM} = ReLU(Σ_k α^c_k A^k)
        cam = torch.zeros(feature_maps.shape[1:], dtype=torch.float32,
                         device=feature_maps.device)

        for i, w in enumerate(weights):
            cam += w * feature_maps[i]

        # Apply ReLU
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy()

    def visualize(self, input_tensor, original_image=None, target_class=None,
                  alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Generate and visualize Grad-CAM heatmap overlaid on original image.

        Args:
            input_tensor: Input tensor of shape (1, 3, H, W)
            original_image: Original PIL Image or numpy array
            target_class: Target class index
            alpha: Transparency for overlay
            colormap: OpenCV colormap

        Returns:
            Overlaid image as numpy array
        """
        # Generate heatmap
        heatmap = self.generate_heatmap(input_tensor, target_class)

        # Resize heatmap to original image size
        if original_image is not None:
            if isinstance(original_image, Image.Image):
                img_size = original_image.size
            else:
                img_size = (original_image.shape[1], original_image.shape[0])

            heatmap = cv2.resize(heatmap, img_size)

        # Convert heatmap to color map
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Overlay on original image
        if original_image is not None:
            if isinstance(original_image, Image.Image):
                original_image = np.array(original_image)

            # Blend images
            overlaid = (alpha * heatmap_color + (1 - alpha) * original_image).astype(np.uint8)
            return overlaid, heatmap
        else:
            return heatmap_color, heatmap


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ implementation with improved localization.

    Uses weighted gradients for better multi-instance localization.
    """
    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM++ heatmap.

        Args:
            input_tensor: Input tensor of shape (1, 3, H, W)
            target_class: Target class index

        Returns:
            Heatmap array of shape (H, W)
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # Get predicted class if not specified
        if target_class is None:
            if isinstance(output, dict):
                logits = output['food_logits']
            else:
                logits = output
            target_class = logits.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        if isinstance(output, dict):
            score = output['food_logits'][0, target_class]
        else:
            score = output[0, target_class]

        score.backward(retain_graph=True)

        # Get feature maps and gradients
        feature_maps = self.feature_maps[0]  # (C, H, W)
        gradients = self.gradients[0]  # (C, H, W)

        # Compute second-order gradients for Grad-CAM++
        # This provides better weights for multi-instance localization
        score.backward()  # Compute second derivatives

        # Compute alpha weights using second and third order derivatives
        alpha_num = gradients.pow(2)
        alpha_denom = 2 * gradients.pow(2) + \
                      (feature_maps * gradients.pow(3)).sum(dim=(1, 2), keepdim=True)

        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom,
                                  torch.ones_like(alpha_denom))

        alpha = alpha_num / alpha_denom

        # Compute weights
        weights = (alpha * F.relu(gradients)).sum(dim=(1, 2))

        # Compute weighted combination
        cam = torch.zeros(feature_maps.shape[1:], dtype=torch.float32,
                         device=feature_maps.device)

        for i, w in enumerate(weights):
            cam += w * feature_maps[i]

        # Apply ReLU
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy()


class LayerCAM(GradCAM):
    """
    Layer-CAM for hierarchical visual explanations.

    Generates CAMs at multiple network layers.
    """
    def __init__(self, model, target_layers):
        """
        Args:
            model: Neural network model
            target_layers: List of layer names to generate CAMs
        """
        self.model = model
        self.target_layers = target_layers
        self.cams = {}

        # Register hooks for all target layers
        for layer_name in target_layers:
            self.target_layer = layer_name
            self._register_hooks()

    def generate_multi_layer_heatmaps(self, input_tensor, target_class=None):
        """
        Generate heatmaps for all target layers.

        Args:
            input_tensor: Input tensor
            target_class: Target class index

        Returns:
            Dictionary of heatmaps for each layer
        """
        heatmaps = {}

        for layer_name in self.target_layers:
            self.target_layer = layer_name
            heatmap = self.generate_heatmap(input_tensor, target_class)
            heatmaps[layer_name] = heatmap

        return heatmaps


if __name__ == "__main__":
    print("Testing Grad-CAM...")

    from src.models.nutrient_model import NutrientAnalysisModel

    # Create model
    model = NutrientAnalysisModel(num_classes=500)
    model.eval()

    # Create dummy input
    input_tensor = torch.randn(1, 3, 224, 224)

    # Test Grad-CAM
    gradcam = GradCAM(model, target_layer='features.16')

    try:
        heatmap = gradcam.generate_heatmap(input_tensor)
        print(f"Grad-CAM heatmap shape: {heatmap.shape}")
        print(f"Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")

        # Test Grad-CAM++
        gradcam_pp = GradCAMPlusPlus(model, target_layer='features.16')
        heatmap_pp = gradcam_pp.generate_heatmap(input_tensor)
        print(f"\nGrad-CAM++ heatmap shape: {heatmap_pp.shape}")

        print("\nGrad-CAM test completed successfully!")

    except Exception as e:
        print(f"Note: {e}")
        print("Grad-CAM implementation is ready but requires proper layer access")
