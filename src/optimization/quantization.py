"""
Model quantization for mobile deployment.

Implements 8-bit quantization (Equation 10) for model compression.

Reference: Equation (10)
q = round(r/s) + z
where q is quantized value, r is real value, s is scale factor, z is zero point
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from pathlib import Path
import copy


def quantize_model(model, quantization_type='static', calibration_data=None):
    """
    Quantize model for mobile deployment.

    Args:
        model: PyTorch model to quantize
        quantization_type: 'dynamic', 'static', or 'qat' (quantization-aware training)
        calibration_data: Calibration data loader for static quantization

    Returns:
        Quantized model
    """
    if quantization_type == 'dynamic':
        return dynamic_quantization(model)
    elif quantization_type == 'static':
        if calibration_data is None:
            raise ValueError("Calibration data required for static quantization")
        return static_quantization(model, calibration_data)
    elif quantization_type == 'qat':
        return quantization_aware_training(model)
    else:
        raise ValueError(f"Unknown quantization type: {quantization_type}")


def dynamic_quantization(model):
    """
    Apply dynamic quantization to model.

    Dynamic quantization quantizes weights ahead of time but quantizes
    activations dynamically during inference.

    Args:
        model: PyTorch model

    Returns:
        Dynamically quantized model
    """
    print("Applying dynamic quantization...")

    # Make a copy of the model
    model_copy = copy.deepcopy(model)
    model_copy.eval()

    # Apply dynamic quantization to linear and LSTM layers
    quantized_model = quant.quantize_dynamic(
        model_copy,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )

    print("Dynamic quantization completed!")

    return quantized_model


def static_quantization(model, calibration_loader, num_calibration_batches=100):
    """
    Apply static quantization with calibration.

    Reference: Equation (10)
    q = round(r/s) + z

    Static quantization quantizes both weights and activations using
    calibration data to determine quantization parameters (scale s and zero-point z).

    Args:
        model: PyTorch model
        calibration_loader: Data loader for calibration
        num_calibration_batches: Number of batches to use for calibration

    Returns:
        Statically quantized model
    """
    print("Applying static quantization...")

    # Make a copy of the model
    model_copy = copy.deepcopy(model)
    model_copy.eval()

    # Fuse modules (Conv + BN + ReLU)
    model_copy = fuse_modules(model_copy)

    # Prepare model for quantization
    model_copy.qconfig = quant.get_default_qconfig('fbgemm')
    quant.prepare(model_copy, inplace=True)

    # Calibrate with representative dataset
    print(f"Calibrating with {num_calibration_batches} batches...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(calibration_loader):
            if batch_idx >= num_calibration_batches:
                break

            images = batch['image']
            _ = model_copy(images)

            if (batch_idx + 1) % 10 == 0:
                print(f"Calibrated {batch_idx + 1}/{num_calibration_batches} batches")

    # Convert to quantized model
    quantized_model = quant.convert(model_copy, inplace=False)

    print("Static quantization completed!")

    return quantized_model


def quantization_aware_training(model):
    """
    Prepare model for quantization-aware training.

    QAT simulates quantization during training for better accuracy.

    Args:
        model: PyTorch model

    Returns:
        Model prepared for QAT
    """
    print("Preparing model for quantization-aware training...")

    # Fuse modules
    model = fuse_modules(model)

    # Set QAT configuration
    model.qconfig = quant.get_default_qat_qconfig('fbgemm')

    # Prepare for QAT
    quant.prepare_qat(model, inplace=True)

    print("Model ready for QAT!")
    print("Train the model, then convert with quant.convert(model)")

    return model


def fuse_modules(model):
    """
    Fuse Conv + BN + ReLU modules for better quantization.

    Args:
        model: PyTorch model

    Returns:
        Model with fused modules
    """
    # List of module patterns to fuse
    # This is a simplified version - actual implementation depends on model architecture

    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            # Try to fuse conv-bn-relu patterns
            try:
                quant.fuse_modules(
                    module,
                    [['0', '1', '2']],  # Conv, BN, ReLU indices
                    inplace=True
                )
            except:
                pass

    return model


def compare_model_sizes(original_model, quantized_model, save_dir='./models'):
    """
    Compare sizes of original and quantized models.

    Args:
        original_model: Original PyTorch model
        quantized_model: Quantized model
        save_dir: Directory to save models

    Returns:
        Dictionary with size information
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save models
    original_path = save_dir / 'original_model.pth'
    quantized_path = save_dir / 'quantized_model.pth'

    torch.save(original_model.state_dict(), original_path)
    torch.save(quantized_model.state_dict(), quantized_path)

    # Get file sizes
    original_size = original_path.stat().st_size / (1024 ** 2)  # MB
    quantized_size = quantized_path.stat().st_size / (1024 ** 2)  # MB

    compression_ratio = original_size / quantized_size

    print(f"\nModel Size Comparison:")
    print(f"Original model: {original_size:.2f} MB")
    print(f"Quantized model: {quantized_size:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Size reduction: {(1 - quantized_size/original_size) * 100:.1f}%")

    return {
        'original_size_mb': original_size,
        'quantized_size_mb': quantized_size,
        'compression_ratio': compression_ratio,
        'size_reduction_percent': (1 - quantized_size/original_size) * 100
    }


def evaluate_quantization_accuracy(original_model, quantized_model,
                                   test_loader, device='cpu'):
    """
    Compare accuracy of original vs quantized model.

    Args:
        original_model: Original model
        quantized_model: Quantized model
        test_loader: Test data loader
        device: Device to run evaluation on

    Returns:
        Dictionary with accuracy comparison
    """
    def evaluate(model, loader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                outputs = model(images)

                if isinstance(outputs, dict):
                    predictions = outputs['food_logits']
                else:
                    predictions = outputs

                _, predicted = torch.max(predictions, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    print("Evaluating models...")

    original_acc = evaluate(original_model, test_loader)
    quantized_acc = evaluate(quantized_model, test_loader)

    accuracy_drop = (original_acc - quantized_acc) * 100

    print(f"\nAccuracy Comparison:")
    print(f"Original model: {original_acc:.4f}")
    print(f"Quantized model: {quantized_acc:.4f}")
    print(f"Accuracy drop: {accuracy_drop:.2f}%")

    return {
        'original_accuracy': original_acc,
        'quantized_accuracy': quantized_acc,
        'accuracy_drop_percent': accuracy_drop
    }


class QuantizationConfig:
    """
    Configuration for quantization parameters.

    Based on paper specifications:
    - 8-bit quantization (qint8)
    - Per-channel quantization
    - Post-training static quantization
    """
    def __init__(self):
        self.dtype = torch.qint8
        self.per_channel = True
        self.reduce_range = False

        # Calibration settings
        self.num_calibration_batches = 100
        self.calibration_method = 'histogram'  # or 'percentile'

    def get_qconfig(self, backend='fbgemm'):
        """
        Get quantization configuration.

        Args:
            backend: 'fbgemm' for x86 or 'qnnpack' for ARM

        Returns:
            QConfig object
        """
        if backend == 'fbgemm':
            return quant.get_default_qconfig('fbgemm')
        elif backend == 'qnnpack':
            return quant.get_default_qconfig('qnnpack')
        else:
            raise ValueError(f"Unknown backend: {backend}")


if __name__ == "__main__":
    print("Testing quantization...")

    from src.models.nutrient_model import NutrientAnalysisModel

    # Create model
    model = NutrientAnalysisModel(num_classes=500)
    model.eval()

    # Test dynamic quantization
    print("\n1. Testing dynamic quantization...")
    quantized_dyn = dynamic_quantization(model)

    # Count parameters
    original_params = sum(p.numel() for p in model.parameters())
    quantized_params = sum(p.numel() for p in quantized_dyn.parameters())

    print(f"Original parameters: {original_params:,}")
    print(f"Quantized parameters: {quantized_params:,}")

    # Test model size comparison
    print("\n2. Comparing model sizes...")
    size_info = compare_model_sizes(model, quantized_dyn)

    print("\nQuantization test completed successfully!")
