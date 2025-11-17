"""
TensorFlow Lite conversion for mobile deployment.

Converts PyTorch models to TFLite format for optimized mobile inference.
"""

import torch
import torch.onnx
import tensorflow as tf
import numpy as np
from pathlib import Path
import onnx
from onnx_tf.backend import prepare


def convert_to_tflite(pytorch_model, input_shape=(1, 3, 224, 224),
                     output_path='model.tflite', quantize=True,
                     representative_dataset=None):
    """
    Convert PyTorch model to TensorFlow Lite format.

    Pipeline: PyTorch → ONNX → TensorFlow → TFLite

    Args:
        pytorch_model: PyTorch model to convert
        input_shape: Input tensor shape
        output_path: Path to save TFLite model
        quantize: Whether to apply quantization
        representative_dataset: Representative dataset for quantization

    Returns:
        Path to saved TFLite model
    """
    print("Converting PyTorch model to TensorFlow Lite...")

    # Step 1: PyTorch → ONNX
    print("\n1. Converting to ONNX...")
    onnx_path = Path(output_path).with_suffix('.onnx')
    pytorch_to_onnx(pytorch_model, onnx_path, input_shape)

    # Step 2: ONNX → TensorFlow
    print("\n2. Converting ONNX to TensorFlow...")
    tf_model_dir = Path(output_path).parent / 'tf_model'
    onnx_to_tensorflow(onnx_path, tf_model_dir)

    # Step 3: TensorFlow → TFLite
    print("\n3. Converting TensorFlow to TFLite...")
    tflite_path = tensorflow_to_tflite(
        tf_model_dir,
        output_path,
        quantize=quantize,
        representative_dataset=representative_dataset
    )

    print(f"\nConversion completed! TFLite model saved to: {tflite_path}")

    return tflite_path


def pytorch_to_onnx(model, output_path, input_shape=(1, 3, 224, 224)):
    """
    Convert PyTorch model to ONNX format.

    Args:
        model: PyTorch model
        output_path: Path to save ONNX model
        input_shape: Input tensor shape

    Returns:
        Path to ONNX model
    """
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(*input_shape)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"ONNX model saved to: {output_path}")

    # Verify ONNX model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed!")

    return output_path


def onnx_to_tensorflow(onnx_path, output_dir):
    """
    Convert ONNX model to TensorFlow SavedModel format.

    Args:
        onnx_path: Path to ONNX model
        output_dir: Directory to save TensorFlow model

    Returns:
        Path to TensorFlow model directory
    """
    # Load ONNX model
    onnx_model = onnx.load(str(onnx_path))

    # Convert to TensorFlow
    tf_rep = prepare(onnx_model)

    # Export to SavedModel
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tf_rep.export_graph(str(output_dir))

    print(f"TensorFlow model saved to: {output_dir}")

    return output_dir


def tensorflow_to_tflite(tf_model_dir, output_path, quantize=True,
                         representative_dataset=None):
    """
    Convert TensorFlow SavedModel to TFLite format.

    Args:
        tf_model_dir: Directory containing TensorFlow SavedModel
        output_path: Path to save TFLite model
        quantize: Whether to apply quantization
        representative_dataset: Generator function for representative data

    Returns:
        Path to TFLite model
    """
    # Create converter
    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_dir))

    # Set optimization flags
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Use representative dataset for full integer quantization
        if representative_dataset is not None:
            converter.representative_dataset = representative_dataset

            # Enable full integer quantization
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

    # Additional optimizations
    converter.experimental_new_converter = True
    converter.experimental_new_quantizer = True

    # Convert model
    tflite_model = converter.convert()

    # Save TFLite model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    # Get model size
    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"TFLite model size: {size_mb:.2f} MB")

    return output_path


def create_representative_dataset(calibration_loader, num_samples=100):
    """
    Create representative dataset generator for quantization calibration.

    Args:
        calibration_loader: Data loader with calibration data
        num_samples: Number of samples to use

    Returns:
        Generator function for representative dataset
    """
    def representative_dataset_gen():
        count = 0
        for batch in calibration_loader:
            if count >= num_samples:
                break

            images = batch['image'].numpy()

            # Yield each image in batch
            for img in images:
                if count >= num_samples:
                    break

                # Add batch dimension and convert to float32
                yield [img[np.newaxis, ...].astype(np.float32)]
                count += 1

    return representative_dataset_gen


def optimize_for_mobile(tflite_path, optimization_level='high'):
    """
    Apply mobile-specific optimizations to TFLite model.

    Args:
        tflite_path: Path to TFLite model
        optimization_level: 'low', 'medium', or 'high'

    Returns:
        Path to optimized model
    """
    print(f"Applying {optimization_level} level optimizations...")

    # Load TFLite model
    with open(tflite_path, 'rb') as f:
        tflite_model = f.read()

    # Create interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"\nModel details:")
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")

    # Apply optimizations based on level
    if optimization_level == 'high':
        # Further optimize model using TFLite optimization tools
        # This is a placeholder - actual implementation depends on TFLite version

        optimized_path = Path(tflite_path).with_suffix('.optimized.tflite')

        # Copy for now (actual optimization would be applied here)
        with open(optimized_path, 'wb') as f:
            f.write(tflite_model)

        print(f"Optimized model saved to: {optimized_path}")
        return optimized_path

    return tflite_path


def benchmark_tflite_model(tflite_path, test_images, num_runs=100):
    """
    Benchmark TFLite model inference time.

    Args:
        tflite_path: Path to TFLite model
        test_images: Test images for benchmarking
        num_runs: Number of inference runs

    Returns:
        Dictionary with benchmark results
    """
    import time

    print(f"Benchmarking TFLite model with {num_runs} runs...")

    # Load model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Warm up
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], test_images[0:1])
        interpreter.invoke()

    # Benchmark
    times = []
    for i in range(num_runs):
        start_time = time.time()

        interpreter.set_tensor(input_details[0]['index'], test_images[i % len(test_images):i % len(test_images) + 1])
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])

        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to ms

    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print(f"\nBenchmark Results:")
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Std deviation: {std_time:.2f} ms")
    print(f"Min time: {min_time:.2f} ms")
    print(f"Max time: {max_time:.2f} ms")

    return {
        'average_ms': avg_time,
        'std_ms': std_time,
        'min_ms': min_time,
        'max_ms': max_time
    }


def verify_tflite_accuracy(tflite_path, pytorch_model, test_loader, num_samples=100):
    """
    Verify TFLite model accuracy against PyTorch model.

    Args:
        tflite_path: Path to TFLite model
        pytorch_model: Original PyTorch model
        test_loader: Test data loader
        num_samples: Number of samples to verify

    Returns:
        Dictionary with accuracy comparison
    """
    print(f"Verifying TFLite model accuracy on {num_samples} samples...")

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Compare predictions
    pytorch_model.eval()
    matches = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if total >= num_samples:
                break

            images = batch['image']

            # PyTorch prediction
            pytorch_out = pytorch_model(images)
            if isinstance(pytorch_out, dict):
                pytorch_pred = pytorch_out['food_logits'].argmax(dim=1)
            else:
                pytorch_pred = pytorch_out.argmax(dim=1)

            # TFLite prediction
            for i in range(images.size(0)):
                if total >= num_samples:
                    break

                img_np = images[i:i+1].numpy()
                interpreter.set_tensor(input_details[0]['index'], img_np)
                interpreter.invoke()
                tflite_out = interpreter.get_tensor(output_details[0]['index'])
                tflite_pred = np.argmax(tflite_out, axis=1)[0]

                if pytorch_pred[i].item() == tflite_pred:
                    matches += 1

                total += 1

    agreement = matches / total if total > 0 else 0

    print(f"\nVerification Results:")
    print(f"Predictions agreement: {agreement:.2%}")
    print(f"Matches: {matches}/{total}")

    return {
        'agreement': agreement,
        'matches': matches,
        'total': total
    }


if __name__ == "__main__":
    print("TFLite converter module loaded successfully!")
    print("\nNote: Full conversion requires PyTorch model and test data")
    print("Use scripts/optimize_mobile.py for actual conversion")
