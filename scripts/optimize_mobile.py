#!/usr/bin/env python3
"""
Mobile optimization script for nutrient analysis model.

Performs quantization and TFLite conversion.

Usage:
    python scripts/optimize_mobile.py --model checkpoints/best_model.pth --output mobile_models/
"""

import argparse
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.nutrient_model import NutrientAnalysisModel
from src.optimization.quantization import (
    quantize_model,
    compare_model_sizes,
    evaluate_quantization_accuracy
)
from src.optimization.tflite_converter import convert_to_tflite, benchmark_tflite_model
from src.data.dataset import create_data_loaders
from src.utils.helpers import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Optimize model for mobile deployment')

    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default='./mobile_models',
                       help='Output directory for optimized models')
    parser.add_argument('--quantization', type=str, default='static',
                       choices=['dynamic', 'static', 'qat'],
                       help='Quantization type')
    parser.add_argument('--tflite', action='store_true',
                       help='Convert to TensorFlow Lite')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to dataset for calibration')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu/cuda)')

    return parser.parse_args()


def main():
    """Main optimization function."""
    args = parse_args()

    # Setup
    logger = setup_logging()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("Mobile Model Optimization")
    logger.info("="*80)

    # Load model
    logger.info(f"\nLoading model from: {args.model}")
    model = NutrientAnalysisModel.from_pretrained(args.model)
    model.eval()

    total_params, _ = sum(p.numel() for p in model.parameters()), \
                      sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,}")

    # Quantization
    logger.info(f"\nApplying {args.quantization} quantization...")

    calibration_loader = None
    if args.quantization == 'static' and args.data_dir:
        logger.info("Loading calibration data...")
        data_loaders = create_data_loaders(
            data_dir=args.data_dir,
            batch_size=32,
            num_workers=2
        )
        calibration_loader = data_loaders['train']

    quantized_model = quantize_model(
        model,
        quantization_type=args.quantization,
        calibration_data=calibration_loader
    )

    # Compare model sizes
    logger.info("\nComparing model sizes...")
    size_info = compare_model_sizes(
        model,
        quantized_model,
        save_dir=output_dir
    )

    # Save quantized model
    quantized_path = output_dir / 'quantized_model.pth'
    torch.save(quantized_model.state_dict(), quantized_path)
    logger.info(f"\nQuantized model saved to: {quantized_path}")

    # TensorFlow Lite conversion
    if args.tflite:
        logger.info("\nConverting to TensorFlow Lite...")
        tflite_path = output_dir / 'model.tflite'

        try:
            convert_to_tflite(
                pytorch_model=quantized_model,
                output_path=str(tflite_path),
                quantize=True
            )

            logger.info(f"TFLite model saved to: {tflite_path}")

            # Benchmark TFLite model
            logger.info("\nBenchmarking TFLite model...")
            test_images = torch.randn(10, 3, 224, 224).numpy()
            benchmark_results = benchmark_tflite_model(
                tflite_path,
                test_images,
                num_runs=100
            )

        except Exception as e:
            logger.error(f"TFLite conversion failed: {e}")
            logger.info("Continuing without TFLite conversion...")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("Optimization Summary")
    logger.info("="*80)
    logger.info(f"Original size: {size_info['original_size_mb']:.2f} MB")
    logger.info(f"Quantized size: {size_info['quantized_size_mb']:.2f} MB")
    logger.info(f"Compression ratio: {size_info['compression_ratio']:.2f}x")
    logger.info(f"Size reduction: {size_info['size_reduction_percent']:.1f}%")
    logger.info(f"\nOptimized models saved to: {output_dir}")


if __name__ == '__main__':
    main()
