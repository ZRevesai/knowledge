"""Helper utility functions."""

import random
import numpy as np
import torch
import json
import logging
from pathlib import Path


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(config=None):
    """
    Setup logging configuration.

    Args:
        config: Logging configuration dictionary

    Returns:
        Logger instance
    """
    if config is None:
        config = {}

    log_level = config.get('level', 'INFO')
    log_format = config.get('format', '%(asctime)s - %(levelname)s - %(message)s')

    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.get('file', 'training.log'))
        ] if config.get('file') else [logging.StreamHandler()]
    )

    return logging.getLogger(__name__)


def save_json(data, filepath):
    """
    Save data to JSON file.

    Args:
        data: Data to save (must be JSON serializable)
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    """
    Load data from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def count_parameters(model):
    """
    Count total and trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_device(prefer_cuda=True):
    """
    Get available device.

    Args:
        prefer_cuda: Whether to prefer CUDA if available

    Returns:
        Device string
    """
    if prefer_cuda and torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
