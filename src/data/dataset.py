"""
PyTorch Dataset and DataLoader utilities for nutritional data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Optional, Tuple


class NutritionalDataset(Dataset):
    """
    PyTorch Dataset for nutritional profiles.

    Wraps preprocessed nutritional data for efficient batching and loading.
    """

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 feature_names: Optional[list] = None,
                 micronutrient_names: Optional[list] = None):
        """
        Initialize dataset.

        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Label array of shape (n_samples, n_micronutrients)
            feature_names: Names of features
            micronutrient_names: Names of micronutrients
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.feature_names = feature_names
        self.micronutrient_names = micronutrient_names

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (features, labels)
        """
        return self.X[idx], self.y[idx]

    def get_feature_names(self) -> Optional[list]:
        """Get feature names."""
        return self.feature_names

    def get_micronutrient_names(self) -> Optional[list]:
        """Get micronutrient names."""
        return self.micronutrient_names

    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        return {
            'n_samples': len(self),
            'n_features': self.X.shape[1],
            'n_micronutrients': self.y.shape[1],
            'deficiency_rates': self.y.mean(dim=0).numpy(),
            'feature_means': self.X.mean(dim=0).numpy(),
            'feature_stds': self.X.std(dim=0).numpy()
        }


def create_dataloaders(
    data_splits: Dict[str, np.ndarray],
    batch_size: int = 64,
    num_workers: int = 4,
    feature_names: Optional[list] = None,
    micronutrient_names: Optional[list] = None
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.

    Args:
        data_splits: Dictionary with 'X_train', 'y_train', etc.
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        feature_names: Names of features
        micronutrient_names: Names of micronutrients

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    # Create datasets
    train_dataset = NutritionalDataset(
        data_splits['X_train'],
        data_splits['y_train'],
        feature_names,
        micronutrient_names
    )

    val_dataset = NutritionalDataset(
        data_splits['X_val'],
        data_splits['y_val'],
        feature_names,
        micronutrient_names
    )

    test_dataset = NutritionalDataset(
        data_splits['X_test'],
        data_splits['y_test'],
        feature_names,
        micronutrient_names
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


if __name__ == "__main__":
    print("Testing Nutritional Dataset...")

    # Create dummy data
    X = np.random.randn(1000, 105)
    y = np.random.randint(0, 2, (1000, 12)).astype(float)

    # Create dataset
    dataset = NutritionalDataset(X, y)
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample: {dataset[0][0].shape}, {dataset[0][1].shape}")

    # Get statistics
    stats = dataset.get_statistics()
    print(f"\nDataset statistics:")
    print(f"  Samples: {stats['n_samples']}")
    print(f"  Features: {stats['n_features']}")
    print(f"  Micronutrients: {stats['n_micronutrients']}")
    print(f"  Avg deficiency rate: {stats['deficiency_rates'].mean():.3f}")

    # Create splits
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    data_splits = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }

    # Create dataloaders
    dataloaders = create_dataloaders(data_splits, batch_size=32)
    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(dataloaders['train'])}")
    print(f"  Val batches: {len(dataloaders['val'])}")
    print(f"  Test batches: {len(dataloaders['test'])}")

    print("\nDataset test completed!")
