"""
Data preprocessing utilities for NUTRINET.

Provides functions for:
- Feature normalization and scaling
- Train/validation/test splitting (80:20 as per paper)
- Missing value imputation
- Feature engineering for vulnerable populations
"""

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from typing import Dict, List, Tuple, Optional


class NutrientPreprocessor:
    """
    Preprocessor for nutrient data.

    Handles:
    - Missing value imputation
    - Feature scaling and normalization
    - Population-specific feature engineering
    """

    def __init__(
        self,
        scaler_type: str = 'standard',
        imputation_strategy: str = 'median',
    ):
        """
        Args:
            scaler_type: Type of scaler ('standard', 'robust', or 'minmax')
            imputation_strategy: Strategy for missing values ('mean', 'median', or 'most_frequent')
        """
        self.scaler_type = scaler_type
        self.imputation_strategy = imputation_strategy

        # Initialize scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()

        # Initialize imputer
        self.imputer = SimpleImputer(strategy=imputation_strategy)

        self.is_fitted = False

    def fit(self, X: np.ndarray):
        """
        Fit preprocessor to training data.

        Args:
            X: Training features [n_samples, n_features]
        """
        # Impute missing values
        X_imputed = self.imputer.fit_transform(X)

        # Fit scaler
        self.scaler.fit(X_imputed)

        self.is_fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features.

        Args:
            X: Features [n_samples, n_features]

        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Impute and scale
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)

        return X_scaled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit preprocessor and transform features.

        Args:
            X: Features [n_samples, n_features]

        Returns:
            Transformed features
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform features to original scale.

        Args:
            X_scaled: Scaled features

        Returns:
            Features in original scale
        """
        return self.scaler.inverse_transform(X_scaled)


def create_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
    stratify: Optional[np.ndarray] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Create train/validation/test splits.

    As per paper: 80:20 train:test split with 5-fold cross-validation

    Args:
        X: Features [n_samples, n_features]
        y: Labels [n_samples] or [n_samples, n_outputs]
        train_ratio: Ratio for training set (default: 0.8)
        val_ratio: Ratio for validation set (default: 0.1)
        test_ratio: Ratio for test set (default: 0.1)
        random_state: Random seed
        stratify: Optional stratification labels

    Returns:
        Dictionary with 'train', 'val', 'test' splits
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    # First split: train+val vs test
    test_size = test_ratio
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Second split: train vs val
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=random_state,
        stratify=None,
    )

    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test),
    }


def normalize_features(
    X: np.ndarray,
    method: str = 'standard',
) -> Tuple[np.ndarray, Dict]:
    """
    Normalize features.

    Args:
        X: Features [n_samples, n_features]
        method: Normalization method ('standard', 'minmax', or 'robust')

    Returns:
        Tuple of (normalized features, normalization parameters)
    """
    if method == 'standard':
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        X_norm = (X - mean) / std
        params = {'mean': mean, 'std': std}

    elif method == 'minmax':
        min_val = X.min(axis=0)
        max_val = X.max(axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
        params = {'min': min_val, 'max': max_val}

    elif method == 'robust':
        median = np.median(X, axis=0)
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        iqr = q75 - q25 + 1e-8
        X_norm = (X - median) / iqr
        params = {'median': median, 'iqr': iqr}

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_norm, params


def engineer_population_features(
    X: np.ndarray,
    population_type: str = 'elderly',
    age: Optional[np.ndarray] = None,
    demographics: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """
    Engineer features specific to vulnerable populations.

    Args:
        X: Base features [n_samples, n_features]
        population_type: Type of population ('elderly', 'pregnant', 'pediatric', 'chronic')
        age: Age values [n_samples]
        demographics: Additional demographic features

    Returns:
        Engineered features with population-specific adjustments
    """
    X_eng = X.copy()

    if population_type == 'elderly':
        # Adjust for age-related absorption changes
        if age is not None:
            # Calcium absorption decreases with age
            calcium_idx = 0  # Placeholder index
            absorption_factor = 1.0 - 0.005 * (age - 50).clip(0, None)
            X_eng[:, calcium_idx] = X[:, calcium_idx] * absorption_factor.squeeze()

    elif population_type == 'pregnant':
        # Increase requirements for certain nutrients
        if demographics and 'trimester' in demographics:
            trimester = demographics['trimester']
            # Folate requirements increase
            folate_idx = 1  # Placeholder
            requirement_factor = 1.0 + 0.5 * (trimester / 3)
            X_eng[:, folate_idx] = X[:, folate_idx] / requirement_factor.squeeze()

    elif population_type == 'pediatric':
        # Age-adjusted requirements
        if age is not None:
            # Scale by developmental stage
            growth_factor = 1.0 + 0.1 * (18 - age).clip(0, None) / 18
            X_eng = X * growth_factor.reshape(-1, 1)

    return X_eng


def create_cross_validation_folds(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create k-fold cross-validation splits.

    As per paper: 5-fold cross-validation

    Args:
        X: Features
        y: Labels
        n_folds: Number of folds (default: 5)
        random_state: Random seed

    Returns:
        List of (X_train, X_val, y_train, y_val) tuples
    """
    from sklearn.model_selection import KFold

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    folds = []
    for train_idx, val_idx in kfold.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        folds.append((X_train, X_val, y_train, y_val))

    return folds
