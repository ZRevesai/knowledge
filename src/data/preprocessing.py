"""
Data preprocessing for nutritional profiles.

Handles:
- Feature scaling and normalization
- Missing value imputation
- Feature engineering
- Data splitting
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import warnings


class NutritionalDataPreprocessor:
    """
    Comprehensive preprocessor for elderly nutritional data.

    Handles the preprocessing steps needed for NHANES-style nutritional data.
    """

    def __init__(self,
                 scaling_method: str = 'standard',
                 imputation_method: str = 'knn',
                 handle_outliers: bool = True,
                 feature_engineering: bool = True,
                 random_state: int = 42):
        """
        Initialize the preprocessor.

        Args:
            scaling_method: Method for feature scaling ('standard', 'robust', 'minmax')
            imputation_method: Method for missing value imputation ('mean', 'median', 'knn')
            handle_outliers: Whether to handle outliers
            feature_engineering: Whether to create engineered features
            random_state: Random seed for reproducibility
        """
        self.scaling_method = scaling_method
        self.imputation_method = imputation_method
        self.handle_outliers = handle_outliers
        self.feature_engineering = feature_engineering
        self.random_state = random_state

        # Initialize scalers and imputers
        self._initialize_transformers()

        # Feature metadata
        self.feature_names = None
        self.feature_categories = None
        self.micronutrient_names = None

    def _initialize_transformers(self):
        """Initialize scaling and imputation transformers."""
        # Scaler
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

        # Imputer
        if self.imputation_method == 'mean':
            self.imputer = SimpleImputer(strategy='mean')
        elif self.imputation_method == 'median':
            self.imputer = SimpleImputer(strategy='median')
        elif self.imputation_method == 'knn':
            self.imputer = KNNImputer(n_neighbors=5)
        else:
            self.imputer = SimpleImputer(strategy='mean')

    def generate_synthetic_data(self,
                               n_samples: int = 5000,
                               n_features: int = 105,
                               n_micronutrients: int = 12) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic nutritional data for testing.

        Simulates NHANES-style elderly nutritional profiles.

        Args:
            n_samples: Number of samples to generate
            n_features: Number of input features
            n_micronutrients: Number of micronutrient deficiency labels

        Returns:
            Tuple of (features_df, labels_df)
        """
        np.random.seed(self.random_state)

        # Define feature categories and their sizes
        categories = {
            'dietary_intake': 50,
            'anthropometric': 8,
            'biochemical': 15,
            'medical_history': 20,
            'functional_assessment': 12
        }

        # Generate features by category
        features_dict = {}
        feature_names = []

        # 1. Dietary intake features
        for i in range(categories['dietary_intake']):
            nutrient = ['Protein', 'Carbs', 'Fat', 'Fiber', 'VitaminD', 'VitaminB12',
                       'VitaminB6', 'Folate', 'Iron', 'Calcium', 'Magnesium', 'Zinc',
                       'VitaminC', 'VitaminE', 'VitaminA', 'Selenium'][i % 16]
            feature_name = f'dietary_{nutrient}_intake_{i}'
            # Simulate dietary intake with some elderly-specific patterns
            features_dict[feature_name] = np.random.gamma(2, 50, n_samples) * (1 - 0.3 * np.random.rand(n_samples))
            feature_names.append(feature_name)

        # 2. Anthropometric measurements
        anthro_features = ['BMI', 'Weight', 'Height', 'WaistCircumference',
                          'HipCircumference', 'ArmCircumference', 'CalfCircumference', 'HandGripStrength']
        for feat in anthro_features:
            if feat == 'BMI':
                features_dict[feat] = np.random.normal(27, 5, n_samples).clip(15, 45)
            elif feat == 'Weight':
                features_dict[feat] = np.random.normal(70, 15, n_samples).clip(40, 120)
            elif feat == 'Height':
                features_dict[feat] = np.random.normal(165, 10, n_samples).clip(140, 190)
            elif 'Circumference' in feat:
                features_dict[feat] = np.random.normal(90, 15, n_samples).clip(50, 150)
            else:
                features_dict[feat] = np.random.normal(25, 8, n_samples).clip(5, 50)
            feature_names.append(feat)

        # 3. Biochemical markers
        biochem_features = ['SerumB12', 'SerumD', 'SerumB6', 'SerumFolate',
                           'SerumIron', 'SerumCalcium', 'SerumMagnesium', 'SerumZinc',
                           'SerumC', 'SerumE', 'SerumA', 'SerumSelenium',
                           'Hemoglobin', 'Albumin', 'CRP']
        for feat in biochem_features:
            if 'Serum' in feat:
                # Simulate serum levels with deficiency patterns
                features_dict[feat] = np.random.gamma(3, 30, n_samples) * (1 - 0.4 * np.random.rand(n_samples))
            else:
                features_dict[feat] = np.random.normal(12, 2, n_samples).clip(5, 20)
            feature_names.append(feat)

        # 4. Medical history (binary and categorical)
        for i in range(categories['medical_history']):
            feature_name = f'medical_condition_{i}'
            features_dict[feature_name] = np.random.binomial(1, 0.3, n_samples)
            feature_names.append(feature_name)

        # 5. Functional assessments
        func_features = ['ADL_Bathing', 'ADL_Dressing', 'ADL_Eating', 'ADL_Toileting',
                        'Mobility_Walking', 'Mobility_Stairs', 'Cognitive_Memory',
                        'Cognitive_Attention', 'Social_Interaction', 'Depression_Score',
                        'Fatigue_Level', 'AppetiteLoss']
        for feat in func_features:
            # Functional scores (0-10 scale)
            features_dict[feat] = np.random.poisson(3, n_samples).clip(0, 10)
            feature_names.append(feat)

        # Create features DataFrame
        features_df = pd.DataFrame(features_dict)

        # Introduce some missing values (realistic for elderly populations)
        missing_mask = np.random.rand(*features_df.shape) < 0.05
        features_df[missing_mask] = np.nan

        # Generate labels (micronutrient deficiencies)
        micronutrient_names = [
            'Vitamin_B12', 'Vitamin_D', 'Vitamin_B6', 'Folate',
            'Iron', 'Calcium', 'Magnesium', 'Zinc',
            'Vitamin_C', 'Vitamin_E', 'Vitamin_A', 'Selenium'
        ]

        labels_dict = {}
        for i, nutrient in enumerate(micronutrient_names):
            # Create deficiency patterns based on related features
            serum_col = f'Serum{nutrient.split("_")[-1]}'
            dietary_cols = [col for col in features_df.columns if nutrient.split("_")[-1] in col]

            if serum_col in features_df.columns:
                # Base deficiency on serum levels
                serum_values = features_df[serum_col].fillna(features_df[serum_col].mean())
                threshold = np.percentile(serum_values, 30)
                base_deficiency = (serum_values < threshold).astype(int)
            else:
                base_deficiency = np.random.binomial(1, 0.25, n_samples)

            # Add some noise
            labels_dict[nutrient] = (base_deficiency + np.random.binomial(1, 0.1, n_samples)) % 2

        labels_df = pd.DataFrame(labels_dict)

        # Store metadata
        self.feature_names = feature_names
        self.micronutrient_names = micronutrient_names
        self.feature_categories = self._categorize_features(feature_names)

        return features_df, labels_df

    def _categorize_features(self, feature_names: List[str]) -> Dict[str, List[int]]:
        """
        Categorize features by type.

        Args:
            feature_names: List of feature names

        Returns:
            Dictionary mapping category names to feature indices
        """
        categories = {
            'dietary_intake': [],
            'anthropometric': [],
            'biochemical': [],
            'medical_history': [],
            'functional_assessment': []
        }

        for idx, name in enumerate(feature_names):
            name_lower = name.lower()
            if 'dietary' in name_lower or 'intake' in name_lower:
                categories['dietary_intake'].append(idx)
            elif any(kw in name_lower for kw in ['bmi', 'weight', 'height', 'circumference', 'grip']):
                categories['anthropometric'].append(idx)
            elif 'serum' in name_lower or 'hemoglobin' in name_lower or 'albumin' in name_lower or 'crp' in name_lower:
                categories['biochemical'].append(idx)
            elif 'medical' in name_lower or 'condition' in name_lower:
                categories['medical_history'].append(idx)
            elif 'adl' in name_lower or 'mobility' in name_lower or 'cognitive' in name_lower or any(kw in name_lower for kw in ['social', 'depression', 'fatigue', 'appetite']):
                categories['functional_assessment'].append(idx)

        return {k: v for k, v in categories.items() if v}

    def handle_missing_values(self, X: np.ndarray) -> np.ndarray:
        """
        Handle missing values using configured imputation method.

        Args:
            X: Feature matrix with potential missing values

        Returns:
            Feature matrix with imputed values
        """
        return self.imputer.fit_transform(X)

    def handle_outliers_func(self, X: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Handle outliers using z-score method.

        Args:
            X: Feature matrix
            threshold: Z-score threshold for outlier detection

        Returns:
            Feature matrix with outliers clipped
        """
        z_scores = np.abs((X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8))
        outliers = z_scores > threshold

        # Clip outliers to threshold
        X_clipped = X.copy()
        for col in range(X.shape[1]):
            col_outliers = outliers[:, col]
            if col_outliers.any():
                lower = np.percentile(X[~col_outliers, col], 1)
                upper = np.percentile(X[~col_outliers, col], 99)
                X_clipped[col_outliers, col] = np.clip(X[col_outliers, col], lower, upper)

        return X_clipped

    def engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features based on nutritional knowledge.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with additional engineered features
        """
        X_eng = X.copy()

        # BMI-related features
        if 'Weight' in X.columns and 'Height' in X.columns:
            if 'BMI' not in X.columns:
                X_eng['BMI'] = X['Weight'] / ((X['Height'] / 100) ** 2)

        # Nutritional ratios
        serum_cols = [col for col in X.columns if 'Serum' in col]
        dietary_cols = [col for col in X.columns if 'dietary' in col]

        if serum_cols and dietary_cols:
            # Average serum levels
            X_eng['avg_serum_level'] = X[serum_cols].mean(axis=1)

            # Average dietary intake
            X_eng['avg_dietary_intake'] = X[dietary_cols].mean(axis=1)

            # Ratio of serum to dietary (absorption indicator)
            X_eng['serum_dietary_ratio'] = X_eng['avg_serum_level'] / (X_eng['avg_dietary_intake'] + 1)

        # Functional composite scores
        adl_cols = [col for col in X.columns if 'ADL' in col]
        if adl_cols:
            X_eng['adl_composite'] = X[adl_cols].sum(axis=1)

        mobility_cols = [col for col in X.columns if 'Mobility' in col]
        if mobility_cols:
            X_eng['mobility_composite'] = X[mobility_cols].sum(axis=1)

        cognitive_cols = [col for col in X.columns if 'Cognitive' in col]
        if cognitive_cols:
            X_eng['cognitive_composite'] = X[cognitive_cols].sum(axis=1)

        return X_eng

    def fit_transform(self,
                     X: pd.DataFrame,
                     y: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit preprocessor and transform data.

        Args:
            X: Feature DataFrame
            y: Optional labels DataFrame

        Returns:
            Tuple of (transformed_features, transformed_labels)
        """
        # Store feature names
        if self.feature_names is None:
            self.feature_names = X.columns.tolist()

        # Feature engineering
        if self.feature_engineering:
            X = self.engineer_features(X)
            self.feature_names = X.columns.tolist()

        # Convert to numpy
        X_array = X.values

        # Handle missing values
        X_array = self.handle_missing_values(X_array)

        # Handle outliers
        if self.handle_outliers:
            X_array = self.handle_outliers_func(X_array)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_array)

        # Transform labels if provided
        y_array = y.values if y is not None else None

        return X_scaled, y_array

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.

        Args:
            X: Feature DataFrame

        Returns:
            Transformed feature array
        """
        # Feature engineering
        if self.feature_engineering:
            X = self.engineer_features(X)

        # Convert to numpy
        X_array = X.values

        # Handle missing values (using fitted imputer)
        X_array = self.imputer.transform(X_array)

        # Scale features (using fitted scaler)
        X_scaled = self.scaler.transform(X_array)

        return X_scaled

    def split_data(self,
                  X: np.ndarray,
                  y: np.ndarray,
                  train_size: float = 0.7,
                  val_size: float = 0.15,
                  test_size: float = 0.15,
                  stratify: bool = True) -> Dict[str, np.ndarray]:
        """
        Split data into train, validation, and test sets.

        Args:
            X: Feature array
            y: Label array
            train_size: Proportion for training
            val_size: Proportion for validation
            test_size: Proportion for testing
            stratify: Whether to stratify split based on labels

        Returns:
            Dictionary with train/val/test splits
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
            "Split proportions must sum to 1.0"

        # First split: separate test set
        stratify_col = y[:, 0] if stratify else None

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_col
        )

        # Second split: separate train and validation
        val_proportion = val_size / (train_size + val_size)
        stratify_col = y_temp[:, 0] if stratify else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_proportion,
            random_state=self.random_state,
            stratify=stratify_col
        )

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }


if __name__ == "__main__":
    print("Testing Nutritional Data Preprocessor...")

    # Initialize preprocessor
    preprocessor = NutritionalDataPreprocessor(random_state=42)

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    features_df, labels_df = preprocessor.generate_synthetic_data(n_samples=1000)
    print(f"Features shape: {features_df.shape}")
    print(f"Labels shape: {labels_df.shape}")
    print(f"\nFeature categories: {list(preprocessor.feature_categories.keys())}")

    # Preprocess data
    print("\nPreprocessing data...")
    X, y = preprocessor.fit_transform(features_df, labels_df)
    print(f"Processed features shape: {X.shape}")
    print(f"Processed labels shape: {y.shape}")

    # Split data
    print("\nSplitting data...")
    data_splits = preprocessor.split_data(X, y)
    print(f"Train: {data_splits['X_train'].shape}")
    print(f"Validation: {data_splits['X_val'].shape}")
    print(f"Test: {data_splits['X_test'].shape}")

    print("\nPreprocessor test completed!")
