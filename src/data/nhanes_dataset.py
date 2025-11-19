"""
NHANES Dataset Preprocessing and Loading for SIDRM.

Implements data preprocessing for the National Health and Nutrition Examination Survey (NHANES)
dataset (2015-2018 cycles) with 15,560 participants across four vulnerable populations:
- Pregnant women (P)
- Elderly individuals (E)
- Children (C)
- Chronic disease patients (CD)

Reference: "Smart Interpretable Dietary Recommender Model for Vulnerable Populations"
by Zvinodashe Revesai and Okuthe P. Kogeda (2025)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings


class NHANESPreprocessor:
    """
    Preprocessor for NHANES dataset tailored for vulnerable populations.

    Handles:
    - Demographic features (age, gender, race/ethnicity)
    - Clinical measurements (biomarkers, anthropometric data)
    - Nutritional intake (24-hour dietary recalls, supplement usage)
    - Laboratory results (serum nutrient levels, metabolic markers)
    - Population categorization (P, E, C, CD)
    """

    POPULATION_CATEGORIES = {
        0: 'Pregnant',
        1: 'Elderly',
        2: 'Children',
        3: 'Chronic_Disease'
    }

    # Key micronutrients tracked (50 total nutrients in full model)
    PRIMARY_NUTRIENTS = [
        'Vitamin_B12', 'Vitamin_D', 'Vitamin_B6', 'Folate',
        'Iron', 'Calcium', 'Magnesium', 'Zinc',
        'Vitamin_C', 'Vitamin_E', 'Vitamin_A', 'Selenium',
        'Thiamin', 'Riboflavin', 'Niacin', 'Vitamin_K',
        'Phosphorus', 'Potassium', 'Sodium', 'Copper'
    ]

    def __init__(self,
                 feature_names: Optional[List[str]] = None,
                 nutrient_names: Optional[List[str]] = None,
                 scale_features: bool = True,
                 handle_missing: str = 'mean',
                 random_state: int = 42):
        """
        Initialize NHANES preprocessor.

        Args:
            feature_names: Names of input features
            nutrient_names: Names of nutrients to predict
            scale_features: Whether to standardize features
            handle_missing: How to handle missing values ('mean', 'median', 'drop')
            random_state: Random seed for reproducibility
        """
        self.feature_names = feature_names
        self.nutrient_names = nutrient_names or self.PRIMARY_NUTRIENTS
        self.scale_features = scale_features
        self.handle_missing = handle_missing
        self.random_state = random_state

        self.scaler = StandardScaler() if scale_features else None
        self.label_encoders = {}
        self.feature_statistics = {}

    def generate_synthetic_nhanes_data(self,
                                      n_samples: int = 15560,
                                      train_ratio: float = 0.7,
                                      val_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic NHANES-style data for vulnerable populations.

        This simulates the real NHANES dataset structure (2015-2018 cycles).
        For production use, replace with actual NHANES data loading.

        Args:
            n_samples: Total number of participants (default: 15,560 as per paper)
            train_ratio: Proportion for training set (default: 0.7)
            val_ratio: Proportion for validation set (default: 0.15)

        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        np.random.seed(self.random_state)

        print(f"Generating synthetic NHANES data for {n_samples} participants...")

        # Population distribution
        # Pregnant: 15%, Elderly: 30%, Children: 25%, Chronic Disease: 30%
        population_probs = [0.15, 0.30, 0.25, 0.30]
        populations = np.random.choice(
            [0, 1, 2, 3],
            size=n_samples,
            p=population_probs
        )

        # Demographics
        ages = np.zeros(n_samples)
        for pop in range(4):
            mask = populations == pop
            if pop == 0:  # Pregnant women (18-45)
                ages[mask] = np.random.normal(28, 6, mask.sum()).clip(18, 45)
            elif pop == 1:  # Elderly (65+)
                ages[mask] = np.random.normal(72, 8, mask.sum()).clip(65, 95)
            elif pop == 2:  # Children (2-17)
                ages[mask] = np.random.normal(10, 4, mask.sum()).clip(2, 17)
            else:  # Chronic disease (any age, skewed older)
                ages[mask] = np.random.normal(55, 15, mask.sum()).clip(20, 85)

        genders = np.random.choice([0, 1], size=n_samples, p=[0.52, 0.48])  # 0=Female, 1=Male
        # Force pregnant to female
        genders[populations == 0] = 0

        races = np.random.choice(
            [0, 1, 2, 3, 4],  # White, Black, Hispanic, Asian, Other
            size=n_samples,
            p=[0.62, 0.13, 0.17, 0.06, 0.02]
        )

        # Anthropometric measurements
        heights = np.random.normal(165, 15, n_samples).clip(60, 200)  # cm
        weights = np.random.normal(75, 20, n_samples).clip(15, 200)  # kg
        bmis = weights / ((heights / 100) ** 2)

        # Clinical biomarkers (simulated)
        glucose = np.random.normal(100, 20, n_samples).clip(60, 200)  # mg/dL
        cholesterol = np.random.normal(195, 40, n_samples).clip(120, 300)  # mg/dL
        hdl = np.random.normal(55, 15, n_samples).clip(20, 100)  # mg/dL
        ldl = cholesterol - hdl - 30
        triglycerides = np.random.normal(130, 50, n_samples).clip(50, 400)  # mg/dL
        hba1c = np.random.normal(5.5, 1.0, n_samples).clip(4.0, 12.0)  # %

        # Blood pressure
        systolic_bp = np.random.normal(120, 18, n_samples).clip(90, 180)  # mmHg
        diastolic_bp = np.random.normal(80, 12, n_samples).clip(60, 120)  # mmHg

        # Serum nutrient levels (representative subset)
        serum_b12 = np.random.lognormal(5.7, 0.6, n_samples).clip(100, 2000)  # pg/mL
        serum_folate = np.random.lognormal(2.5, 0.8, n_samples).clip(2, 50)  # ng/mL
        serum_iron = np.random.normal(85, 30, n_samples).clip(20, 200)  # μg/dL
        serum_ferritin = np.random.lognormal(4.0, 1.0, n_samples).clip(10, 500)  # ng/mL
        serum_calcium = np.random.normal(9.5, 0.5, n_samples).clip(8.0, 11.0)  # mg/dL
        serum_vitamin_d = np.random.normal(25, 12, n_samples).clip(5, 80)  # ng/mL

        # Dietary intake (24-hour recall - daily values)
        # Energy and macronutrients
        energy_kcal = np.random.normal(2000, 600, n_samples).clip(500, 5000)
        protein_g = np.random.normal(80, 30, n_samples).clip(20, 250)
        carbs_g = np.random.normal(250, 80, n_samples).clip(50, 600)
        fat_g = np.random.normal(70, 30, n_samples).clip(20, 200)
        fiber_g = np.random.normal(18, 8, n_samples).clip(2, 60)

        # Micronutrients (dietary intake)
        vitamin_a_mcg = np.random.lognormal(6.5, 0.8, n_samples).clip(200, 5000)
        vitamin_c_mg = np.random.normal(85, 50, n_samples).clip(10, 500)
        vitamin_e_mg = np.random.normal(12, 6, n_samples).clip(2, 50)
        thiamin_mg = np.random.normal(1.5, 0.6, n_samples).clip(0.3, 5)
        riboflavin_mg = np.random.normal(1.8, 0.8, n_samples).clip(0.3, 6)
        niacin_mg = np.random.normal(22, 10, n_samples).clip(5, 60)
        vitamin_b6_mg = np.random.normal(1.9, 0.9, n_samples).clip(0.3, 8)
        folate_mcg = np.random.normal(450, 200, n_samples).clip(100, 1500)
        vitamin_b12_mcg = np.random.normal(5, 3, n_samples).clip(0.5, 30)
        calcium_mg = np.random.normal(950, 400, n_samples).clip(200, 2500)
        iron_mg = np.random.normal(15, 7, n_samples).clip(3, 50)
        magnesium_mg = np.random.normal(320, 120, n_samples).clip(80, 800)
        phosphorus_mg = np.random.normal(1200, 400, n_samples).clip(300, 3000)
        potassium_mg = np.random.normal(2800, 900, n_samples).clip(800, 6000)
        sodium_mg = np.random.normal(3400, 1200, n_samples).clip(500, 8000)
        zinc_mg = np.random.normal(11, 5, n_samples).clip(2, 40)
        selenium_mcg = np.random.normal(110, 40, n_samples).clip(20, 300)
        copper_mg = np.random.normal(1.2, 0.6, n_samples).clip(0.3, 5)

        # Supplement usage (binary indicators)
        multivitamin_use = np.random.binomial(1, 0.35, n_samples)
        vitamin_d_supplement = np.random.binomial(1, 0.25, n_samples)
        calcium_supplement = np.random.binomial(1, 0.20, n_samples)
        iron_supplement = np.random.binomial(1, 0.12, n_samples)

        # Physical activity (MET-minutes per week)
        physical_activity = np.random.lognormal(6.0, 1.5, n_samples).clip(0, 5000)

        # Lifestyle factors
        smoking_status = np.random.choice([0, 1, 2], n_samples, p=[0.65, 0.20, 0.15])  # Never, Former, Current
        alcohol_consumption = np.random.lognormal(1.5, 1.8, n_samples).clip(0, 100)  # drinks/week

        # Chronic conditions (binary)
        diabetes = np.random.binomial(1, 0.12, n_samples)
        hypertension = np.random.binomial(1, 0.28, n_samples)
        heart_disease = np.random.binomial(1, 0.08, n_samples)
        kidney_disease = np.random.binomial(1, 0.05, n_samples)

        # Increase chronic conditions for chronic disease population
        diabetes[populations == 3] = np.random.binomial(1, 0.45, (populations == 3).sum())
        hypertension[populations == 3] = np.random.binomial(1, 0.60, (populations == 3).sum())

        # Construct feature DataFrame
        features_df = pd.DataFrame({
            # Demographics
            'age': ages,
            'gender': genders,
            'race': races,
            'population_category': populations,

            # Anthropometrics
            'height_cm': heights,
            'weight_kg': weights,
            'bmi': bmis,

            # Clinical biomarkers
            'glucose_mg_dl': glucose,
            'total_cholesterol_mg_dl': cholesterol,
            'hdl_mg_dl': hdl,
            'ldl_mg_dl': ldl,
            'triglycerides_mg_dl': triglycerides,
            'hba1c_percent': hba1c,
            'systolic_bp_mmhg': systolic_bp,
            'diastolic_bp_mmhg': diastolic_bp,

            # Serum nutrients
            'serum_b12_pg_ml': serum_b12,
            'serum_folate_ng_ml': serum_folate,
            'serum_iron_ug_dl': serum_iron,
            'serum_ferritin_ng_ml': serum_ferritin,
            'serum_calcium_mg_dl': serum_calcium,
            'serum_vitamin_d_ng_ml': serum_vitamin_d,

            # Dietary intake
            'energy_kcal': energy_kcal,
            'protein_g': protein_g,
            'carbohydrate_g': carbs_g,
            'total_fat_g': fat_g,
            'fiber_g': fiber_g,
            'vitamin_a_mcg': vitamin_a_mcg,
            'vitamin_c_mg': vitamin_c_mg,
            'vitamin_e_mg': vitamin_e_mg,
            'thiamin_mg': thiamin_mg,
            'riboflavin_mg': riboflavin_mg,
            'niacin_mg': niacin_mg,
            'vitamin_b6_mg': vitamin_b6_mg,
            'folate_mcg': folate_mcg,
            'vitamin_b12_mcg': vitamin_b12_mcg,
            'calcium_mg': calcium_mg,
            'iron_mg': iron_mg,
            'magnesium_mg': magnesium_mg,
            'phosphorus_mg': phosphorus_mg,
            'potassium_mg': potassium_mg,
            'sodium_mg': sodium_mg,
            'zinc_mg': zinc_mg,
            'selenium_mcg': selenium_mcg,
            'copper_mg': copper_mg,

            # Supplements
            'multivitamin_use': multivitamin_use,
            'vitamin_d_supplement': vitamin_d_supplement,
            'calcium_supplement': calcium_supplement,
            'iron_supplement': iron_supplement,

            # Lifestyle
            'physical_activity_met_min': physical_activity,
            'smoking_status': smoking_status,
            'alcohol_drinks_week': alcohol_consumption,

            # Chronic conditions
            'diabetes': diabetes,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'kidney_disease': kidney_disease
        })

        # Generate deficiency labels (ground truth)
        labels_df = self._generate_deficiency_labels(features_df)

        # Split data
        train_val_df, test_df = train_test_split(
            features_df,
            test_size=1.0 - train_ratio - val_ratio,
            random_state=self.random_state,
            stratify=populations
        )

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio / (train_ratio + val_ratio),
            random_state=self.random_state,
            stratify=train_val_df['population_category']
        )

        # Corresponding labels
        train_labels = labels_df.loc[train_df.index]
        val_labels = labels_df.loc[val_df.index]
        test_labels = labels_df.loc[test_df.index]

        print(f"Data split:")
        print(f"  Training: {len(train_df)} samples ({len(train_df)/n_samples*100:.1f}%)")
        print(f"  Validation: {len(val_df)} samples ({len(val_df)/n_samples*100:.1f}%)")
        print(f"  Test: {len(test_df)} samples ({len(test_df)/n_samples*100:.1f}%)")

        return {
            'train': {'features': train_df, 'labels': train_labels},
            'val': {'features': val_df, 'labels': val_labels},
            'test': {'features': test_df, 'labels': test_labels}
        }

    def _generate_deficiency_labels(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate nutrient deficiency labels based on established thresholds.

        Uses serum levels and dietary intake to determine deficiency status.

        Args:
            features_df: Feature DataFrame

        Returns:
            DataFrame with binary deficiency labels for each nutrient
        """
        n_samples = len(features_df)
        labels = {}

        # Vitamin B12 deficiency (serum < 200 pg/mL)
        labels['Vitamin_B12_deficiency'] = (features_df['serum_b12_pg_ml'] < 200).astype(int)

        # Vitamin D deficiency (serum < 20 ng/mL)
        labels['Vitamin_D_deficiency'] = (features_df['serum_vitamin_d_ng_ml'] < 20).astype(int)

        # Iron deficiency (serum ferritin < 15 ng/mL for women, < 30 for men)
        iron_threshold = features_df['gender'].map({0: 15, 1: 30})
        labels['Iron_deficiency'] = (features_df['serum_ferritin_ng_ml'] < iron_threshold).astype(int)

        # Folate deficiency (serum < 3 ng/mL)
        labels['Folate_deficiency'] = (features_df['serum_folate_ng_ml'] < 3).astype(int)

        # Calcium (based on dietary intake < 600 mg/day)
        labels['Calcium_deficiency'] = (features_df['calcium_mg'] < 600).astype(int)

        # Magnesium (dietary < 200 mg/day)
        labels['Magnesium_deficiency'] = (features_df['magnesium_mg'] < 200).astype(int)

        # Zinc (dietary < 6 mg/day)
        labels['Zinc_deficiency'] = (features_df['zinc_mg'] < 6).astype(int)

        # Vitamin C (dietary < 30 mg/day)
        labels['Vitamin_C_deficiency'] = (features_df['vitamin_c_mg'] < 30).astype(int)

        # Vitamin E (dietary < 5 mg/day)
        labels['Vitamin_E_deficiency'] = (features_df['vitamin_e_mg'] < 5).astype(int)

        # Vitamin A (dietary < 400 mcg/day)
        labels['Vitamin_A_deficiency'] = (features_df['vitamin_a_mcg'] < 400).astype(int)

        # Additional nutrients (simplified thresholds)
        labels['Selenium_deficiency'] = (features_df['selenium_mcg'] < 40).astype(int)
        labels['Vitamin_B6_deficiency'] = (features_df['vitamin_b6_mg'] < 0.8).astype(int)
        labels['Thiamin_deficiency'] = (features_df['thiamin_mg'] < 0.6).astype(int)
        labels['Riboflavin_deficiency'] = (features_df['riboflavin_mg'] < 0.6).astype(int)
        labels['Niacin_deficiency'] = (features_df['niacin_mg'] < 8).astype(int)
        labels['Phosphorus_deficiency'] = (features_df['phosphorus_mg'] < 580).astype(int)
        labels['Potassium_deficiency'] = (features_df['potassium_mg'] < 2000).astype(int)
        labels['Copper_deficiency'] = (features_df['copper_mg'] < 0.6).astype(int)

        # Vitamin K (simulated, no direct measurement)
        labels['Vitamin_K_deficiency'] = np.random.binomial(1, 0.08, n_samples)

        # Sodium (rarely deficient, but for completeness)
        labels['Sodium_deficiency'] = (features_df['sodium_mg'] < 1000).astype(int)

        labels_df = pd.DataFrame(labels, index=features_df.index)

        # Print deficiency rates
        print("\nDeficiency rates:")
        for nutrient in labels_df.columns:
            rate = labels_df[nutrient].mean()
            print(f"  {nutrient}: {rate*100:.1f}%")

        return labels_df

    def preprocess_features(self, features_df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Preprocess features for model input.

        Args:
            features_df: Feature DataFrame
            fit: Whether to fit the scaler (True for training data)

        Returns:
            Preprocessed feature array
        """
        # Select features (exclude population_category as it's used separately)
        feature_cols = [col for col in features_df.columns if col != 'population_category']

        X = features_df[feature_cols].copy()

        # Handle missing values
        if self.handle_missing == 'mean':
            X = X.fillna(X.mean())
        elif self.handle_missing == 'median':
            X = X.fillna(X.median())
        elif self.handle_missing == 'drop':
            X = X.dropna()

        # Convert to numpy array
        X_array = X.values.astype(np.float32)

        # Scale features
        if self.scale_features:
            if fit:
                X_array = self.scaler.fit_transform(X_array)
                self.feature_statistics = {
                    'mean': self.scaler.mean_,
                    'std': self.scaler.scale_
                }
            else:
                X_array = self.scaler.transform(X_array)

        # Store feature names
        if self.feature_names is None:
            self.feature_names = feature_cols

        return X_array


class NHANESDataset(Dataset):
    """
    PyTorch Dataset for NHANES data.
    """

    def __init__(self,
                 features: np.ndarray,
                 labels: np.ndarray,
                 population_indices: np.ndarray):
        """
        Initialize NHANES dataset.

        Args:
            features: Feature array (n_samples, n_features)
            labels: Label array (n_samples, n_nutrients)
            population_indices: Population category array (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.population_indices = torch.LongTensor(population_indices)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'features': self.features[idx],
            'labels': self.labels[idx],
            'population': self.population_indices[idx]
        }


def create_nhanes_dataloaders(data_splits: Dict,
                              preprocessor: NHANESPreprocessor,
                              batch_size: int = 32,
                              num_workers: int = 4,
                              shuffle_train: bool = True) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for NHANES dataset.

    Args:
        data_splits: Dictionary with 'train', 'val', 'test' data
        preprocessor: NHANES preprocessor instance
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle_train: Whether to shuffle training data

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    dataloaders = {}

    for split_name, split_data in data_splits.items():
        features_df = split_data['features']
        labels_df = split_data['labels']

        # Preprocess features
        fit_scaler = (split_name == 'train')
        X = preprocessor.preprocess_features(features_df, fit=fit_scaler)

        # Extract labels
        y = labels_df.values.astype(np.float32)

        # Extract population indices
        population_indices = features_df['population_category'].values.astype(np.int64)

        # Create dataset
        dataset = NHANESDataset(X, y, population_indices)

        # Create dataloader
        shuffle = shuffle_train if split_name == 'train' else False
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

        dataloaders[split_name] = dataloader

        print(f"{split_name.capitalize()} DataLoader: {len(dataset)} samples, {len(dataloader)} batches")

    return dataloaders


if __name__ == "__main__":
    print("Testing NHANES Dataset Module...")
    print("=" * 70)

    # Initialize preprocessor
    preprocessor = NHANESPreprocessor(random_state=42)

    # Generate synthetic data
    data_splits = preprocessor.generate_synthetic_nhanes_data(
        n_samples=1000,  # Smaller for testing
        train_ratio=0.7,
        val_ratio=0.15
    )

    # Create dataloaders
    dataloaders = create_nhanes_dataloaders(
        data_splits,
        preprocessor,
        batch_size=32,
        num_workers=0  # 0 for testing
    )

    # Test dataloader
    print("\nTesting DataLoader:")
    batch = next(iter(dataloaders['train']))
    print(f"Batch features shape: {batch['features'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
    print(f"Batch population shape: {batch['population'].shape}")
    print(f"Population distribution in batch: {torch.bincount(batch['population'])}")

    print("\nNHANES Dataset module test completed!")
