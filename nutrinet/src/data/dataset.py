"""
Dataset classes for NUTRINET.

Provides dataset loaders for the three public datasets used in the paper:
1. USDA FoodData Central: Nutrient composition prediction
2. NHANES: Biomarker-nutrient correlation and deficiency risk assessment
3. Framingham Heart Study: Long-term health outcome prediction
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class NutrientDataset(Dataset):
    """
    Base dataset class for nutrient analysis.

    Provides common functionality for all nutrient datasets.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[callable] = None,
    ):
        """
        Args:
            data_dir: Directory containing dataset files
            split: Dataset split ('train', 'val', or 'test')
            transform: Optional transform to apply to samples
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        # Load data
        self.samples = []
        self.labels = []

    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Returns:
            Dictionary containing:
                - 'features': Input features
                - 'label': Target label
                - 'context': Context information
                - 'metadata': Additional metadata
        """
        sample = self.samples[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return {
            'features': torch.FloatTensor(sample),
            'label': torch.FloatTensor([label]),
            'metadata': {'index': idx, 'split': self.split},
        }


class USDAFoodDataset(NutrientDataset):
    """
    USDA FoodData Central dataset.

    Used for:
    - Nutrient composition prediction
    - Food category analysis

    Dataset size: 50,000 food items with complete nutrient profiles
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[callable] = None,
        num_samples: int = 50000,
    ):
        super().__init__(data_dir, split, transform)

        self.num_samples = num_samples
        self._load_usda_data()

    def _load_usda_data(self):
        """Load USDA FoodData Central dataset"""
        # Check if processed data exists
        processed_file = self.data_dir / f"usda_{self.split}.pt"

        if processed_file.exists():
            # Load processed data
            data = torch.load(processed_file)
            self.samples = data['features']
            self.labels = data['labels']
            self.nutrient_names = data['nutrient_names']
            self.food_descriptions = data['food_descriptions']
        else:
            # Generate synthetic data (placeholder)
            print(f"Generating synthetic USDA data ({self.num_samples} samples)...")
            self._generate_synthetic_usda_data()

    def _generate_synthetic_usda_data(self):
        """Generate synthetic USDA-like data"""
        np.random.seed(hash(self.split) % 2**32)

        # Generate nutrient profiles (13 vitamins + 12 minerals + macronutrients)
        num_nutrients = 30
        self.samples = np.random.lognormal(
            mean=2.0, sigma=1.5, size=(self.num_samples, num_nutrients)
        ).astype(np.float32)

        # Add some correlations (e.g., protein and B vitamins)
        protein_idx = 0
        b_vitamin_indices = range(1, 10)
        for i in b_vitamin_indices:
            self.samples[:, i] += 0.3 * self.samples[:, protein_idx]

        # Generate labels (deficiency risk scores)
        self.labels = np.random.beta(2, 5, size=self.num_samples).astype(np.float32)

        # Metadata
        self.nutrient_names = [f"Nutrient_{i}" for i in range(num_nutrients)]
        self.food_descriptions = [f"Food_{i}" for i in range(self.num_samples)]


class NHANESDataset(NutrientDataset):
    """
    NHANES (National Health and Nutrition Examination Survey) dataset.

    Used for:
    - Biomarker-nutrient correlation analysis
    - Deficiency risk assessment

    Dataset size: 8,500 participants with complete dietary and biomarker data
    Links dietary intake with 200+ biomarkers across diverse U.S. populations
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[callable] = None,
        num_samples: int = 8500,
        include_demographics: bool = True,
    ):
        super().__init__(data_dir, split, transform)

        self.num_samples = num_samples
        self.include_demographics = include_demographics
        self._load_nhanes_data()

    def _load_nhanes_data(self):
        """Load NHANES dataset"""
        processed_file = self.data_dir / f"nhanes_{self.split}.pt"

        if processed_file.exists():
            data = torch.load(processed_file)
            self.samples = data['features']
            self.labels = data['labels']
            self.demographics = data.get('demographics', None)
            self.biomarkers = data.get('biomarkers', None)
        else:
            print(f"Generating synthetic NHANES data ({self.num_samples} samples)...")
            self._generate_synthetic_nhanes_data()

    def _generate_synthetic_nhanes_data(self):
        """Generate synthetic NHANES-like data"""
        np.random.seed(hash(self.split) % 2**32 + 1)

        # Dietary intake features (30 nutrients)
        num_nutrients = 30
        dietary_intake = np.random.lognormal(
            mean=1.5, sigma=1.2, size=(self.num_samples, num_nutrients)
        ).astype(np.float32)

        # Demographics (age, gender, ethnicity, BMI, etc.)
        if self.include_demographics:
            age = np.random.normal(65, 15, size=(self.num_samples, 1)).clip(18, 100)
            gender = np.random.binomial(1, 0.5, size=(self.num_samples, 1))
            bmi = np.random.normal(27, 5, size=(self.num_samples, 1)).clip(15, 50)

            self.demographics = np.concatenate([age, gender, bmi], axis=1).astype(np.float32)

            # Combine dietary intake and demographics
            self.samples = np.concatenate([dietary_intake, self.demographics], axis=1)
        else:
            self.samples = dietary_intake
            self.demographics = None

        # Biomarkers (simulated)
        num_biomarkers = 20
        self.biomarkers = np.random.normal(
            loc=0, scale=1, size=(self.num_samples, num_biomarkers)
        ).astype(np.float32)

        # Generate deficiency labels (binary or continuous)
        # Based on nutrient intake and biomarkers
        deficiency_score = (
            -0.5 * dietary_intake.mean(axis=1) +
            0.3 * (age.squeeze() / 100) +
            np.random.normal(0, 0.1, self.num_samples)
        )
        self.labels = (1 / (1 + np.exp(-deficiency_score))).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample with demographics and biomarkers"""
        base_sample = super().__getitem__(idx)

        if self.biomarkers is not None:
            base_sample['biomarkers'] = torch.FloatTensor(self.biomarkers[idx])

        if self.demographics is not None:
            base_sample['demographics'] = torch.FloatTensor(self.demographics[idx])

        return base_sample


class FraminghamDataset(NutrientDataset):
    """
    Framingham Heart Study dataset.

    Used for:
    - Long-term health outcome prediction
    - Causal inference

    Dataset size: 12,000 participants with longitudinal data spanning 5+ years
    Tracks nutritional patterns and cardiovascular outcomes across three generations
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[callable] = None,
        num_samples: int = 12000,
        sequence_length: int = 5,  # Years of data
    ):
        super().__init__(data_dir, split, transform)

        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self._load_framingham_data()

    def _load_framingham_data(self):
        """Load Framingham Heart Study dataset"""
        processed_file = self.data_dir / f"framingham_{self.split}.pt"

        if processed_file.exists():
            data = torch.load(processed_file)
            self.samples = data['features']
            self.labels = data['labels']
            self.sequences = data.get('sequences', None)
            self.outcomes = data.get('outcomes', None)
        else:
            print(f"Generating synthetic Framingham data ({self.num_samples} samples)...")
            self._generate_synthetic_framingham_data()

    def _generate_synthetic_framingham_data(self):
        """Generate synthetic Framingham-like data"""
        np.random.seed(hash(self.split) % 2**32 + 2)

        # Baseline features (30 nutrients + demographics)
        num_nutrients = 30
        dietary_baseline = np.random.lognormal(
            mean=1.8, sigma=1.0, size=(self.num_samples, num_nutrients)
        ).astype(np.float32)

        # Demographics
        age_baseline = np.random.normal(55, 12, size=(self.num_samples, 1)).clip(30, 90)
        gender = np.random.binomial(1, 0.52, size=(self.num_samples, 1))  # Slightly more females
        bmi_baseline = np.random.normal(26, 4, size=(self.num_samples, 1)).clip(18, 45)

        demographics = np.concatenate([age_baseline, gender, bmi_baseline], axis=1).astype(np.float32)

        # Combine for baseline features
        self.samples = np.concatenate([dietary_baseline, demographics], axis=1)

        # Generate longitudinal sequences
        self.sequences = []
        for i in range(self.num_samples):
            sequence = []
            for t in range(self.sequence_length):
                # Dietary intake changes over time (with some drift)
                dietary_t = dietary_baseline[i] + np.random.normal(0, 0.1, num_nutrients)
                dietary_t = dietary_t.clip(0, None)  # Non-negative

                # Age increases
                age_t = age_baseline[i, 0] + t

                # BMI may change
                bmi_t = bmi_baseline[i, 0] + np.random.normal(0, 0.5)

                timestep_features = np.concatenate([
                    dietary_t,
                    [age_t, gender[i, 0], bmi_t]
                ])
                sequence.append(timestep_features)

            self.sequences.append(np.array(sequence, dtype=np.float32))

        # Generate health outcomes (cardiovascular events)
        # Risk increases with age, poor nutrition, high BMI
        risk_score = (
            0.02 * age_baseline.squeeze() +
            -0.3 * dietary_baseline.mean(axis=1) +
            0.1 * bmi_baseline.squeeze() +
            np.random.normal(0, 0.5, self.num_samples)
        )
        self.labels = (1 / (1 + np.exp(-risk_score + 2))).astype(np.float32)  # Shift for lower baseline risk

        # Binary outcomes (event occurred or not)
        self.outcomes = (self.labels > 0.5).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample with longitudinal sequence"""
        base_sample = super().__getitem__(idx)

        if self.sequences is not None:
            base_sample['sequence'] = torch.FloatTensor(self.sequences[idx])

        if self.outcomes is not None:
            base_sample['outcome'] = torch.FloatTensor([self.outcomes[idx]])

        return base_sample


def create_dataloaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create train/val/test dataloaders for a dataset.

    Args:
        dataset_name: Name of dataset ('usda', 'nhanes', or 'framingham')
        data_dir: Directory containing dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        **dataset_kwargs: Additional arguments for dataset

    Returns:
        Dictionary of dataloaders for each split
    """
    # Select dataset class
    dataset_classes = {
        'usda': USDAFoodDataset,
        'nhanes': NHANESDataset,
        'framingham': FraminghamDataset,
    }

    if dataset_name not in dataset_classes:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_class = dataset_classes[dataset_name]

    # Create datasets for each split
    datasets = {}
    for split in ['train', 'val', 'test']:
        datasets[split] = dataset_class(
            data_dir=data_dir,
            split=split,
            **dataset_kwargs
        )

    # Create dataloaders
    dataloaders = {}
    for split, dataset in datasets.items():
        shuffle = (split == 'train')
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    return dataloaders
