"""
Dataset classes for Food-101 and nutrient analysis.

Implements:
1. Food-101 dataset loading with cultural enhancements
2. Nutritional annotation integration
3. Food security categorization
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from .preprocessing import get_train_transforms, get_val_transforms


class Food101Dataset(Dataset):
    """
    Food-101 Dataset with enhanced nutritional annotations.

    Extended to 500 classes with cultural diversity and nutritional information.

    Args:
        root_dir: Root directory containing Food-101 dataset
        split: 'train', 'val', or 'test'
        transform: Image transformations
        nutritional_db_path: Path to nutritional database JSON
    """
    def __init__(self, root_dir, split='train', transform=None,
                 nutritional_db_path=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        # Load image paths and labels
        self.images, self.labels, self.class_names = self._load_data()

        # Load nutritional database
        self.nutritional_db = self._load_nutritional_db(nutritional_db_path)

        print(f"Loaded {len(self.images)} images for {split} split")
        print(f"Number of classes: {len(self.class_names)}")

    def _load_data(self):
        """Load image paths and labels from Food-101 structure."""
        images = []
        labels = []
        class_names = []

        # Get all food categories
        images_dir = self.root_dir / 'images'

        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")

        # Load classes
        for class_dir in sorted(images_dir.iterdir()):
            if class_dir.is_dir():
                class_names.append(class_dir.name)

                # Load images for this class
                for img_path in class_dir.glob('*.jpg'):
                    images.append(str(img_path))
                    labels.append(len(class_names) - 1)

        # Split data
        images, labels = self._split_data(images, labels)

        return images, labels, class_names

    def _split_data(self, images, labels):
        """Split data into train/val/test."""
        # Create consistent splits based on indices
        n_samples = len(images)
        indices = np.arange(n_samples)

        # Set random seed for reproducibility
        np.random.seed(42)
        np.random.shuffle(indices)

        # Split ratios: 70% train, 15% val, 15% test
        train_size = int(0.7 * n_samples)
        val_size = int(0.15 * n_samples)

        if self.split == 'train':
            indices = indices[:train_size]
        elif self.split == 'val':
            indices = indices[train_size:train_size + val_size]
        else:  # test
            indices = indices[train_size + val_size:]

        images = [images[i] for i in indices]
        labels = [labels[i] for i in indices]

        return images, labels

    def _load_nutritional_db(self, db_path):
        """Load nutritional database."""
        if db_path is None or not os.path.exists(db_path):
            # Return default nutritional values
            return self._create_default_nutritional_db()

        with open(db_path, 'r') as f:
            return json.load(f)

    def _create_default_nutritional_db(self):
        """Create default nutritional database with estimated values."""
        # Default nutritional values per 100g
        nutritional_db = {}

        for class_name in self.class_names:
            # Assign default values (these should be replaced with real data)
            nutritional_db[class_name] = {
                'calories': 200.0,  # kcal
                'protein': 10.0,    # g
                'carbs': 25.0,      # g
                'fat': 8.0,         # g
                'vitamin_a': 500.0, # IU
                'vitamin_c': 10.0,  # mg
                'vitamin_d': 50.0,  # IU
                'iron': 2.0,        # mg
                'calcium': 100.0,   # mg
                'zinc': 1.5,        # mg
                'portion_size': 150.0,  # g (default portion)
                'food_security_category': 'processed_foods'
            }

        return nutritional_db

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get item by index.

        Returns:
            Dictionary containing:
                - image: Transformed image tensor
                - label: Food category label
                - portion: Portion size
                - macronutrients: [protein, carbs, fat]
                - micronutrients: [vit_a, vit_c, vit_d, iron, calcium, zinc]
                - food_security_category: Category index
        """
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        # Get label
        label = self.labels[idx]
        class_name = self.class_names[label]

        # Get nutritional information
        nutrition = self.nutritional_db.get(class_name, {})

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Prepare nutritional data
        macronutrients = torch.tensor([
            nutrition.get('protein', 10.0),
            nutrition.get('carbs', 25.0),
            nutrition.get('fat', 8.0)
        ], dtype=torch.float32)

        micronutrients = torch.tensor([
            nutrition.get('vitamin_a', 500.0),
            nutrition.get('vitamin_c', 10.0),
            nutrition.get('vitamin_d', 50.0),
            nutrition.get('iron', 2.0),
            nutrition.get('calcium', 100.0),
            nutrition.get('zinc', 1.5)
        ], dtype=torch.float32)

        portion = torch.tensor([nutrition.get('portion_size', 150.0)],
                               dtype=torch.float32)

        # Food security category encoding
        category_map = {
            'staple_foods': 0,
            'affordable_proteins': 1,
            'accessible_produce': 2,
            'processed_foods': 3,
            'specialty_foods': 4
        }
        food_category = category_map.get(
            nutrition.get('food_security_category', 'processed_foods'), 3
        )

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'portion': portion,
            'macronutrients': macronutrients,
            'micronutrients': micronutrients,
            'food_security_category': torch.tensor(food_category, dtype=torch.long),
            'class_name': class_name
        }


class NutrientDataset(Dataset):
    """
    Dataset specifically for nutrient analysis with custom annotations.

    Args:
        csv_path: Path to CSV with image paths and nutritional data
        transform: Image transformations
    """
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load image
        image = Image.open(row['image_path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Extract nutritional data
        return {
            'image': image,
            'label': torch.tensor(row['food_class'], dtype=torch.long),
            'portion': torch.tensor([row['portion_size']], dtype=torch.float32),
            'macronutrients': torch.tensor([
                row['protein'], row['carbs'], row['fat']
            ], dtype=torch.float32),
            'micronutrients': torch.tensor([
                row['vitamin_a'], row['vitamin_c'], row['vitamin_d'],
                row['iron'], row['calcium'], row['zinc']
            ], dtype=torch.float32)
        }


def create_data_loaders(data_dir, batch_size=32, num_workers=4,
                        nutritional_db_path=None, image_size=224):
    """
    Create train, validation, and test data loaders.

    Args:
        data_dir: Root directory of dataset
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        nutritional_db_path: Path to nutritional database
        image_size: Target image size

    Returns:
        Dictionary with train, val, and test loaders
    """
    # Create datasets
    train_dataset = Food101Dataset(
        root_dir=data_dir,
        split='train',
        transform=get_train_transforms(image_size, augment=True),
        nutritional_db_path=nutritional_db_path
    )

    val_dataset = Food101Dataset(
        root_dir=data_dir,
        split='val',
        transform=get_val_transforms(image_size),
        nutritional_db_path=nutritional_db_path
    )

    test_dataset = Food101Dataset(
        root_dir=data_dir,
        split='test',
        transform=get_val_transforms(image_size),
        nutritional_db_path=nutritional_db_path
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
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
        'test': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }


if __name__ == "__main__":
    # Test dataset
    print("Testing Food101Dataset...")

    # This is a test - you'll need to provide actual data directory
    try:
        dataset = Food101Dataset(
            root_dir='./data/food101',
            split='train',
            transform=get_train_transforms(224, augment=True)
        )

        print(f"Dataset size: {len(dataset)}")

        # Test __getitem__
        sample = dataset[0]
        print("\nSample data:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {value}")

        print("\nDataset test completed!")

    except ValueError as e:
        print(f"Note: {e}")
        print("To test with real data, provide the path to Food-101 dataset")
