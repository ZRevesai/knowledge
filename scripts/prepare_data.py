#!/usr/bin/env python3
"""
Data preparation script for Food-101 dataset.

Downloads and prepares the Food-101 dataset with nutritional annotations.

Usage:
    python scripts/prepare_data.py --output ./data/food101
"""

import argparse
import json
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare Food-101 dataset')

    parser.add_argument('--output', type=str, default='./data/food101',
                       help='Output directory for processed data')
    parser.add_argument('--create_nutritional_db', action='store_true',
                       help='Create default nutritional database')

    return parser.parse_args()


def create_nutritional_database(output_path):
    """
    Create a default nutritional database with sample values.

    In practice, this should be populated with real nutritional data
    from databases like USDA FoodData Central.

    Args:
        output_path: Path to save nutritional database JSON
    """
    # Sample food categories (subset of Food-101)
    food_categories = [
        'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio',
        'beef_tartare', 'beet_salad', 'beignets', 'bibimbap',
        'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad',
        'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche',
        'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla'
        # ... (extend to 500 categories)
    ]

    # Food security categorization
    category_mapping = {
        'staple_foods': ['rice', 'bread', 'pasta', 'oatmeal', 'potatoes'],
        'affordable_proteins': ['eggs', 'chicken', 'beans', 'lentils', 'tofu'],
        'accessible_produce': ['carrots', 'cabbage', 'onions', 'apples', 'bananas'],
        'processed_foods': ['pizza', 'hamburger', 'hot_dog', 'french_fries'],
        'specialty_foods': ['sushi', 'lobster', 'caviar', 'foie_gras']
    }

    nutritional_db = {}

    for food in food_categories:
        # Assign food security category
        category = 'processed_foods'  # default
        for cat_name, cat_foods in category_mapping.items():
            if any(item in food.lower() for item in cat_foods):
                category = cat_name
                break

        # Default nutritional values (should be replaced with real data)
        nutritional_db[food] = {
            'calories': 200.0 + hash(food) % 300,  # 200-500 kcal
            'protein': 10.0 + hash(food) % 20,      # 10-30 g
            'carbs': 25.0 + hash(food) % 30,        # 25-55 g
            'fat': 8.0 + hash(food) % 15,           # 8-23 g
            'vitamin_a': 500.0 + hash(food) % 500,  # 500-1000 IU
            'vitamin_c': 10.0 + hash(food) % 40,    # 10-50 mg
            'vitamin_d': 50.0 + hash(food) % 150,   # 50-200 IU
            'iron': 2.0 + hash(food) % 8,           # 2-10 mg
            'calcium': 100.0 + hash(food) % 300,    # 100-400 mg
            'zinc': 1.5 + hash(food) % 5,           # 1.5-6.5 mg
            'portion_size': 150.0,                   # 150 g default
            'food_security_category': category
        }

    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(nutritional_db, f, indent=2)

    print(f"Nutritional database created: {output_path}")
    print(f"Number of foods: {len(nutritional_db)}")

    return nutritional_db


def main():
    """Main data preparation function."""
    args = parse_args()

    print("="*80)
    print("Food-101 Data Preparation")
    print("="*80)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # Create nutritional database
    if args.create_nutritional_db:
        print("\nCreating nutritional database...")
        nutritional_db_path = output_dir / 'nutritional_database.json'
        create_nutritional_database(nutritional_db_path)

    print("\nData preparation instructions:")
    print("1. Download Food-101 dataset from:")
    print("   https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/")
    print("2. Extract to:", output_dir)
    print("3. The directory structure should be:")
    print(f"   {output_dir}/")
    print("   ├── images/")
    print("   │   ├── apple_pie/")
    print("   │   ├── baby_back_ribs/")
    print("   │   └── ...")
    print("   └── nutritional_database.json")

    print("\nData preparation complete!")


if __name__ == '__main__':
    main()
