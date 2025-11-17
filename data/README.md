# Data Directory

This directory is used for storing nutritional data for KGNN training and evaluation.

## Data Format

### Input Features (X)
The model expects nutritional profiles with the following feature categories:

1. **Dietary Intake** (~50 features)
   - Daily intake values for proteins, carbohydrates, fats, vitamins, minerals
   - Example: `Vitamin_B12_intake`, `Iron_intake`, `Calcium_intake`

2. **Anthropometric Measurements** (~8 features)
   - Body measurements: BMI, Weight, Height, Waist Circumference, etc.
   - Example: `BMI`, `Weight`, `WaistCircumference`

3. **Biochemical Indicators** (~15 features)
   - Serum/blood levels of nutrients and biomarkers
   - Example: `Serum_B12`, `Serum_D`, `Hemoglobin`, `Albumin`

4. **Medical History** (~20 features)
   - Binary indicators for chronic conditions and medications
   - Example: `diabetes`, `hypertension`, `medication_count`

5. **Functional Assessments** (~12 features)
   - Activities of Daily Living (ADL), mobility, cognitive scores
   - Example: `ADL_Bathing`, `Mobility_Walking`, `Cognitive_Memory`

### Output Labels (y)
Binary labels (0 or 1) indicating micronutrient deficiency for each of 12 nutrients:

- Vitamin_B12
- Vitamin_D
- Vitamin_B6
- Folate
- Iron
- Calcium
- Magnesium
- Zinc
- Vitamin_C
- Vitamin_E
- Vitamin_A
- Selenium

## Synthetic Data Generation

The codebase includes a synthetic data generator that creates NHANES-style elderly nutritional profiles:

```python
from src.data.preprocessing import NutritionalDataPreprocessor

preprocessor = NutritionalDataPreprocessor()
features_df, labels_df = preprocessor.generate_synthetic_data(
    n_samples=5000,
    n_features=105,
    n_micronutrients=12
)
```

## Using Real Data

To use your own nutritional data:

1. Format your data as Pandas DataFrames matching the structure above
2. Ensure feature names follow the naming conventions (e.g., include keywords like 'serum', 'dietary', 'BMI')
3. Use the preprocessor to transform your data:

```python
X, y = preprocessor.fit_transform(features_df, labels_df)
data_splits = preprocessor.split_data(X, y)
```

## NHANES Data

The paper uses data from the National Health and Nutrition Examination Survey (NHANES) 1988-2018.

To obtain real NHANES data:
1. Visit: https://www.cdc.gov/nchs/nhanes/
2. Download dietary, examination, and laboratory files
3. Extract relevant variables for elderly participants (age 65+)
4. Format according to the structure above

## Data Privacy

When using real patient data:
- Ensure compliance with HIPAA and local data protection regulations
- Remove all personally identifiable information (PII)
- Obtain necessary approvals from IRB or ethics committees
- Consider data anonymization and aggregation techniques

## File Organization

```
data/
├── raw/              # Raw data files
├── processed/        # Preprocessed data
├── synthetic/        # Synthetic data for testing
└── README.md         # This file
```
