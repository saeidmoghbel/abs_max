# Digital Alchemy: Predicting Absorption Maxima from SMILES

## Project Overview

This project focuses on predicting the absorption maxima (λₘₐₓ) of organic chromophores based solely on their molecular structure. Using a supervised machine learning approach, I developed a complete pipeline that takes SMILES representations as input and outputs predicted absorption wavelengths in nanometers. The model is trained on a large, experimentally measured dataset and built around a Random Forest regressor, which has proven effective for structured, high-dimensional input like molecular fingerprints.

The pipeline begins by preprocessing the dataset: cleaning missing values, canonicalizing SMILES, and generating 1024-bit Morgan fingerprints to numerically encode molecular structure. A stratified train/test split ensures balanced distribution of outliers and avoids data leakage. After training, the model’s performance is evaluated using standard regression metrics, including Mean Squared Error (MSE) and R² score. An additional outlier detection step is integrated to identify predictions with unusually high error, helping improve the model’s robustness and interpretation.

Overall, this project demonstrates how machine learning can offer a fast, low-cost alternative to experimental UV–Vis measurements. While inspired by open-source examples, all code and pipeline logic were developed independently and adapted to suit the specific dataset and goals of this work.


## Data Processing Pipeline

### 1. Data Preprocessing (`data_preprocessing.py`)
- Loads and cleans raw dataset
- Canonicalizes SMILES using RDKit
- Converts spectral peaks into a consistent format
- Flags outliers based on IQR
- Outputs cleaned dataset ready for modeling

### 2. Model Training (`ML_model_training.py`)
- Generates 1024-bit Morgan fingerprints
- Splits data into train/test while preserving outlier distribution
- Trains a Random Forest regressor with tuned hyperparameters
- Evaluates performance and saves metrics

---

## Model Details

- **Model**: Random Forest Regressor  
- **Features**: Morgan fingerprints (radius=2, nBits=1024)  
- **Target**: Absorption maximum (λₘₐₓ in nm)

### 🔧 Hyperparameters
- `n_estimators`: 134  
- `max_depth`: 78  
- `max_features`: 0.3007  
- `min_samples_split`: 2  
- `bootstrap`: True

### Evaluation Metrics
- Mean Squared Error (MSE)
- Coefficient of Determination (R²)
- Parity plot and error histogram (saved in `/figures/`)

---

## Results Summary
- Test R²: 0.9339
- Test MSE: 748.52 nm²
- Improved results after integrating outlier flagging logic

---

## How to Use

### 1. Set up Dataset

Download the experimental chromophore dataset from:
[DB for Chromophore – Figshare](https://figshare.com/articles/dataset/DB_for_chromophore/12045567)

Save the `.csv` file inside:
/dataset/raw_dataset/

### 2. Run Pipeline

```bash
python script/run_pipeline.py

This will process the data, train the model, and save outputs to processed_dataset/ and figures/.
