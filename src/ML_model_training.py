#!/usr/bin/env python3
"""
Model Training Script for predicting Chromophore Absorption Maxima

Pipeline:
1. Load processed train/test CSVs from config
2. Canonicalize SMILES and generate fingerprints (if needed)
3. Extract features (fingerprint arrays) and target (Abs_nm)
4. Ensure no data leakage by using pre-split train/test
5. Train a baseline RandomForestRegressor
6. Evaluate on test set (MSE, R2) for Absorption maxima (nm)
7. Save model and metrics
"""
import os
import logging
import pickle
import json

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from src.data_preprocessing import ChromophoreProcessor
from scipy.stats import randint, uniform
from sklearn.metrics import mean_squared_error, r2_score
import optuna
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def canonicalize(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None


def fingerprint(smiles: str, radius: int = 2, n_bits: int = 1024) -> str:
    mol = Chem.MolFromSmiles(smiles)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return gen.GetFingerprint(mol).ToBitString()


def load_dataframe(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


def prepare_features(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
    """
    Ensure SMILES are canonical and generate fingerprint arrays.
    """
    # If 'Fingerprint' column exists, skip generation
    if 'Fingerprint' in df.columns:
        logger.info("Using existing fingerprint column")
        bitstrings = df['Fingerprint']
    else:
        logger.info("Generating canonical SMILES & fingerprints...")
        bitstrings = []
        for smiles in df['Chromophore']:
            can = canonicalize(smiles)
            bitstrings.append(fingerprint(can))
    # Convert bitstrings to numeric arrays
    X = np.stack([np.frombuffer(s.encode(), dtype=np.uint8) - ord('0') for s in bitstrings])
    # Target: absorption max in nm
    y = df['Absorption max (nm)'].values
    return X, y

def train_and_tune_rf(X_train, y_train,
                     optimization_method: str = 'final'):
    """
    Train a RandomForestRegressor under different optimization regimes.

    Args:
        X_train, y_train: training data
        optimization_method: one of
           - 'broad_search'  : RandomizedSearchCV over a wide grid
           - 'optuna'        : Optuna-based bayesian search
           - 'default'       : sklearn’s default RF hyperparams
           - 'final'         : your final, pre-tuned hyperparameters
    Returns:
        A fitted RandomForestRegressor.
    """
    if optimization_method == 'broad_search':
        logging.info("Running broad RandomizedSearchCV...")
        base_rf = RandomForestRegressor(random_state=42)
        param_dist = {
            "n_estimators": randint(100, 300),
            "max_depth": randint(50, 100),
            "max_features": uniform(0.1, 0.4),
            "min_samples_split": randint(2, 10),
        }
        search = RandomizedSearchCV(
            estimator=base_rf,
            param_distributions=param_dist,
            n_iter=100,
            scoring="neg_mean_squared_error",
            cv=5,
            n_jobs=-1,
            verbose=2,
            random_state=42
        )
        search.fit(X_train, y_train)
        logging.info("Best params (broad_search): %s", search.best_params_)
        return search.best_estimator_

    elif optimization_method == 'optuna':
        logging.info("Running Optuna hyperparameter optimization...")
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 20, 100),
                "max_features": trial.suggest_float("max_features", 0.1, 0.5),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            }
            rf = RandomForestRegressor(**params, random_state=42)
            # we do a simple 3-fold CV here; you can swap in your own
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(
                rf, X_train, y_train,
                cv=3,
                scoring="neg_mean_squared_error",
                n_jobs=-1
            )
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)
        best = study.best_params
        logging.info("Best params (optuna): %s", best)
        rf = RandomForestRegressor(**best, random_state=42)
        rf.fit(X_train, y_train)
        return rf

    elif optimization_method == 'default':
        logging.info("Using sklearn default RandomForestRegressor...")
        rf = RandomForestRegressor(random_state=42)
        rf.fit(X_train, y_train)
        return rf

    else:  # 'final'
        logging.info("Using pre-optimized RF parameters for final model")
        rf = RandomForestRegressor(
            n_estimators=134,
            max_depth=78,
            max_features=0.3007,
            min_samples_split=2,
            bootstrap=True,
            random_state=42
        )
        rf.fit(X_train, y_train)
        return rf

def train_model(X_train: np.ndarray, y_train: np.ndarray,
                optimization_method: str = config.OPTIMIZATION_METHOD) -> RandomForestRegressor:
    """
     Wrapper around train_and_tune_rf so you can pass optimization_method
     (e.g. 'default', 'broad_search', 'optuna', 'final') from run_pipeline.
    """
    logger.info("Training RandomForestRegressor for Absorption Maxima... (mode={optimization_method})")
    model = train_and_tune_rf(X_train, y_train,
                              optimization_method=optimization_method)
#        n_estimators=getattr(config, 'RF_N_ESTIMATORS', 100),
#        random_state=config.RANDOM_SEED,
#        n_jobs=-1,
#    )
#    model.fit(X_train, y_train)
    return model


def evaluate(processor: ChromophoreProcessor,
             model: RandomForestRegressor,
             X: np.ndarray, y: np.ndarray,
             test_df: pd.DataFrame = None):
    """
    +    1) Predict
    +    2) Compute MSE & R²
    +    3) Flag IQR-based residual outliers (processor.flag_outliers)
    +    4) Pull preprocessing 'Outlier_nm' if present
    +    Returns: (preds, pred_mask, stat_mask)
    +    """
    logger.info("Evaluating model performance...")
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)
    logger.info(f"MSE (nm^2): {mse:.4f}, R2: {r2:.4f}")

    # Flag prediction-based outliers via IQR on residuals
    residuals = pd.Series(preds - y)
    pred_mask = processor.flag_outliers(residuals).values

    # get preprocessing-outliers if available
    stat_mask = None
    if test_df is not None and 'Outlier_nm' in test_df.columns:
        stat_mask = test_df['Outlier_nm'].values.astype(bool)

    # log outlier count
    logger.info("Prediction outliers flagged: %d", int(pred_mask.sum()))
    if stat_mask is not None:
        logger.info("Statistical outliers flagged: %d", int(stat_mask.sum()))
    return preds, pred_mask, stat_mask


def main():
    # 1. Load pre-split data
    train_df = load_dataframe(config.TRAIN_CSV)
    test_df  = load_dataframe(config.TEST_CSV)

    # 2. Feature preparation (SMILES -> fingerprint arrays)
    X_train, y_train = prepare_features(train_df)
    X_test,  y_test  = prepare_features(test_df)

    # 3. Train model
    model = train_model(X_train, y_train)

    # 4. Evaluate
    train_metrics = evaluate(model, X_train, y_train)
    test_metrics  = evaluate(model, X_test,  y_test)

    # 5. Save model and metrics
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model_path = os.path.join(config.MODEL_DIR, config.DEFAULT_MODEL_FILENAME)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Saved trained model to {model_path}")

    metrics_path = os.path.join(config.MODEL_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({'train': train_metrics, 'test': test_metrics}, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

if __name__ == '__main__':
    main()
