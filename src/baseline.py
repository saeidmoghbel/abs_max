#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import logging

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import config
from ML_model_training import load_dataframe, prepare_features

# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("baseline")

def main():
    # 1) Load train/test splits
    train_df = load_dataframe(config.TRAIN_CSV)
    test_df = load_dataframe(config.TEST_CSV)
    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)
    logger.info("Train: X=%s, y=%s; Test: X=%s, y=%s",
                X_train.shape, y_train.shape,
                X_test.shape,  y_test.shape)

    # 2) Define models
    models = {
        "MeanBaseline": DummyRegressor(strategy="mean"),
        "LinearReg":    LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            random_state=config.RANDOM_SEED
        )
    }

    # 3) Evaluate each
    records = []
    preds = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        records.append((name, r2, mse))
        preds[name] = y_pred
        logger.info("%s -> R²=%.4f, MSE=%.2f", name, r2, mse)

    # 4) Save baseline table
    os.makedirs("results", exist_ok=True)
    df_baseline = pd.DataFrame(records, columns=["Model","R2","MSE"])
    df_baseline.to_csv("results/baseline_results.csv", index=False)
    logger.info("Saved baseline results → results/baseline_results.csv")

    # 5) Outlier analysis on RF
    rf_pred = preds["RandomForest"]
    abs_err = np.abs(rf_pred - y_test)
    top5_idx = np.argsort(-abs_err)[:5]  # five largest errors
    outliers = test_df.iloc[top5_idx].copy()
    outliers["True"] = y_test[top5_idx]
    outliers["Predicted"] = rf_pred[top5_idx]
    outliers["Error"] = abs_err[top5_idx]
    smiles_col = "Canonical_SMILES"
    outliers["Formula"] = outliers[smiles_col].apply(
        lambda s: rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(s))
    )
    outliers = outliers[["Formula","True","Predicted","Error"]]
    outliers.to_csv("results/top5_outliers.csv", index=False)
    logger.info("Saved top‐5 outliers → results/top5_outliers.csv")

if __name__ == "__main__":
    main()
