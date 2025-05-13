#!/usr/bin/env python3
import os
import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# import your data‐loading & feature code
from ML_model_training import load_dataframe, prepare_features
import config  # make sure this points to where your TRAIN_CSV lives

# set up logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    # 1. Load your pre‐split train CSV
    train_df = load_dataframe(config.TRAIN_CSV)

    # 2. Turn SMILES → fingerprint arrays + target
    X_train, y_train = prepare_features(train_df)

    # 3. Define the grid we want to search
    param_grid = {
        "n_estimators": [50, 100, 200, 500],
        "max_depth":    [None, 10, 20, 30],
    }

    rf = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=2
    )

    logger.info("Starting 5-fold grid search over RF hyperparameters")
    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    best_score  = grid.best_score_
    logger.info(f"Best params: {best_params}, Best CV R²: {best_score:.4f}")

    # 4. Save the full CV table for your LaTeX
    os.makedirs("results", exist_ok=True)
    results = pd.DataFrame(grid.cv_results_)[
        ["param_n_estimators", "param_max_depth", "mean_test_score"]
    ]
    results.columns = ["n_estimators", "max_depth", "mean_cv_r2"]
    results.to_csv("results/hyperparam_results.csv", index=False)
    logger.info("Saved CV results → results/hyperparam_results.csv")

if __name__ == "__main__":
    main()