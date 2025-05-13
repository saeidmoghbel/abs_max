#!/usr/bin/env python3
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, learning_curve
from sklearn.metrics import r2_score, mean_squared_error

import config
from ML_model_training import load_dataframe, prepare_features

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate")

# -----------------------------------------------------------------------------
# 1) Load train split
# -----------------------------------------------------------------------------
train_df = load_dataframe(config.TRAIN_CSV)
X_train, y_train = prepare_features(train_df)
logger.info("Loaded train set: X=%s, y=%s", X_train.shape, y_train.shape)

# -----------------------------------------------------------------------------
# 2) 5-Fold Cross-Validation Stability
# -----------------------------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
r2_scores = []
mse_scores = []

for fold, (idx_tr, idx_val) in enumerate(kf.split(X_train), 1):
    X_tr, X_val = X_train[idx_tr], X_train[idx_val]
    y_tr, y_val = y_train[idx_tr], y_train[idx_val]
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=config.RANDOM_SEED
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    r2_scores.append(r2)
    mse_scores.append(mse)
    logger.info(f"Fold {fold}: R²={r2:.4f}, MSE={mse:.2f}")

# Save fold-wise results
cv_df = pd.DataFrame({
    "Fold": range(1, 6),
    "R2": r2_scores,
    "MSE (nm²)": mse_scores
})
os.makedirs("results", exist_ok=True)
cv_df.to_csv("results/cv_fold_results.csv", index=False)
logger.info("Saved CV fold results → results/cv_fold_results.csv")

# -----------------------------------------------------------------------------
# 3) Learning Curve for R²
# -----------------------------------------------------------------------------
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=config.RANDOM_SEED
)
train_sizes, train_scores, valid_scores = learning_curve(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="r2",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_mean, label="Training R²")
plt.plot(train_sizes, valid_mean, label="Validation R²")
plt.xlabel("Training Set Size")
plt.ylabel("R²")
plt.title("Learning Curve for Random Forest")
plt.legend()
plt.tight_layout()
curve_path = "results/learning_curve_r2.png"
plt.savefig(curve_path)
logger.info(f"Saved learning curve → {curve_path}")
