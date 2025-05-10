#!/usr/bin/env python3
"""
Orchestrates the full chromophore workflow:
  1. Preprocess raw data into stratified train/test CSVs
  2. Train the Random Forest model on absorption maxima
  3. Evaluate and produce diagnostic plots
  4. Save model, metrics, and figures
"""

import os
import logging

import matplotlib.pyplot as plt

import config
from src.data_preprocessing import ChromophoreProcessor
from src.ML_model_training import load_dataframe, prepare_features, train_and_tune_rf, train_model, evaluate
from src.data_preprocessing import ChromophoreProcessor
import numpy as np
# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("pipeline")

# -----------------------------------------------------------------------------
# Plotting utilities
# -----------------------------------------------------------------------------
def plot_pred_vs_actual(y_true, y_pred, out_path,
                        pred_mask=None, stat_mask=None):
    """Predicted vs Actual, highlighting prediction‐ and stat‐outliers."""
    plt.figure(figsize=(6,6))

    if pred_mask is not None:
        normal = ~pred_mask
        plt.scatter(y_true[normal], y_pred[normal],
                    c='blue', alpha=0.6, s=20, label='Normal')
        plt.scatter(y_true[pred_mask], y_pred[pred_mask],
                    c='red',  alpha=0.8, s=20, label='Pred outlier')
    else:
        plt.scatter(y_true, y_pred, alpha=0.5, edgecolor='k')

    if stat_mask is not None:
        plt.scatter(y_true[stat_mask], y_pred[stat_mask],
                    facecolors='none', edgecolors='orange',
                    s=60, linewidths=1.5, label='Stat outlier')

    mn, mx = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1, label='Perfect')

    plt.xlabel("Actual Abs max (nm)")
    plt.ylabel("Predicted Abs max (nm)")
    plt.title("Predicted vs. Actual Absorption Maxima")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved parity plot to %s", out_path)


def plot_error_hist(y_true, y_pred, out_path, pred_mask=None):
    """Error histogram, stacked Normal vs Pred‐outlier."""
    errors = y_pred - y_true
    plt.figure(figsize=(6,4))

    if pred_mask is not None:
        normal = errors[~pred_mask]
        bad    = errors[ pred_mask]
        plt.hist(normal, bins=30, alpha=0.7,
                 edgecolor='k', label='Normal')
        plt.hist(bad,    bins=30, alpha=0.7,
                 edgecolor='k', label='Pred outlier')
        plt.legend()
    else:
        plt.hist(errors, bins=30, alpha=0.7, edgecolor='k')

    plt.axvline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel("Prediction Error (nm)")
    plt.ylabel("Count")
    plt.title("Distribution of Prediction Errors")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved error histogram to %s", out_path)
# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------
def main():
    # 1) Preprocessing
    logger.info("=== Preprocessing raw data ===")
    processor = ChromophoreProcessor(
        csv_path=config.RAW_CSV,
        output_dir=config.PROCESSED_DATA_DIR
    )
    df_raw  = processor.load_data()
    logger.info("Raw entries: %d", len(df_raw))
    df_proc = processor.preprocess(df_raw)
    logger.info("Processed entries: %d", len(df_proc))
    processor.split_and_export(df_proc)

    # 2) Load train/test splits
    logger.info("=== Loading train/test data ===")
    train_df = load_dataframe(config.TRAIN_CSV)
    test_df  = load_dataframe(config.TEST_CSV)

    # 3) Prepare feature matrices and targets
    X_train, y_train = prepare_features(train_df)
    X_test,  y_test  = prepare_features(test_df)

    # 4) Train the model
    logger.info("=== Training RandomForestRegressor ===")
    model = train_model(
        X_train,
        y_train,
        optimization_method=config.OPTIMIZATION_METHOD
    )

    # 5) Evaluate performance (training & test)
    logger.info("=== Evaluating on training set ===")
    y_train_pred, train_pred_mask, train_stat_mask = evaluate(
        processor,
        model,
        X_train,
        y_train,
        test_df=train_df
    )

    logger.info("=== Evaluating on test set ===")
    y_pred, pred_mask, stat_mask = evaluate(
        processor,
        model,
        X_test,
        y_test,
        test_df=test_df
    )

    # 6) Plot full 2×2 diagnostics
    logger.info("=== Plotting diagnostics ===")
    os.makedirs(config.FIGURE_DIR, exist_ok=True)
    # Parity (nm) with both outlier masks
    plot_pred_vs_actual(
        y_test, y_pred,
        os.path.join(config.FIGURE_DIR, "pred_vs_actual.png"),
        pred_mask=pred_mask,
        stat_mask=stat_mask
    )

    # Error histogram (nm) showing prediction outliers
    plot_error_hist(
        y_test, y_pred,
        os.path.join(config.FIGURE_DIR, "error_hist.png"),
        pred_mask=pred_mask
    )

    # 7) Finish
    logger.info("Pipeline complete!")

if __name__ == "__main__":
    main()
