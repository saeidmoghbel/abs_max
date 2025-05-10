import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR       = os.path.join(PROJECT_ROOT, 'dataset', 'raw_dataset')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'processed_dataset')

RAW_CSV = os.path.join(RAW_DATA_DIR, "DB_for_chromophore.csv")
PROCESSED_CSV = os.path.join(PROCESSED_DATA_DIR, "processed_chromophore.csv")
TRAIN_CSV = os.path.join(PROCESSED_DATA_DIR, "train_chromophore.csv")
TEST_CSV = os.path.join(PROCESSED_DATA_DIR, "test_chromophore.csv")

TEST_SIZE = 0.2
RANDOM_SEED = 42

FP_RADIUS = 2
FP_SIZE = 1024

OUTLIER_MULTIPLIER = 1.5     # for IQR outlier detection
# TEST_SIZE and RANDOM_SEED already exist

# For ML training:
DEFAULT_RF_PARAMS = {
    'n_estimators': 100,
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}
RF_PARAM_DIST = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 20, 50, 78, 100],
    'max_features': [0.2, 0.3, 0.5, 'sqrt'],
    'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False],
}
RF_RANDOM_ITER = 20
OPTUNA_TRIALS = 50

# Which RF optimization strategy to use in run_pipeline.py:
OPTIMIZATION_METHOD = 'final'  # Options: 'default', 'broad_search', 'optuna', or 'final'

MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
FIGURE_DIR = os.path.join(PROJECT_ROOT, 'figures')

DEFAULT_MODEL_FILENAME = "chromophore_model.pkl"
MODEL_PATH = os.path.join(MODEL_DIR, DEFAULT_MODEL_FILENAME)

for folder in (PROCESSED_DATA_DIR, MODEL_DIR, FIGURE_DIR):
    if not os.path.exists(folder):
        os.makedirs(folder)
