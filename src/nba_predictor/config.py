"""Configuration constants for NBA Predictor."""
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Directories
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Data files
RAW_DATA_FILE = RAW_DATA_DIR / "games.csv"
TRAIN_FILE = PROCESSED_DATA_DIR / "train.parquet"
TEST_FILE = PROCESSED_DATA_DIR / "test.parquet"

# Model files
BASELINE_MODEL_PATH = MODELS_DIR / "baseline.joblib"

# Column names (match Kaggle dataset schema)
GAME_DATE_COL = "GAME_DATE_EST"
HOME_TEAM_COL = "HOME_TEAM_ID"
VISITOR_TEAM_COL = "VISITOR_TEAM_ID"
HOME_PTS_COL = "PTS_home"
AWAY_PTS_COL = "PTS_away"
LABEL_COL = "home_win"

# Feature columns for modeling
FEATURE_COLS = [HOME_TEAM_COL, VISITOR_TEAM_COL]

# Train/test split
TRAIN_SPLIT_RATIO = 0.8

# Reproducibility
RANDOM_SEED = 42

# Model hyperparameters (PRD v1 baseline)
LOGISTIC_REGRESSION_PARAMS = {
    "random_state": RANDOM_SEED,
    "max_iter": 1000,
    "solver": "lbfgs",
    "class_weight": None  # No class balancing for baseline
}
