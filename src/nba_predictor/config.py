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

# ==========================================
# FEATURE ENGINEERING CONFIGURATION (v1.1)
# ==========================================

# Rolling window parameters
ROLLING_WINDOW_SIZE = 10  # Last N games for rolling statistics
H2H_WINDOW_SIZE = 5       # Last N head-to-head matchups

# Season column
SEASON_COL = "SEASON"

# Game statistics columns (from raw dataset)
GAME_STATS_COLS = {
    'home': ['FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home'],
    'away': ['FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away']
}

# Feature groups (for tracking and ablation studies)
FEATURE_GROUPS = {
    'baseline': [HOME_TEAM_COL, VISITOR_TEAM_COL],
    'season': [SEASON_COL],
    'rolling_record': ['home_win_pct_L10', 'away_win_pct_L10'],
    'rolling_performance': [
        'home_fg_pct_L10', 'away_fg_pct_L10',
        'home_fg3_pct_L10', 'away_fg3_pct_L10',
        'home_ft_pct_L10', 'away_ft_pct_L10',
        'home_pt_diff_L10', 'away_pt_diff_L10'
    ],
    'rest': ['home_rest_days', 'away_rest_days',
             'home_back_to_back', 'away_back_to_back'],
    'streak': ['home_win_streak', 'away_win_streak'],
    'h2h': ['home_h2h_win_pct', 'home_h2h_pt_diff']
}

# All engineered features (flattened)
ENGINEERED_FEATURE_COLS = (
    FEATURE_GROUPS['baseline'] +
    FEATURE_GROUPS['season'] +
    FEATURE_GROUPS['rolling_record'] +
    FEATURE_GROUPS['rolling_performance'] +
    FEATURE_GROUPS['rest'] +
    FEATURE_GROUPS['streak'] +
    FEATURE_GROUPS['h2h']
)

# Feature-engineered output files
TRAIN_FEATURES_FILE = PROCESSED_DATA_DIR / "train_features.parquet"
TEST_FEATURES_FILE = PROCESSED_DATA_DIR / "test_features.parquet"

# v1.1 Model path
MODEL_V1_1_PATH = MODELS_DIR / "model_v1_1.joblib"

# ==========================================
# v2.0 CONFIGURATION (Real-time data)
# ==========================================

# Database
DB_DIR = PROJECT_ROOT / "data" / "db"
DB_PATH = DB_DIR / "nba_predictor.db"

# balldontlie.io API
BALLDONTLIE_API_BASE_URL = "https://api.balldontlie.io/v1"
BALLDONTLIE_API_KEY_ENV_VAR = "BALLDONTLIE_API_KEY"
BALLDONTLIE_API_TIMEOUT = 10  # seconds
BALLDONTLIE_API_MAX_RETRIES = 3
BALLDONTLIE_FREE_TIER_RATE_LIMIT = 5  # requests per minute

# v2.0 Feature groups (NO shooting stats)
V2_0_FEATURE_GROUPS = {
    'baseline': ['home_team_id', 'visitor_team_id'],  # balldontlie team IDs
    'season': ['season'],
    'rolling_record': ['home_win_pct_L10', 'away_win_pct_L10'],
    'rolling_performance': ['home_pt_diff_L10', 'away_pt_diff_L10'],  # Only pt diff, no shooting stats
    'rest': ['home_rest_days', 'away_rest_days', 'home_back_to_back', 'away_back_to_back'],
    'streak': ['home_win_streak', 'away_win_streak'],
    'h2h': ['home_h2h_win_pct', 'home_h2h_pt_diff']
}

# v2.0 Feature columns (flattened)
V2_0_FEATURE_COLS = (
    V2_0_FEATURE_GROUPS['baseline'] +
    V2_0_FEATURE_GROUPS['season'] +
    V2_0_FEATURE_GROUPS['rolling_record'] +
    V2_0_FEATURE_GROUPS['rolling_performance'] +
    V2_0_FEATURE_GROUPS['rest'] +
    V2_0_FEATURE_GROUPS['streak'] +
    V2_0_FEATURE_GROUPS['h2h']
)

# v2.0 Feature defaults (league averages)
V2_0_DEFAULT_FEATURES = {
    'win_pct_L10': 0.5,          # 50% win rate
    'pt_diff_L10': 0.0,          # Neutral point differential
    'rest_days': 2.0,            # Typical 2 days rest
    'back_to_back': 0.0,         # Not back-to-back
    'win_streak': 0.0,           # No streak
    'h2h_win_pct': 0.5,          # Neutral H2H
    'h2h_pt_diff': 0.0,          # Neutral H2H
}

# v2.0 Model path
MODEL_V2_0_PATH = MODELS_DIR / "model_v2_0.joblib"
