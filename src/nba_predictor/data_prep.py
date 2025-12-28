"""Data preprocessing module for NBA Predictor.

This module handles loading, cleaning, labeling, and time-based splitting
of NBA games data to prevent data leakage.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple
from . import config


def load_raw_data(file_path: Path = config.RAW_DATA_FILE) -> pd.DataFrame:
    """Load and validate raw NBA games data.

    Args:
        file_path: Path to raw CSV file

    Returns:
        Cleaned DataFrame with parsed dates

    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If required columns are missing
    """
    # Check file exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            f"Please download the NBA games dataset and place it at {file_path}\n"
            f"Download from: https://www.kaggle.com/datasets/nathanlauga/nba-games"
        )

    print(f"Loading data from {file_path}...")

    # Load CSV
    df = pd.read_csv(file_path)

    # Validate schema
    required_cols = [
        config.GAME_DATE_COL,
        config.HOME_TEAM_COL,
        config.VISITOR_TEAM_COL,
        config.HOME_PTS_COL,
        config.AWAY_PTS_COL
    ]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Parse dates
    df[config.GAME_DATE_COL] = pd.to_datetime(df[config.GAME_DATE_COL])

    # Drop missing values in required columns
    initial_len = len(df)
    df = df.dropna(subset=required_cols)
    dropped = initial_len - len(df)

    if dropped > 0:
        print(f"Dropped {dropped} rows with missing values")

    print(f"Loaded {len(df):,} games from {df[config.GAME_DATE_COL].min().date()} to {df[config.GAME_DATE_COL].max().date()}")

    return df


def create_label(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary home_win label.

    Args:
        df: DataFrame with home and away points

    Returns:
        DataFrame with added 'home_win' column
    """
    df = df.copy()

    # Create label: home_win = 1 if PTS_home > PTS_away else 0
    df[config.LABEL_COL] = (df[config.HOME_PTS_COL] > df[config.AWAY_PTS_COL]).astype(int)

    # Check for ties (should be rare/non-existent in NBA)
    num_ties = (df[config.HOME_PTS_COL] == df[config.AWAY_PTS_COL]).sum()
    if num_ties > 0:
        print(f"Warning: {num_ties} tied games found (assigned to away wins)")

    # Log class balance
    home_wins = df[config.LABEL_COL].sum()
    home_win_rate = df[config.LABEL_COL].mean()
    print(f"Home win rate: {home_win_rate:.3f} ({home_wins:,} / {len(df):,} games)")

    return df


def split_data(
    df: pd.DataFrame,
    train_ratio: float = config.TRAIN_SPLIT_RATIO
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based train/test split.

    CRITICAL: Sorts by date before splitting to prevent data leakage.
    Ensures all games on the same date go to the same split.

    Args:
        df: DataFrame with GAME_DATE_EST column
        train_ratio: Proportion of data for training (default 0.8)

    Returns:
        Tuple of (train_df, test_df)

    Raises:
        AssertionError: If data leakage is detected (train dates overlap test dates)
    """
    print(f"\nSplitting data (train ratio: {train_ratio})...")

    # Sort by date (MUST DO THIS FIRST to prevent leakage)
    df_sorted = df.sort_values(config.GAME_DATE_COL).reset_index(drop=True)

    # Find split index
    split_idx = int(len(df_sorted) * train_ratio)
    
    # Get the date at the split point
    split_date = df_sorted.iloc[split_idx][config.GAME_DATE_COL]
    
    # Simple approach: train gets dates < split_date, test gets dates >= split_date
    # This ensures no overlap since we use strict < for train
    train_df = df_sorted[df_sorted[config.GAME_DATE_COL] < split_date].copy()
    test_df = df_sorted[df_sorted[config.GAME_DATE_COL] >= split_date].copy()

    # Validate no leakage - train dates must be earlier than test dates
    if len(train_df) == 0:
        raise AssertionError("Train set is empty! Adjust train_ratio.")
    
    if len(test_df) == 0:
        raise AssertionError("Test set is empty! Adjust train_ratio.")
    
    actual_max_train_date = train_df[config.GAME_DATE_COL].max()
    min_test_date = test_df[config.GAME_DATE_COL].min()

    # Since train uses < split_date and test uses >= split_date, 
    # actual_max_train_date should always be < min_test_date
    if actual_max_train_date >= min_test_date:
        raise AssertionError(
            f"Data leakage detected! "
            f"Train max date {actual_max_train_date.date()} >= Test min date {min_test_date.date()}"
        )

    print(f"Train: {len(train_df):,} games ({train_df[config.GAME_DATE_COL].min().date()} to {actual_max_train_date.date()})")
    print(f"Test:  {len(test_df):,} games ({min_test_date.date()} to {test_df[config.GAME_DATE_COL].max().date()})")
    print("[OK] No data leakage detected (train dates < test dates)")

    return train_df, test_df


def save_processed_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Save processed train/test sets to parquet format.

    Args:
        train_df: Training data
        test_df: Test data
    """
    # Create output directory if needed
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save to parquet (faster than CSV, preserves dtypes)
    print(f"\nSaving processed data...")
    train_df.to_parquet(config.TRAIN_FILE, index=False)
    test_df.to_parquet(config.TEST_FILE, index=False)

    print(f"[OK] Train data saved to: {config.TRAIN_FILE}")
    print(f"[OK] Test data saved to: {config.TEST_FILE}")


def save_processed_data_with_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Save feature-engineered train/test sets to parquet format.

    Args:
        train_df: Training data with engineered features
        test_df: Test data with engineered features
    """
    # Create output directory if needed
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save to parquet
    print(f"\nSaving feature-engineered data...")
    train_df.to_parquet(config.TRAIN_FEATURES_FILE, index=False)
    test_df.to_parquet(config.TEST_FEATURES_FILE, index=False)

    print(f"[OK] Train features saved to: {config.TRAIN_FEATURES_FILE}")
    print(f"[OK] Test features saved to: {config.TEST_FEATURES_FILE}")


def preprocess_data() -> None:
    """Main preprocessing pipeline with v1.1 feature engineering.

    Loads raw data, creates labels, engineers features, splits with
    time-based validation, and saves processed datasets.

    Can be run as: uv run python -m nba_predictor.data_prep
    """
    print("=" * 60)
    print("NBA Predictor - Data Preprocessing v1.1")
    print("=" * 60)

    try:
        # Load raw data
        df = load_raw_data()

        # Create label
        print("\nCreating labels...")
        df = create_label(df)

        # ===== NEW: FEATURE ENGINEERING (BEFORE SPLIT) =====
        print("\nEngineering features...")
        print("  This may take a few minutes...")

        from .feature_engineering import engineer_all_features, validate_engineered_features

        feature_config = {
            'rolling_window': config.ROLLING_WINDOW_SIZE,
            'h2h_window': config.H2H_WINDOW_SIZE
        }

        df_with_features = engineer_all_features(df, feature_config)
        print(f"[OK] Added {len(config.ENGINEERED_FEATURE_COLS)} features")

        # ===== THEN SPLIT (preserves temporal causality) =====
        train_df, test_df = split_data(df_with_features)

        # Validate features
        validate_engineered_features(train_df, test_df)

        # Save processed data with features
        save_processed_data_with_features(train_df, test_df)

        print("\n" + "=" * 60)
        print("Preprocessing complete!")
        print("=" * 60)
        print("\nNext step: Train v1.1 models with")
        print("  uv run python -m nba_predictor.train_v1_1")

    except Exception as e:
        print(f"\n[ERROR] Error during preprocessing: {e}")
        raise


if __name__ == "__main__":
    preprocess_data()
