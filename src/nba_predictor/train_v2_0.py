"""Train v2.0 model using features computed from SQLite database.

This script:
1. Queries all completed games from SQLite database
2. Computes features on-demand using DatabaseFeatureComputer
3. Trains a LogisticRegression model (same as v1.1 but with v2.0 features)
4. Evaluates on time-based test split
5. Saves model to models/model_v2_0.joblib

Key differences from v1.1:
- Features computed from SQLite, not parquet files
- NO shooting stats (FG%, FT%, 3P%) in feature set
- Reduced feature count: 15 features vs 21
- Team IDs are balldontlie IDs (simpler, no mapping)
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.calibration import calibration_curve

from . import config
from .db.connection import get_db_connection
from .db.feature_computer import DatabaseFeatureComputer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_all_completed_games() -> pd.DataFrame:
    """Fetch all completed games from database.

    Returns:
        DataFrame with columns: game_id, game_date, season, home_team_id,
                                visitor_team_id, home_team_score, visitor_team_score, home_win
    """
    logger.info("Fetching completed games from database...")

    with get_db_connection(config.DB_PATH) as conn:
        query = """
        SELECT
            game_id,
            game_date,
            season,
            home_team_id,
            visitor_team_id,
            home_team_score,
            visitor_team_score,
            home_win
        FROM games
        WHERE status = 'Final'
          AND home_team_score IS NOT NULL
          AND visitor_team_score IS NOT NULL
        ORDER BY game_date ASC
        """

        df = pd.read_sql_query(query, conn)

    logger.info(f"✓ Fetched {len(df)} completed games")
    logger.info(f"  Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    logger.info(f"  Seasons: {sorted(df['season'].unique())}")

    return df


def compute_features_for_all_games(
    games_df: pd.DataFrame,
    feature_computer: DatabaseFeatureComputer,
    skip_first_n: int = 20
) -> pd.DataFrame:
    """Compute features for all games using on-demand feature computation.

    Args:
        games_df: DataFrame of games from database
        feature_computer: DatabaseFeatureComputer instance
        skip_first_n: Skip first N games per season (insufficient history for features)

    Returns:
        DataFrame with features and labels
    """
    logger.info(f"Computing features for {len(games_df)} games...")
    logger.info(f"Skipping first {skip_first_n} games per season (insufficient history)")

    features_list: List[Dict[str, Any]] = []
    labels_list: List[int] = []

    # Skip early season games (not enough history for rolling features)
    for season in sorted(games_df['season'].unique()):
        season_games = games_df[games_df['season'] == season]
        games_to_process = season_games.iloc[skip_first_n:]

        logger.info(f"Season {season}: Processing {len(games_to_process)} games (skipped {skip_first_n})")

        for idx, row in games_to_process.iterrows():
            game_id = row['game_id']
            game_date = row['game_date']
            home_team_id = row['home_team_id']
            visitor_team_id = row['visitor_team_id']
            home_win = row['home_win']
            season_year = row['season']

            # Compute features as of this game date
            try:
                features = feature_computer.compute_features_for_game(
                    home_team_id=home_team_id,
                    away_team_id=visitor_team_id,
                    game_date=game_date,
                    season=season_year
                )

                # Check if any critical features are None
                critical_features = [
                    'home_win_pct_L10', 'away_win_pct_L10',
                    'home_pt_diff_L10', 'away_pt_diff_L10'
                ]

                if any(features.get(f) is None for f in critical_features):
                    logger.debug(f"Skipping game {game_id}: Missing critical features")
                    continue

                # Fill remaining None values with defaults
                features = fill_missing_features(features)

                # Add game_date for train/test split
                features['game_date'] = game_date

                features_list.append(features)
                labels_list.append(home_win)

            except Exception as e:
                logger.warning(f"Failed to compute features for game {game_id}: {e}")
                continue

    logger.info(f"✓ Successfully computed features for {len(features_list)} games")

    # Create DataFrame
    features_df = pd.DataFrame(features_list)
    features_df['home_win'] = labels_list

    return features_df


def fill_missing_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """Replace None values in features with defaults.

    Args:
        features: Feature dictionary (may contain None values)

    Returns:
        Feature dictionary with None replaced by defaults
    """
    defaults_map = {
        'home_win_pct_L10': config.V2_0_DEFAULT_FEATURES['win_pct_L10'],
        'away_win_pct_L10': config.V2_0_DEFAULT_FEATURES['win_pct_L10'],
        'home_pt_diff_L10': config.V2_0_DEFAULT_FEATURES['pt_diff_L10'],
        'away_pt_diff_L10': config.V2_0_DEFAULT_FEATURES['pt_diff_L10'],
        'home_rest_days': config.V2_0_DEFAULT_FEATURES['rest_days'],
        'away_rest_days': config.V2_0_DEFAULT_FEATURES['rest_days'],
        'home_back_to_back': config.V2_0_DEFAULT_FEATURES['back_to_back'],
        'away_back_to_back': config.V2_0_DEFAULT_FEATURES['back_to_back'],
        'home_win_streak': config.V2_0_DEFAULT_FEATURES['win_streak'],
        'away_win_streak': config.V2_0_DEFAULT_FEATURES['win_streak'],
        'home_h2h_win_pct': config.V2_0_DEFAULT_FEATURES['h2h_win_pct'],
        'home_h2h_pt_diff': config.V2_0_DEFAULT_FEATURES['h2h_pt_diff'],
    }

    for key, default in defaults_map.items():
        if features.get(key) is None:
            features[key] = default

    return features


def time_based_train_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8
) -> tuple:
    """Split data by time (first 80% train, last 20% test).

    Args:
        df: DataFrame with game_date column
        train_ratio: Fraction of data for training

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Splitting data (train ratio: {train_ratio})...")

    # Sort by date
    df = df.sort_values('game_date')

    # Split by time
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    logger.info(f"  Train: {len(train_df)} games ({train_df['game_date'].min()} to {train_df['game_date'].max()})")
    logger.info(f"  Test:  {len(test_df)} games ({test_df['game_date'].min()} to {test_df['game_date'].max()})")

    # Separate features and labels
    X_train = train_df[config.V2_0_FEATURE_COLS]
    X_test = test_df[config.V2_0_FEATURE_COLS]
    y_train = train_df['home_win']
    y_test = test_df['home_win']

    logger.info(f"  Features used: {len(config.V2_0_FEATURE_COLS)}")
    logger.info(f"  Feature columns: {config.V2_0_FEATURE_COLS}")

    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Train logistic regression model.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        Trained LogisticRegression model
    """
    logger.info("Training LogisticRegression model...")

    model = LogisticRegression(
        random_state=config.RANDOM_SEED,
        max_iter=1000,
        solver='lbfgs'
    )

    model.fit(X_train, y_train)

    logger.info("✓ Model training complete")

    return model


def evaluate_model(
    model: LogisticRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """Evaluate model on test set.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating model on test set...")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)

    logger.info("=" * 60)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Log Loss:  {logloss:.4f}")
    logger.info("")
    logger.info("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Away Win', 'Home Win']))

    # Calibration analysis
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_pred_proba, n_bins=10, strategy='uniform'
    )

    logger.info("Calibration Curve (predicted vs actual):")
    for i, (actual, pred) in enumerate(zip(fraction_of_positives, mean_predicted_value)):
        logger.info(f"  Bin {i+1}: Predicted={pred:.3f}, Actual={actual:.3f}")

    logger.info("=" * 60)

    return {
        'accuracy': accuracy,
        'log_loss': logloss,
        'n_test_samples': len(y_test),
        'calibration_curve': {
            'fraction_of_positives': fraction_of_positives.tolist(),
            'mean_predicted_value': mean_predicted_value.tolist()
        }
    }


def save_model(model: LogisticRegression, model_path: Path):
    """Save trained model to disk.

    Args:
        model: Trained model
        model_path: Path to save model
    """
    logger.info(f"Saving model to {model_path}...")

    # Create models directory if needed
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(model, model_path)

    logger.info(f"✓ Model saved to {model_path}")


def main():
    """Main training pipeline for v2.0 model."""
    logger.info("=" * 60)
    logger.info("TRAINING v2.0 MODEL FROM SQLITE DATABASE")
    logger.info("=" * 60)

    # Check database exists
    if not config.DB_PATH.exists():
        logger.error(f"Database not found: {config.DB_PATH}")
        logger.error("Please run: uv run python scripts/init_db.py")
        logger.error("Then fetch data: uv run python scripts/fetch_games.py --season 2023")
        return

    # Initialize feature computer
    feature_computer = DatabaseFeatureComputer(config.DB_PATH)

    # Step 1: Fetch all completed games
    games_df = fetch_all_completed_games()

    if len(games_df) == 0:
        logger.error("No games found in database. Please fetch data first:")
        logger.error("  uv run python scripts/fetch_games.py --season 2023")
        return

    # Step 2: Compute features for all games
    features_df = compute_features_for_all_games(games_df, feature_computer)

    if len(features_df) < 100:
        logger.error(f"Insufficient data for training ({len(features_df)} games).")
        logger.error("Please fetch more historical data.")
        return

    # Step 3: Train/test split
    X_train, X_test, y_train, y_test = time_based_train_test_split(features_df)

    # Step 4: Train model
    model = train_model(X_train, y_train)

    # Step 5: Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

    # Step 6: Save model
    save_model(model, config.MODEL_V2_0_PATH)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Model version: v2.0")
    logger.info(f"Features used: {len(config.V2_0_FEATURE_COLS)}")
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test log loss: {metrics['log_loss']:.4f}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Test prediction: uv run python test_api.py")
    logger.info("  2. Start API: uv run python run_api.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
