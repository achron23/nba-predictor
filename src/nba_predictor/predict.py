"""Prediction interface for NBA Predictor.

This module provides functions for loading the trained model and
making predictions on new games.
"""

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from pathlib import Path
from typing import Union
from . import config


def load_model(model_path: Path = config.BASELINE_MODEL_PATH) -> Pipeline:
    """Load trained model from disk.

    Args:
        model_path: Path to saved model file

    Returns:
        Loaded sklearn Pipeline

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Train the model first with:\n"
            f"  uv run python -m nba_predictor.train_baseline"
        )

    pipeline = joblib.load(model_path)

    return pipeline


def predict_proba(
    pipeline: Pipeline,
    home_team_id: int,
    visitor_team_id: int
) -> float:
    """Predict probability of home team winning.

    Args:
        pipeline: Trained model pipeline
        home_team_id: Home team ID
        visitor_team_id: Visitor team ID

    Returns:
        Probability of home win (float in [0, 1])
    """
    # Create input DataFrame
    X = pd.DataFrame({
        config.HOME_TEAM_COL: [home_team_id],
        config.VISITOR_TEAM_COL: [visitor_team_id]
    })

    # Get probability for class 1 (home win)
    proba = pipeline.predict_proba(X)[0, 1]

    return float(proba)


def predict_winner(
    pipeline: Pipeline,
    home_team_id: int,
    visitor_team_id: int,
    threshold: float = 0.5
) -> str:
    """Predict which team will win.

    Args:
        pipeline: Trained model pipeline
        home_team_id: Home team ID
        visitor_team_id: Visitor team ID
        threshold: Probability threshold for predicting home win (default 0.5)

    Returns:
        'home' or 'away' depending on predicted winner
    """
    prob = predict_proba(pipeline, home_team_id, visitor_team_id)

    return 'home' if prob >= threshold else 'away'


def predict_batch(
    pipeline: Pipeline,
    games_df: pd.DataFrame
) -> pd.DataFrame:
    """Predict outcomes for multiple games.

    Args:
        pipeline: Trained model pipeline
        games_df: DataFrame with HOME_TEAM_ID and VISITOR_TEAM_ID columns

    Returns:
        DataFrame with added 'home_win_prob' and 'predicted_winner' columns

    Raises:
        ValueError: If required columns are missing
    """
    # Validate columns
    required_cols = [config.HOME_TEAM_COL, config.VISITOR_TEAM_COL]
    missing_cols = set(required_cols) - set(games_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Create a copy to avoid modifying original
    result_df = games_df.copy()

    # Get predictions
    X = games_df[required_cols]
    probas = pipeline.predict_proba(X)[:, 1]

    # Add predictions to dataframe
    result_df['home_win_prob'] = probas
    result_df['predicted_winner'] = ['home' if p >= 0.5 else 'away' for p in probas]

    return result_df


if __name__ == "__main__":
    # Example usage demonstrating the prediction API
    print("=" * 60)
    print("NBA Predictor - Prediction Examples")
    print("=" * 60)

    try:
        # Load model
        print("\nLoading trained model...")
        model = load_model()
        print("✓ Model loaded successfully")

        # Example 1: Single prediction (Lakers vs Celtics)
        print("\n" + "-" * 60)
        print("Example 1: Single Prediction")
        print("-" * 60)
        home_id = 1610612747  # Lakers
        visitor_id = 1610612738  # Celtics

        prob = predict_proba(model, home_id, visitor_id)
        winner = predict_winner(model, home_id, visitor_id)

        print(f"Matchup: Lakers (home) vs Celtics (away)")
        print(f"Home Team ID: {home_id}")
        print(f"Visitor Team ID: {visitor_id}")
        print(f"Home win probability: {prob:.3f}")
        print(f"Predicted winner: {winner}")

        # Example 2: Multiple predictions
        print("\n" + "-" * 60)
        print("Example 2: Batch Predictions")
        print("-" * 60)

        games = pd.DataFrame({
            config.HOME_TEAM_COL: [1610612747, 1610612751, 1610612744],  # Lakers, Nets, Warriors
            config.VISITOR_TEAM_COL: [1610612738, 1610612752, 1610612761]  # Celtics, Knicks, Raptors
        })

        predictions = predict_batch(model, games)
        print(predictions[[config.HOME_TEAM_COL, config.VISITOR_TEAM_COL,
                          'home_win_prob', 'predicted_winner']].to_string(index=False))

        print("\n" + "=" * 60)
        print("Prediction examples complete!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\n❌ {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
