"""Prediction logic for NBA game outcomes.

This module handles model loading, feature preparation, and inference.
For v2, we use default feature values. In future versions, this can be
enhanced with live team statistics from a database.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, Optional
from pathlib import Path
from .. import config


class NBAPredictor:
    """NBA game outcome predictor.

    Loads the trained model and provides prediction interface with
    feature engineering for inference.
    """

    def __init__(self, model_path: Path = config.MODEL_V1_1_PATH):
        """Initialize predictor and load model.

        Args:
            model_path: Path to trained model joblib file

        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = joblib.load(model_path)
        self.model_version = "v1.1"
        self.feature_cols = config.ENGINEERED_FEATURE_COLS

        # Load historical data for feature computation
        self._load_historical_data()

    def _load_historical_data(self):
        """Load recent historical data for computing rolling features.

        For v2, we'll use test set as a proxy for recent data.
        In production, this would query a database of recent games.
        """
        try:
            # Load test data as proxy for recent games
            test_df = pd.read_parquet(config.TEST_FEATURES_FILE)

            # Keep only most recent games per team (last 20 games)
            test_df = test_df.sort_values(config.GAME_DATE_COL)

            # Create team statistics lookup from recent games
            self.team_stats = self._compute_team_stats_from_data(test_df)

        except Exception as e:
            print(f"Warning: Could not load historical data: {e}")
            self.team_stats = {}

    def _compute_team_stats_from_data(self, df: pd.DataFrame) -> Dict:
        """Extract recent team statistics from historical data.

        Args:
            df: DataFrame with recent games and features

        Returns:
            Dictionary mapping team_id to their recent statistics
        """
        team_stats = {}

        # Get unique teams
        home_teams = df[config.HOME_TEAM_COL].unique()
        away_teams = df[config.VISITOR_TEAM_COL].unique()
        all_teams = set(list(home_teams) + list(away_teams))

        for team_id in all_teams:
            # Get recent games for this team (as home or away)
            home_games = df[df[config.HOME_TEAM_COL] == team_id].tail(10)
            away_games = df[df[config.VISITOR_TEAM_COL] == team_id].tail(10)

            # Extract average stats
            if len(home_games) > 0:
                stats = {
                    'win_pct_L10': home_games['home_win_pct_L10'].mean(),
                    'fg_pct_L10': home_games['home_fg_pct_L10'].mean(),
                    'fg3_pct_L10': home_games['home_fg3_pct_L10'].mean(),
                    'ft_pct_L10': home_games['home_ft_pct_L10'].mean(),
                    'pt_diff_L10': home_games['home_pt_diff_L10'].mean(),
                    'rest_days': home_games['home_rest_days'].mean(),
                    'back_to_back': home_games['home_back_to_back'].mean(),
                    'win_streak': home_games['home_win_streak'].mean(),
                }
            elif len(away_games) > 0:
                stats = {
                    'win_pct_L10': away_games['away_win_pct_L10'].mean(),
                    'fg_pct_L10': away_games['away_fg_pct_L10'].mean(),
                    'fg3_pct_L10': away_games['away_fg3_pct_L10'].mean(),
                    'ft_pct_L10': away_games['away_ft_pct_L10'].mean(),
                    'pt_diff_L10': away_games['away_pt_diff_L10'].mean(),
                    'rest_days': away_games['away_rest_days'].mean(),
                    'back_to_back': away_games['away_back_to_back'].mean(),
                    'win_streak': away_games['away_win_streak'].mean(),
                }
            else:
                # Use league-average defaults if no data
                stats = self._get_default_stats()

            team_stats[team_id] = stats

        return team_stats

    def _get_default_stats(self) -> Dict:
        """Get default/league-average statistics for unknown teams.

        Returns:
            Dictionary with default feature values
        """
        return {
            'win_pct_L10': 0.5,      # 50% win rate
            'fg_pct_L10': 0.46,      # League avg FG%
            'fg3_pct_L10': 0.36,     # League avg 3P%
            'ft_pct_L10': 0.78,      # League avg FT%
            'pt_diff_L10': 0.0,      # Neutral point differential
            'rest_days': 2.0,        # Typical 2 days rest
            'back_to_back': 0.0,     # Not back-to-back
            'win_streak': 0.0,       # No streak
        }

    def prepare_features(
        self,
        home_team_id: int,
        away_team_id: int,
        season: Optional[int] = None,
        game_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Prepare features for prediction.

        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            season: Optional season year (defaults to most recent)
            game_date: Optional game date (not used in v2, reserved for future)

        Returns:
            DataFrame with single row containing all required features
        """
        # Get team stats (or defaults if not found)
        home_stats = self.team_stats.get(home_team_id, self._get_default_stats())
        away_stats = self.team_stats.get(away_team_id, self._get_default_stats())

        # Default season to most recent in data
        if season is None:
            season = 2023  # Default to 2023-24 season

        # Build feature dictionary
        features = {
            config.HOME_TEAM_COL: home_team_id,
            config.VISITOR_TEAM_COL: away_team_id,
            config.SEASON_COL: season,
            'home_win_pct_L10': home_stats['win_pct_L10'],
            'away_win_pct_L10': away_stats['win_pct_L10'],
            'home_fg_pct_L10': home_stats['fg_pct_L10'],
            'away_fg_pct_L10': away_stats['fg_pct_L10'],
            'home_fg3_pct_L10': home_stats['fg3_pct_L10'],
            'away_fg3_pct_L10': away_stats['fg3_pct_L10'],
            'home_ft_pct_L10': home_stats['ft_pct_L10'],
            'away_ft_pct_L10': away_stats['ft_pct_L10'],
            'home_pt_diff_L10': home_stats['pt_diff_L10'],
            'away_pt_diff_L10': away_stats['pt_diff_L10'],
            'home_rest_days': home_stats['rest_days'],
            'away_rest_days': away_stats['rest_days'],
            'home_back_to_back': home_stats['back_to_back'],
            'away_back_to_back': away_stats['back_to_back'],
            'home_win_streak': home_stats['win_streak'],
            'away_win_streak': away_stats['win_streak'],
            'home_h2h_win_pct': 0.5,  # Default neutral H2H
            'home_h2h_pt_diff': 0.0,  # Default neutral H2H
        }

        # Create DataFrame with correct column order
        return pd.DataFrame([features])[self.feature_cols]

    def predict(
        self,
        home_team_id: int,
        away_team_id: int,
        season: Optional[int] = None,
        game_date: Optional[str] = None
    ) -> Dict:
        """Predict home team win probability.

        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            season: Optional season year
            game_date: Optional game date

        Returns:
            Dictionary with prediction results:
                - p_home_win: Probability of home team winning
                - model_version: Model version used
                - features_used: Number of features
                - home_team_id: Input home team
                - away_team_id: Input away team

        Raises:
            ValueError: If team IDs are invalid
        """
        # Validate inputs
        if home_team_id == away_team_id:
            raise ValueError("Home and away teams must be different")

        # Prepare features
        X = self.prepare_features(home_team_id, away_team_id, season, game_date)

        # Make prediction
        try:
            # Get probability of home win (class 1)
            p_home_win = float(self.model.predict_proba(X)[0, 1])
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

        return {
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'p_home_win': p_home_win,
            'model_version': self.model_version,
            'features_used': len(self.feature_cols)
        }
