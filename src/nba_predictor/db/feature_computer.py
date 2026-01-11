"""On-demand feature computation from SQLite database.

This module adapts the logic from feature_engineering.py to work with database
queries instead of in-memory DataFrames. Features are computed on-the-fly for
each prediction request.

Key principles:
- Anti-leakage: All queries use WHERE game_date < prediction_date
- Fallback: Returns None for features with insufficient data
- Simplicity: v2.0 focuses on win/loss and point differential (no shooting stats)
"""

import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging

from .connection import get_db_connection

logger = logging.getLogger(__name__)


class DatabaseFeatureComputer:
    """Computes features on-demand from SQLite database."""

    def __init__(self, db_path: Path):
        """Initialize feature computer.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path

    def compute_features_for_game(
        self,
        home_team_id: int,
        away_team_id: int,
        game_date: str,
        season: int
    ) -> Dict[str, Any]:
        """Compute all features for a game.

        Args:
            home_team_id: balldontlie team ID for home team
            away_team_id: balldontlie team ID for away team
            game_date: Game date in YYYY-MM-DD format
            season: Season year

        Returns:
            Dictionary with all features (some may be None if insufficient data)
        """
        features = {
            "home_team_id": home_team_id,
            "visitor_team_id": away_team_id,
            "season": season,
        }

        # Compute rolling win percentages
        home_win_pct = self.compute_rolling_win_pct(home_team_id, game_date)
        away_win_pct = self.compute_rolling_win_pct(away_team_id, game_date)

        features["home_win_pct_L10"] = home_win_pct
        features["away_win_pct_L10"] = away_win_pct

        # Compute rolling point differentials
        home_pt_diff = self.compute_rolling_pt_diff(home_team_id, game_date)
        away_pt_diff = self.compute_rolling_pt_diff(away_team_id, game_date)

        features["home_pt_diff_L10"] = home_pt_diff
        features["away_pt_diff_L10"] = away_pt_diff

        # Compute rest features
        home_rest, home_b2b = self.compute_rest_days(home_team_id, game_date)
        away_rest, away_b2b = self.compute_rest_days(away_team_id, game_date)

        features["home_rest_days"] = home_rest
        features["home_back_to_back"] = home_b2b
        features["away_rest_days"] = away_rest
        features["away_back_to_back"] = away_b2b

        # Compute win streaks
        home_streak = self.compute_win_streak(home_team_id, game_date)
        away_streak = self.compute_win_streak(away_team_id, game_date)

        features["home_win_streak"] = home_streak
        features["away_win_streak"] = away_streak

        # Compute H2H stats
        h2h_win_pct, h2h_pt_diff = self.compute_h2h_stats(
            home_team_id, away_team_id, game_date
        )

        features["home_h2h_win_pct"] = h2h_win_pct
        features["home_h2h_pt_diff"] = h2h_pt_diff

        return features

    def compute_rolling_win_pct(
        self,
        team_id: int,
        as_of_date: str,
        window: int = 10
    ) -> Optional[float]:
        """Compute rolling win percentage for last N games.

        Args:
            team_id: Team ID
            as_of_date: Compute features as of this date (exclusive)
            window: Number of games to include

        Returns:
            Win percentage (0.0-1.0) or None if insufficient data
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            # Query last N games before as_of_date where team played
            query = """
            SELECT home_win
            FROM games
            WHERE game_date < ?
              AND (home_team_id = ? OR visitor_team_id = ?)
              AND status = 'Final'
            ORDER BY game_date DESC
            LIMIT ?
            """

            cursor.execute(query, (as_of_date, team_id, team_id, window))
            games = cursor.fetchall()

            if not games:
                return None

            # Determine if team won each game
            wins = 0
            for row in games:
                # Need to determine if this team was home or away
                # Re-query to get full game details
                pass

            # Alternative: use UNION to handle home/away properly
            query = """
            SELECT
                CASE
                    WHEN home_team_id = ? THEN home_win
                    ELSE (1 - home_win)
                END as team_won
            FROM games
            WHERE game_date < ?
              AND (home_team_id = ? OR visitor_team_id = ?)
              AND status = 'Final'
            ORDER BY game_date DESC
            LIMIT ?
            """

            cursor.execute(query, (team_id, as_of_date, team_id, team_id, window))
            games = cursor.fetchall()

            if not games:
                return None

            wins = sum(row[0] for row in games if row[0] is not None)
            return wins / len(games)

    def compute_rolling_pt_diff(
        self,
        team_id: int,
        as_of_date: str,
        window: int = 10
    ) -> Optional[float]:
        """Compute rolling average point differential.

        Args:
            team_id: Team ID
            as_of_date: Compute features as of this date (exclusive)
            window: Number of games to include

        Returns:
            Average point differential or None if insufficient data
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            query = """
            SELECT
                CASE
                    WHEN home_team_id = ? THEN (home_team_score - visitor_team_score)
                    ELSE (visitor_team_score - home_team_score)
                END as point_diff
            FROM games
            WHERE game_date < ?
              AND (home_team_id = ? OR visitor_team_id = ?)
              AND status = 'Final'
              AND home_team_score IS NOT NULL
              AND visitor_team_score IS NOT NULL
            ORDER BY game_date DESC
            LIMIT ?
            """

            cursor.execute(query, (team_id, as_of_date, team_id, team_id, window))
            games = cursor.fetchall()

            if not games:
                return None

            pt_diffs = [row[0] for row in games]
            return sum(pt_diffs) / len(pt_diffs)

    def compute_rest_days(
        self,
        team_id: int,
        game_date: str
    ) -> Tuple[Optional[float], int]:
        """Compute rest days since last game.

        Args:
            team_id: Team ID
            game_date: Game date in YYYY-MM-DD format

        Returns:
            Tuple of (rest_days, back_to_back_flag)
            rest_days is None if no previous game found
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            # Find most recent game before this date
            query = """
            SELECT game_date
            FROM games
            WHERE game_date < ?
              AND (home_team_id = ? OR visitor_team_id = ?)
              AND status = 'Final'
            ORDER BY game_date DESC
            LIMIT 1
            """

            cursor.execute(query, (game_date, team_id, team_id))
            row = cursor.fetchone()

            if not row:
                return None, 0

            prev_game_date = row[0]

            # Calculate days between
            current = datetime.strptime(game_date, "%Y-%m-%d")
            previous = datetime.strptime(prev_game_date, "%Y-%m-%d")
            rest_days = (current - previous).days

            # Back-to-back indicator
            back_to_back = 1 if rest_days == 1 else 0

            return float(rest_days), back_to_back

    def compute_win_streak(
        self,
        team_id: int,
        as_of_date: str,
        max_lookback: int = 20
    ) -> float:
        """Compute current win/loss streak.

        Args:
            team_id: Team ID
            as_of_date: Compute features as of this date (exclusive)
            max_lookback: Maximum number of recent games to examine

        Returns:
            Streak count (positive = wins, negative = losses, 0 = no games)
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            # Get recent games in reverse chronological order
            query = """
            SELECT
                CASE
                    WHEN home_team_id = ? THEN home_win
                    ELSE (1 - home_win)
                END as team_won
            FROM games
            WHERE game_date < ?
              AND (home_team_id = ? OR visitor_team_id = ?)
              AND status = 'Final'
            ORDER BY game_date DESC
            LIMIT ?
            """

            cursor.execute(query, (team_id, as_of_date, team_id, team_id, max_lookback))
            games = cursor.fetchall()

            if not games:
                return 0.0

            # Calculate streak from most recent games
            current_streak = 0
            last_result = None

            for row in games:
                won = row[0]
                if won is None:
                    continue

                if last_result is None:
                    # First game in sequence
                    last_result = won
                    current_streak = 1 if won else -1
                elif won == last_result:
                    # Streak continues
                    if won:
                        current_streak += 1
                    else:
                        current_streak -= 1
                else:
                    # Streak broken
                    break

            return float(current_streak)

    def compute_h2h_stats(
        self,
        home_team_id: int,
        away_team_id: int,
        as_of_date: str,
        window: int = 5
    ) -> Tuple[float, float]:
        """Compute head-to-head statistics.

        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            as_of_date: Compute features as of this date (exclusive)
            window: Number of previous meetings to consider

        Returns:
            Tuple of (h2h_win_pct, h2h_pt_diff) from home team's perspective
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            # Find previous meetings between these two teams
            query = """
            SELECT
                home_team_id,
                home_team_score,
                visitor_team_score
            FROM games
            WHERE game_date < ?
              AND (
                  (home_team_id = ? AND visitor_team_id = ?)
                  OR (home_team_id = ? AND visitor_team_id = ?)
              )
              AND status = 'Final'
              AND home_team_score IS NOT NULL
              AND visitor_team_score IS NOT NULL
            ORDER BY game_date DESC
            LIMIT ?
            """

            cursor.execute(query, (
                as_of_date,
                home_team_id, away_team_id,
                away_team_id, home_team_id,
                window
            ))
            meetings = cursor.fetchall()

            if not meetings:
                # No previous meetings, return neutral
                return 0.5, 0.0

            # Calculate home team's performance in these meetings
            wins = 0
            pt_diffs = []

            for row in meetings:
                meeting_home_id = row[0]
                meeting_home_score = row[1]
                meeting_away_score = row[2]

                if meeting_home_id == home_team_id:
                    # Current home team was home in that meeting
                    if meeting_home_score > meeting_away_score:
                        wins += 1
                    pt_diffs.append(meeting_home_score - meeting_away_score)
                else:
                    # Current home team was away in that meeting
                    if meeting_away_score > meeting_home_score:
                        wins += 1
                    pt_diffs.append(meeting_away_score - meeting_home_score)

            h2h_win_pct = wins / len(meetings)
            h2h_pt_diff = sum(pt_diffs) / len(pt_diffs)

            return h2h_win_pct, h2h_pt_diff
