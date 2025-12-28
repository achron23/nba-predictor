"""Feature engineering module for NBA Predictor v1.1.

This module implements time-aware feature engineering with strict anti-leakage guarantees.

CRITICAL RULE: For a game on date D, only use data from dates < D
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from . import config


def engineer_all_features(df: pd.DataFrame, feature_config: Dict) -> pd.DataFrame:
    """Main feature engineering orchestration function.

    Applies all feature engineering steps in sequence while maintaining
    strict temporal causality to prevent data leakage.

    Args:
        df: Raw games DataFrame sorted by GAME_DATE_EST
        feature_config: Configuration dict with keys:
            - rolling_window: int, number of games for rolling stats
            - h2h_window: int, number of head-to-head games to consider

    Returns:
        DataFrame with all engineered features added
    """
    print("  Sorting by date to ensure temporal order...")
    df = df.sort_values(config.GAME_DATE_COL).copy()

    print("  Adding season features...")
    df = add_season_features(df)

    print("  Computing rolling win percentages...")
    df = add_rolling_win_percentage(df, feature_config['rolling_window'])

    print("  Computing rolling performance statistics...")
    df = add_rolling_performance_stats(df, feature_config['rolling_window'])

    print("  Computing rest features...")
    df = add_rest_features(df)

    print("  Computing win/loss streaks...")
    df = add_streak_features(df)

    print("  Computing head-to-head statistics...")
    df = add_head_to_head_features(df, feature_config['h2h_window'])

    # Clean up temporary index column if it exists
    if '_game_idx' in df.columns:
        df = df.drop(columns=['_game_idx'])

    return df


def add_season_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add season feature (already exists in dataset).

    The SEASON column is already present in the raw data, so we just
    ensure it's included for one-hot encoding during model training.

    Args:
        df: Games DataFrame

    Returns:
        DataFrame with SEASON column ensured
    """
    if config.SEASON_COL not in df.columns:
        raise ValueError(f"Expected column '{config.SEASON_COL}' not found in dataset")

    return df


def create_team_game_level_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert game-level (1 row per game) to team-game-level (2 rows per game).

    This transformation enables groupby operations on team_id for computing
    rolling statistics from each team's perspective.

    Args:
        df: Game-level DataFrame

    Returns:
        Team-game-level DataFrame with one row per team per game
    """
    # Columns to select (include _game_idx if it exists)
    cols_to_select = [
        config.GAME_DATE_COL, config.HOME_TEAM_COL, config.VISITOR_TEAM_COL,
        config.HOME_PTS_COL, config.AWAY_PTS_COL, config.LABEL_COL,
        'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home'
    ]
    if '_game_idx' in df.columns:
        cols_to_select.append('_game_idx')

    # Home games perspective
    home_games = df[cols_to_select].copy()
    home_games['team_id'] = home_games[config.HOME_TEAM_COL]
    home_games['opponent_id'] = home_games[config.VISITOR_TEAM_COL]
    home_games['is_home'] = 1
    home_games['team_points'] = home_games[config.HOME_PTS_COL]
    home_games['opponent_points'] = home_games[config.AWAY_PTS_COL]
    home_games['won'] = home_games[config.LABEL_COL]
    home_games['fg_pct'] = home_games['FG_PCT_home']
    home_games['ft_pct'] = home_games['FT_PCT_home']
    home_games['fg3_pct'] = home_games['FG3_PCT_home']
    home_games['point_diff'] = home_games['team_points'] - home_games['opponent_points']

    # Away games perspective - include _game_idx and FG_PCT columns
    cols_to_select_away = [
        config.GAME_DATE_COL, config.HOME_TEAM_COL, config.VISITOR_TEAM_COL,
        config.HOME_PTS_COL, config.AWAY_PTS_COL, config.LABEL_COL,
        'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away'
    ]
    if '_game_idx' in df.columns:
        cols_to_select_away.append('_game_idx')

    away_games = df[cols_to_select_away].copy()
    away_games['team_id'] = away_games[config.VISITOR_TEAM_COL]
    away_games['opponent_id'] = away_games[config.HOME_TEAM_COL]
    away_games['is_home'] = 0
    away_games['team_points'] = away_games[config.AWAY_PTS_COL]
    away_games['opponent_points'] = away_games[config.HOME_PTS_COL]
    away_games['won'] = 1 - away_games[config.LABEL_COL]
    away_games['fg_pct'] = away_games['FG_PCT_away']
    away_games['ft_pct'] = away_games['FT_PCT_away']
    away_games['fg3_pct'] = away_games['FG3_PCT_away']
    away_games['point_diff'] = away_games['team_points'] - away_games['opponent_points']

    # Combine and sort
    team_games = pd.concat([home_games, away_games], ignore_index=True)
    team_games = team_games.sort_values(['team_id', config.GAME_DATE_COL]).reset_index(drop=True)

    return team_games


def add_rolling_win_percentage(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Add rolling win percentage features for home and away teams.

    Computes each team's win percentage over their last N games (excluding
    the current game to prevent leakage).

    Args:
        df: Games DataFrame
        window: Number of previous games to consider

    Returns:
        DataFrame with added columns:
            - home_win_pct_L10: Home team's rolling win %
            - away_win_pct_L10: Away team's rolling win %
    """
    # Add unique index to track games
    df = df.reset_index(drop=True)
    df['_game_idx'] = df.index

    # Convert to team-level
    team_games = create_team_game_level_data(df)

    # Compute rolling win % with shift(1) to exclude current game (anti-leakage)
    team_games['win_pct_LN'] = (
        team_games
        .groupby('team_id')['won']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    # Pivot back to game level
    # Match home team stats
    home_stats = team_games[team_games['is_home'] == 1][[
        '_game_idx', 'win_pct_LN'
    ]].rename(columns={'win_pct_LN': f'home_win_pct_L{window}'})

    # Match away team stats
    away_stats = team_games[team_games['is_home'] == 0][[
        '_game_idx', 'win_pct_LN'
    ]].rename(columns={'win_pct_LN': f'away_win_pct_L{window}'})

    # Merge back to original df using game index
    df = df.merge(home_stats, on='_game_idx', how='left')
    df = df.merge(away_stats, on='_game_idx', how='left')
    df = df.drop(columns=['_game_idx'])

    return df


def add_rolling_performance_stats(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Add rolling performance statistics (shooting percentages, point differential).

    Computes rolling averages of:
    - Field goal percentage (FG%)
    - 3-point percentage (FG3%)
    - Free throw percentage (FT%)
    - Point differential

    Args:
        df: Games DataFrame
        window: Number of previous games to consider

    Returns:
        DataFrame with added rolling performance columns
    """
    # Add unique index
    if '_game_idx' not in df.columns:
        df = df.reset_index(drop=True)
        df['_game_idx'] = df.index

    team_games = create_team_game_level_data(df)

    # Compute rolling stats with shift(1) for anti-leakage
    stats_to_roll = ['fg_pct', 'fg3_pct', 'ft_pct', 'point_diff']

    for stat in stats_to_roll:
        team_games[f'{stat}_LN'] = (
            team_games
            .groupby('team_id')[stat]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    # Pivot back to game level using game index
    home_stats = team_games[team_games['is_home'] == 1][[
        '_game_idx', 'fg_pct_LN', 'fg3_pct_LN', 'ft_pct_LN', 'point_diff_LN'
    ]].rename(columns={
        'fg_pct_LN': f'home_fg_pct_L{window}',
        'fg3_pct_LN': f'home_fg3_pct_L{window}',
        'ft_pct_LN': f'home_ft_pct_L{window}',
        'point_diff_LN': f'home_pt_diff_L{window}'
    })

    away_stats = team_games[team_games['is_home'] == 0][[
        '_game_idx', 'fg_pct_LN', 'fg3_pct_LN', 'ft_pct_LN', 'point_diff_LN'
    ]].rename(columns={
        'fg_pct_LN': f'away_fg_pct_L{window}',
        'fg3_pct_LN': f'away_fg3_pct_L{window}',
        'ft_pct_LN': f'away_ft_pct_L{window}',
        'point_diff_LN': f'away_pt_diff_L{window}'
    })

    df = df.merge(home_stats, on='_game_idx', how='left')
    df = df.merge(away_stats, on='_game_idx', how='left')

    return df


def add_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rest features (days since last game, back-to-back indicators).

    Computes:
    - rest_days: Number of days since team's last game
    - back_to_back: Binary indicator if playing on consecutive days

    Args:
        df: Games DataFrame

    Returns:
        DataFrame with added rest feature columns
    """
    # Add unique index
    if '_game_idx' not in df.columns:
        df = df.reset_index(drop=True)
        df['_game_idx'] = df.index

    team_games = create_team_game_level_data(df)

    # Compute days since last game
    team_games['prev_game_date'] = (
        team_games
        .groupby('team_id')[config.GAME_DATE_COL]
        .shift(1)
    )

    team_games['rest_days'] = (
        (team_games[config.GAME_DATE_COL] - team_games['prev_game_date']).dt.days
    )

    # Back-to-back indicator (1 day rest)
    team_games['back_to_back'] = (team_games['rest_days'] == 1).astype(int)

    # Fill NaN for first game (use median rest days as default)
    median_rest = team_games['rest_days'].median()
    team_games['rest_days'] = team_games['rest_days'].fillna(median_rest)

    # Pivot back using game index
    home_stats = team_games[team_games['is_home'] == 1][[
        '_game_idx', 'rest_days', 'back_to_back'
    ]].rename(columns={
        'rest_days': 'home_rest_days',
        'back_to_back': 'home_back_to_back'
    })

    away_stats = team_games[team_games['is_home'] == 0][[
        '_game_idx', 'rest_days', 'back_to_back'
    ]].rename(columns={
        'rest_days': 'away_rest_days',
        'back_to_back': 'away_back_to_back'
    })

    df = df.merge(home_stats, on='_game_idx', how='left')
    df = df.merge(away_stats, on='_game_idx', how='left')

    return df


def add_streak_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add win/loss streak features.

    Computes current streak:
    - Positive values = winning streak length
    - Negative values = losing streak length
    - 0 = first game

    Args:
        df: Games DataFrame

    Returns:
        DataFrame with streak columns
    """
    # Add unique index
    if '_game_idx' not in df.columns:
        df = df.reset_index(drop=True)
        df['_game_idx'] = df.index

    team_games = create_team_game_level_data(df)

    def compute_streak(wins_series):
        """Compute streak from binary win series."""
        streaks = []
        current_streak = 0

        for won in wins_series:
            if pd.isna(won):
                streaks.append(0)
            elif won == 1:
                current_streak = current_streak + 1 if current_streak >= 0 else 1
                streaks.append(current_streak)
            else:
                current_streak = current_streak - 1 if current_streak <= 0 else -1
                streaks.append(current_streak)

        return pd.Series(streaks, index=wins_series.index)

    # Shift to exclude current game
    team_games['won_shifted'] = team_games.groupby('team_id')['won'].shift(1)

    # Compute streak
    team_games['win_streak'] = (
        team_games
        .groupby('team_id')['won_shifted']
        .transform(compute_streak)
    )

    # Pivot back using game index
    home_stats = team_games[team_games['is_home'] == 1][[
        '_game_idx', 'win_streak'
    ]].rename(columns={'win_streak': 'home_win_streak'})

    away_stats = team_games[team_games['is_home'] == 0][[
        '_game_idx', 'win_streak'
    ]].rename(columns={'win_streak': 'away_win_streak'})

    df = df.merge(home_stats, on='_game_idx', how='left')
    df = df.merge(away_stats, on='_game_idx', how='left')

    return df


def add_head_to_head_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add head-to-head matchup history features.

    For each game, computes home team's performance in the last N meetings
    with the away team.

    Args:
        df: Games DataFrame
        window: Number of previous matchups to consider

    Returns:
        DataFrame with H2H feature columns
    """
    df = df.sort_values(config.GAME_DATE_COL).copy()

    # Create matchup key (sorted team IDs to find all meetings)
    df['matchup_key'] = df.apply(
        lambda row: tuple(sorted([row[config.HOME_TEAM_COL], row[config.VISITOR_TEAM_COL]])),
        axis=1
    )

    h2h_win_pcts = []
    h2h_pt_diffs = []

    for idx, row in df.iterrows():
        home_team = row[config.HOME_TEAM_COL]
        away_team = row[config.VISITOR_TEAM_COL]
        current_date = row[config.GAME_DATE_COL]

        # Find previous meetings
        prev_meetings = df[
            (df['matchup_key'] == row['matchup_key']) &
            (df[config.GAME_DATE_COL] < current_date)
        ].tail(window)

        if len(prev_meetings) == 0:
            h2h_win_pcts.append(np.nan)
            h2h_pt_diffs.append(np.nan)
        else:
            # Compute home team's performance in these meetings
            home_wins = 0
            pt_diffs = []

            for _, meeting in prev_meetings.iterrows():
                if meeting[config.HOME_TEAM_COL] == home_team:
                    # Current home team was home in that meeting
                    home_wins += meeting[config.LABEL_COL]
                    pt_diffs.append(meeting[config.HOME_PTS_COL] - meeting[config.AWAY_PTS_COL])
                else:
                    # Current home team was away in that meeting
                    home_wins += (1 - meeting[config.LABEL_COL])
                    pt_diffs.append(meeting[config.AWAY_PTS_COL] - meeting[config.HOME_PTS_COL])

            h2h_win_pcts.append(home_wins / len(prev_meetings))
            h2h_pt_diffs.append(np.mean(pt_diffs))

    df['home_h2h_win_pct'] = h2h_win_pcts
    df['home_h2h_pt_diff'] = h2h_pt_diffs

    # Fill NaN with neutral values
    df['home_h2h_win_pct'] = df['home_h2h_win_pct'].fillna(0.5)
    df['home_h2h_pt_diff'] = df['home_h2h_pt_diff'].fillna(0.0)

    # Drop temporary column
    df = df.drop(columns=['matchup_key'])

    return df


def validate_engineered_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Validate engineered features for quality and detect potential issues.

    Checks for:
    - Infinite values
    - High percentage of NaN values
    - Suspicious statistics

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame

    Raises:
        ValueError: If critical validation fails
    """
    print("\nValidating engineered features...")

    feature_cols = config.ENGINEERED_FEATURE_COLS

    for name, df in [("Train", train_df), ("Test", test_df)]:
        # Check for inf values
        num_cols = df[feature_cols].select_dtypes(include=[np.number])
        inf_count = np.isinf(num_cols).sum().sum()

        if inf_count > 0:
            raise ValueError(f"{name} set has {inf_count} infinite values!")

        # Check NaN percentage
        nan_pcts = df[feature_cols].isna().sum() / len(df) * 100
        high_nan = nan_pcts[nan_pcts > 10]

        if len(high_nan) > 0:
            print(f"  Warning: {name} has features with >10% NaN:")
            for feat, pct in high_nan.items():
                print(f"    {feat}: {pct:.1f}%")

        # Summary stats
        print(f"\n  {name} set feature statistics:")
        print(f"    Total features: {len(feature_cols)}")
        print(f"    Numeric features: {len(num_cols.columns)}")
        print(f"    Avg NaN%: {nan_pcts.mean():.2f}%")
        print(f"    Max NaN%: {nan_pcts.max():.2f}%")

    print("\n[OK] Feature validation passed")
