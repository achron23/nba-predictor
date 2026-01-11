"""Setup script for v2.0 - Initialize database and fetch initial data.

This script guides you through the v2.0 setup process:
1. Initialize SQLite database
2. Fetch historical data from balldontlie.io
3. Verify data ingestion
4. Ready for training

Usage:
    uv run python scripts/setup_v2_0.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nba_predictor import config
from src.nba_predictor.db.schema import init_db, get_db_stats
from src.nba_predictor.data_fetcher.balldontlie_client import BallDontLieClient
from datetime import datetime, timedelta

import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_api_key() -> str:
    """Check for API key in environment.

    Returns:
        API key string

    Raises:
        ValueError: If API key not found
    """
    api_key = os.getenv(config.BALLDONTLIE_API_KEY_ENV_VAR)

    if not api_key:
        raise ValueError(
            f"API key not found. Please set {config.BALLDONTLIE_API_KEY_ENV_VAR}:\n"
            f"  1. Rename .env.example to .env\n"
            f"  2. Or export {config.BALLDONTLIE_API_KEY_ENV_VAR}=your_key_here"
        )

    return api_key


def main():
    """Main setup workflow."""
    print("=" * 70)
    print("NBA PREDICTOR v2.0 SETUP")
    print("=" * 70)
    print()

    # Step 1: Check API key
    print("Step 1: Checking API key...")
    try:
        api_key = check_api_key()
        print("[OK] API key found")
    except ValueError as e:
        print(f"[ERROR] {e}")
        return
    print()

    # Step 2: Initialize database
    print("Step 2: Initializing database...")
    if config.DB_PATH.exists():
        print(f"[WARNING] Database already exists at {config.DB_PATH}")
        response = input("  Recreate database? (y/N): ").strip().lower()
        if response == 'y':
            config.DB_PATH.unlink()
            print("  Deleted existing database")
        else:
            print("  Keeping existing database")

    if not config.DB_PATH.exists():
        init_db(config.DB_PATH)
        print(f"[OK] Database initialized at {config.DB_PATH}")
    print()

    # Step 3: Fetch historical data
    print("Step 3: Fetching historical data from balldontlie.io...")
    print("[WARNING] This will take several minutes due to rate limiting (5 req/min)")
    print()

    seasons_to_fetch = [2023, 2024 , 2025]  # Last 2 seasons
    print(f"  Fetching seasons: {seasons_to_fetch}")
    response = input("  Continue? (Y/n): ").strip().lower()

    if response == 'n':
        print("Skipping data fetch. You can run manually later:")
        print("  uv run python scripts/fetch_games.py --season 2023")
        print("  uv run python scripts/fetch_games.py --season 2024")
        return

    client = BallDontLieClient(api_key)

    for season in seasons_to_fetch:
        print()
        print(f"Fetching season {season}...")
        print(f"  This will take ~5-10 minutes per season...")

        try:
            # Fetch games
            games = client.fetch_games_for_season(season)
            print(f"  [OK] Fetched {len(games)} games from API")

            # Insert into database
            from src.nba_predictor.db.connection import get_db_connection

            stats = {
                "fetched": len(games),
                "inserted": 0,
                "skipped": 0,
                "errors": 0,
            }

            with get_db_connection(config.DB_PATH) as conn:
                cursor = conn.cursor()

                for game in games:
                    try:
                        parsed_game = client.parse_game_to_db_format(game)

                        # Insert or ignore duplicate
                        query = """
                        INSERT OR IGNORE INTO games (
                            game_id, game_date, season, home_team_id, visitor_team_id,
                            home_team_score, visitor_team_score, home_win,
                            postseason, status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """

                        params = (
                            parsed_game["game_id"],
                            parsed_game["game_date"],
                            parsed_game["season"],
                            parsed_game["home_team_id"],
                            parsed_game["visitor_team_id"],
                            parsed_game["home_team_score"],
                            parsed_game["visitor_team_score"],
                            parsed_game["home_win"],
                            parsed_game["postseason"],
                            parsed_game["status"],
                        )

                        cursor.execute(query, params)

                        if cursor.rowcount > 0:
                            stats["inserted"] += 1
                        else:
                            stats["skipped"] += 1

                    except Exception as e:
                        stats["errors"] += 1
                        logger.warning(f"Error inserting game: {e}")

                conn.commit()

            print(f"  [OK] Season {season} complete:")
            print(f"      Fetched: {stats['fetched']}")
            print(f"      Inserted: {stats['inserted']}")
            print(f"      Skipped: {stats['skipped']}")
            if stats['errors'] > 0:
                print(f"      Errors: {stats['errors']}")

        except Exception as e:
            print(f"  [ERROR] Error fetching season {season}: {e}")
            print("  You can retry manually:")
            print(f"    uv run python scripts/fetch_games.py --season {season}")

    print()

    # Step 4: Verify data
    print("Step 4: Verifying data...")
    stats = get_db_stats(config.DB_PATH)
    print(f"[OK] Database contains {stats['total_games']} games")
    print(f"  Completed games: {stats['completed_games']}")
    print(f"  Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
    print(f"  Unique teams: {stats['unique_teams']}")
    print()

    if stats['completed_games'] < 100:
        print("[WARNING] Very few completed games in database")
        print("  You may need to fetch more data for training.")
        print()

    # Step 5: Next steps
    print("=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Train v2.0 model:")
    print("     uv run python -m nba_predictor.train_v2_0")
    print()
    print("  2. Test prediction:")
    print("     uv run python test_api.py")
    print()
    print("  3. Start API server:")
    print("     uv run python run_api.py")
    print()
    print("To update data regularly, run:")
    print("  uv run python scripts/fetch_games.py --yesterday")
    print("=" * 70)


if __name__ == "__main__":
    # Load .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    main()
