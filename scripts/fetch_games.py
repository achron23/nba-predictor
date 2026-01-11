"""Fetch NBA games from balldontlie.io API and store in SQLite database.

Usage:
    # Fetch yesterday's games
    uv run python scripts/fetch_games.py --yesterday

    # Fetch specific date
    uv run python scripts/fetch_games.py --date 2024-01-15

    # Fetch date range
    uv run python scripts/fetch_games.py --start-date 2024-01-01 --end-date 2024-01-31

    # Fetch entire season
    uv run python scripts/fetch_games.py --season 2024
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nba_predictor import config
from nba_predictor.data_fetcher.balldontlie_client import BallDontLieClient
from nba_predictor.db.connection import get_db_connection


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch NBA games from balldontlie.io API"
    )

    # Date options (mutually exclusive)
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument(
        "--yesterday",
        action="store_true",
        help="Fetch yesterday's games"
    )
    date_group.add_argument(
        "--date",
        type=str,
        help="Fetch games for specific date (YYYY-MM-DD)"
    )
    date_group.add_argument(
        "--start-date",
        type=str,
        help="Start date for range (YYYY-MM-DD), requires --end-date"
    )
    date_group.add_argument(
        "--season",
        type=int,
        help="Fetch all games for season (e.g., 2024 for 2024-25 season)"
    )

    # End date for range
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for range (YYYY-MM-DD), requires --start-date"
    )

    args = parser.parse_args()

    # Validate start-date / end-date pair
    if args.start_date and not args.end_date:
        parser.error("--start-date requires --end-date")
    if args.end_date and not args.start_date:
        parser.error("--end-date requires --start-date")

    return args


def insert_game(conn: sqlite3.Connection, game: dict) -> bool:
    """Insert a game into the database.

    Args:
        conn: Database connection
        game: Game dictionary matching database schema

    Returns:
        True if inserted, False if already exists
    """
    cursor = conn.cursor()

    # Use INSERT OR IGNORE to skip duplicates
    query = """
    INSERT OR IGNORE INTO games (
        game_id, game_date, season, home_team_id, visitor_team_id,
        home_team_score, visitor_team_score, home_win,
        postseason, status
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    params = (
        game["game_id"],
        game["game_date"],
        game["season"],
        game["home_team_id"],
        game["visitor_team_id"],
        game["home_team_score"],
        game["visitor_team_score"],
        game["home_win"],
        game["postseason"],
        game["status"],
    )

    cursor.execute(query, params)
    return cursor.rowcount > 0


def print_summary(stats: dict):
    """Print summary of fetch operation."""
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Games fetched from API: {stats['fetched']}")
    print(f"  Games inserted to DB: {stats['inserted']}")
    print(f"  Games already in DB: {stats['skipped']}")
    print(f"  Errors: {stats['errors']}")
    print()

    if stats['inserted'] > 0:
        print(f"[OK] Successfully added {stats['inserted']} new games")
    elif stats['skipped'] > 0:
        print(f"[INFO] All games already exist in database")
    else:
        print(f"[WARNING] No games were added")

    print("=" * 60)


def main():
    """Main execution function."""
    args = parse_args()

    print("=" * 60)
    print("NBA Predictor v2.0 - Data Ingestion")
    print("=" * 60)
    print()

    # Determine date range
    if args.yesterday:
        date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"Fetching games for yesterday ({date})")
        start_date = end_date = date
        season = None

    elif args.date:
        start_date = end_date = args.date
        season = None
        print(f"Fetching games for {args.date}")

    elif args.start_date:
        start_date = args.start_date
        end_date = args.end_date
        season = None
        print(f"Fetching games from {start_date} to {end_date}")

    elif args.season:
        season = args.season
        start_date = None
        end_date = None
        print(f"Fetching all games for season {season}")

    # Initialize API client
    try:
        client = BallDontLieClient()
        print("[OK] API client initialized")
    except ValueError as e:
        print(f"[ERROR] {e}")
        print()
        print("Please set your API key:")
        print("  export BALLDONTLIE_API_KEY='your_key_here'")
        print()
        print("Get a free API key at https://www.balldontlie.io")
        sys.exit(1)

    # Check database exists
    if not config.DB_PATH.exists():
        print(f"[ERROR] Database not found at {config.DB_PATH}")
        print()
        print("Please initialize the database first:")
        print("  uv run python scripts/init_db.py")
        sys.exit(1)

    print(f"[OK] Database found at {config.DB_PATH}")
    print()

    # Fetch games from API
    print("Fetching games from balldontlie.io...")
    try:
        if season:
            games = client.fetch_games_for_season(season)
        else:
            games = client.fetch_games_by_date_range(start_date, end_date, season=season)
    except Exception as e:
        print(f"[ERROR] Failed to fetch games: {e}")
        sys.exit(1)

    print(f"[OK] Fetched {len(games)} games from API")
    print()

    # Insert games into database
    print("Inserting games into database...")
    stats = {
        "fetched": len(games),
        "inserted": 0,
        "skipped": 0,
        "errors": 0,
    }

    with get_db_connection(config.DB_PATH) as conn:
        for game in games:
            try:
                # Parse game to database format
                parsed_game = client.parse_game_to_db_format(game)

                # Insert game
                inserted = insert_game(conn, parsed_game)

                if inserted:
                    stats["inserted"] += 1
                else:
                    stats["skipped"] += 1

            except Exception as e:
                stats["errors"] += 1
                print(f"  Error inserting game {game.get('id', 'unknown')}: {e}")

        # Commit transaction
        conn.commit()

    # Print summary
    print_summary(stats)


if __name__ == "__main__":
    # Load .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    main()
