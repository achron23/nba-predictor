"""Initialize the NBA Predictor v2.0 database.

Usage:
    uv run python scripts/init_db.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nba_predictor import config
from nba_predictor.db.schema import init_db, verify_schema, get_db_stats


def main():
    """Initialize database and verify schema."""
    print("=" * 60)
    print("NBA Predictor v2.0 - Database Initialization")
    print("=" * 60)
    print()

    db_path = config.DB_PATH

    # Check if database already exists
    if db_path.exists():
        print(f"⚠ Database already exists at {db_path}")
        response = input("Do you want to reinitialize? This will NOT delete existing data. (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Initialize database
    print(f"Initializing database at {db_path}...")
    try:
        init_db(db_path)
    except Exception as e:
        print(f"✗ Failed to initialize database: {e}")
        sys.exit(1)

    print()

    # Verify schema
    print("Verifying schema...")
    if not verify_schema(db_path):
        print("✗ Schema verification failed")
        sys.exit(1)

    print()

    # Show database stats
    print("Database statistics:")
    stats = get_db_stats(db_path)
    print(f"  Total games: {stats['total_games']}")
    print(f"  Unique teams: {stats['unique_teams']}")
    if stats['date_range']['min']:
        print(f"  Date range: {stats['date_range']['min']} to {stats['date_range']['max']}")
    if stats['games_by_season']:
        print(f"  Seasons: {list(stats['games_by_season'].keys())}")

    print()
    print("=" * 60)
    print("✓ Database initialized successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Get a balldontlie.io API key from https://www.balldontlie.io")
    print("  2. Set environment variable: export BALLDONTLIE_API_KEY='your_key'")
    print("  3. Fetch games: uv run python scripts/fetch_games.py --season 2024")


if __name__ == "__main__":
    main()
