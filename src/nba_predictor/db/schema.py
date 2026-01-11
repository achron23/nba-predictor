"""SQLite database schema for NBA Predictor v2.0.

This module defines the database structure for storing NBA game data
from balldontlie.io API. The schema is intentionally simple with a single
`games` table containing only essential data needed for feature computation.

Key design principles:
- Single source of truth: games table stores raw game results
- No pre-computed features: features computed on-demand for flexibility
- balldontlie team IDs: no ID mapping complexity in v2.0
- No shooting stats: v2.0 focuses on win/loss and point differential
"""

import sqlite3
from pathlib import Path
from typing import Optional


# SQL schema for games table
CREATE_GAMES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS games (
    game_id INTEGER PRIMARY KEY,
    game_date TEXT NOT NULL,
    season INTEGER NOT NULL,
    home_team_id INTEGER NOT NULL,
    visitor_team_id INTEGER NOT NULL,
    home_team_score INTEGER,
    visitor_team_score INTEGER,
    home_win INTEGER,
    postseason INTEGER DEFAULT 0,
    status TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(game_id)
);
"""

# Indexes for efficient queries (critical for on-demand feature computation)
CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_game_date ON games(game_date);",
    "CREATE INDEX IF NOT EXISTS idx_home_team_date ON games(home_team_id, game_date);",
    "CREATE INDEX IF NOT EXISTS idx_visitor_team_date ON games(visitor_team_id, game_date);",
]


def init_db(db_path: Path) -> None:
    """Initialize the database with schema and indexes.

    Args:
        db_path: Path to SQLite database file

    Raises:
        sqlite3.Error: If database creation fails
    """
    # Create parent directory if it doesn't exist
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Connect and create schema
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()

        # Create games table
        cursor.execute(CREATE_GAMES_TABLE_SQL)

        # Create indexes
        for index_sql in CREATE_INDEXES_SQL:
            cursor.execute(index_sql)

        conn.commit()

        # Verify table creation
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='games';")
        if not cursor.fetchone():
            raise sqlite3.Error("Failed to create games table")

        print(f"✓ Database initialized successfully at {db_path}")
        print(f"✓ Created table: games")
        print(f"✓ Created {len(CREATE_INDEXES_SQL)} indexes")

    finally:
        conn.close()


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Get a connection to the database.

    Args:
        db_path: Path to SQLite database file

    Returns:
        SQLite connection object

    Raises:
        sqlite3.Error: If connection fails
    """
    if not db_path.exists():
        raise sqlite3.Error(f"Database not found at {db_path}. Run init_db() first.")

    conn = sqlite3.connect(str(db_path))
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON;")
    # Return rows as dictionaries
    conn.row_factory = sqlite3.Row
    return conn


def verify_schema(db_path: Path) -> bool:
    """Verify that the database schema is correct.

    Args:
        db_path: Path to SQLite database file

    Returns:
        True if schema is valid, False otherwise
    """
    try:
        conn = get_connection(db_path)
        cursor = conn.cursor()

        # Check table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='games';")
        if not cursor.fetchone():
            print("✗ Table 'games' not found")
            return False

        # Check indexes exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='games';")
        indexes = [row[0] for row in cursor.fetchall()]
        expected_indexes = ['idx_game_date', 'idx_home_team_date', 'idx_visitor_team_date']

        for idx in expected_indexes:
            if idx not in indexes:
                print(f"✗ Index '{idx}' not found")
                return False

        # Check columns
        cursor.execute("PRAGMA table_info(games);")
        columns = [row[1] for row in cursor.fetchall()]
        required_columns = [
            'game_id', 'game_date', 'season', 'home_team_id',
            'visitor_team_id', 'home_team_score', 'visitor_team_score',
            'home_win', 'postseason', 'status', 'created_at'
        ]

        for col in required_columns:
            if col not in columns:
                print(f"✗ Column '{col}' not found")
                return False

        conn.close()
        print("✓ Schema verification passed")
        return True

    except sqlite3.Error as e:
        print(f"✗ Schema verification failed: {e}")
        return False


def get_db_stats(db_path: Path) -> dict:
    """Get basic statistics about the database.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Dictionary with database statistics
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    stats = {}

    # Total games
    cursor.execute("SELECT COUNT(*) FROM games;")
    stats['total_games'] = cursor.fetchone()[0]

    # Games by season
    cursor.execute("SELECT season, COUNT(*) as count FROM games GROUP BY season ORDER BY season;")
    stats['games_by_season'] = {row[0]: row[1] for row in cursor.fetchall()}

    # Date range
    cursor.execute("SELECT MIN(game_date), MAX(game_date) FROM games;")
    min_date, max_date = cursor.fetchone()
    stats['date_range'] = {'min': min_date, 'max': max_date}

    # Unique teams
    cursor.execute("SELECT COUNT(DISTINCT home_team_id) FROM games;")
    stats['unique_teams'] = cursor.fetchone()[0]

    conn.close()
    return stats
