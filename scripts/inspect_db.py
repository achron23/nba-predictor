"""Inspect and verify the NBA Predictor database content.

This script provides a comprehensive view of the database:
- Database statistics (row counts, date ranges)
- Table schema information
- Sample data previews
- Quick queries for verification

Usage:
    uv run python scripts/inspect_db.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nba_predictor import config
from nba_predictor.db.schema import get_db_stats, verify_schema, get_connection
from nba_predictor.db.connection import execute_query


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show_database_info(db_path: Path):
    """Show database file information."""
    print_section("DATABASE FILE INFO")
    
    if not db_path.exists():
        print(f"✗ Database not found at: {db_path}")
        return False
    
    file_size = db_path.stat().st_size / (1024 * 1024)  # MB
    print(f"Location: {db_path}")
    print(f"Size: {file_size:.2f} MB")
    print(f"Exists: ✓")
    
    return True


def show_schema_info(db_path: Path):
    """Show database schema information."""
    print_section("SCHEMA VERIFICATION")
    
    if verify_schema(db_path):
        print("\nTable Structure:")
        conn = get_connection(db_path)
        cursor = conn.cursor()
        
        # Get table info
        cursor.execute("PRAGMA table_info(games);")
        columns = cursor.fetchall()
        
        print("\n  games table:")
        print(f"    {'Column':<25} {'Type':<15} {'Nullable'}")
        print("    " + "-" * 60)
        for col in columns:
            nullable = "NULL" if col[3] == 0 else "NOT NULL"
            print(f"    {col[1]:<25} {col[2]:<15} {nullable}")
        
        # Get indexes
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='games';")
        indexes = cursor.fetchall()
        print(f"\n  Indexes ({len(indexes)}):")
        for idx in indexes:
            print(f"    - {idx[0]}")
        
        conn.close()
    else:
        print("✗ Schema verification failed")
        return False
    
    return True


def show_statistics(db_path: Path):
    """Show database statistics."""
    print_section("DATABASE STATISTICS")
    
    stats = get_db_stats(db_path)
    
    print(f"Total games: {stats['total_games']:,}")
    print(f"Unique teams: {stats['unique_teams']}")
    
    if stats['date_range']['min']:
        print(f"Date range: {stats['date_range']['min']} to {stats['date_range']['max']}")
    else:
        print("Date range: No data")
    
    if stats['games_by_season']:
        print("\nGames by season:")
        for season in sorted(stats['games_by_season'].keys()):
            count = stats['games_by_season'][season]
            print(f"  {season}: {count:,} games")
    else:
        print("\nGames by season: No data")


def show_sample_data(db_path: Path, limit: int = 10):
    """Show sample data from the games table."""
    print_section(f"SAMPLE DATA (first {limit} rows)")
    
    query = """
        SELECT 
            game_id,
            game_date,
            season,
            home_team_id,
            visitor_team_id,
            home_team_score,
            visitor_team_score,
            home_win,
            postseason,
            status
        FROM games
        ORDER BY game_date DESC, game_id DESC
        LIMIT ?
    """
    
    rows = execute_query(db_path, query, (limit,))
    
    if not rows:
        print("No data found in games table")
        return
    
    # Print header
    print(f"\n{'ID':<10} {'Date':<12} {'Season':<8} {'Home':<8} {'Away':<8} {'Score':<12} {'Home Win':<10} {'Post':<6}")
    print("-" * 90)
    
    # Print rows
    for row in rows:
        score = f"{row['home_team_score']}-{row['visitor_team_score']}" if row['home_team_score'] else "NULL"
        home_win = "✓" if row['home_win'] == 1 else "✗" if row['home_win'] == 0 else "?"
        post = "✓" if row['postseason'] == 1 else ""
        
        print(f"{row['game_id']:<10} {row['game_date']:<12} {row['season']:<8} "
              f"{row['home_team_id']:<8} {row['visitor_team_id']:<8} {score:<12} "
              f"{home_win:<10} {post:<6}")


def show_recent_games(db_path: Path, limit: int = 5):
    """Show most recent games."""
    print_section(f"RECENT GAMES (last {limit})")
    
    query = """
        SELECT 
            game_date,
            season,
            home_team_id,
            visitor_team_id,
            home_team_score,
            visitor_team_score,
            home_win,
            status
        FROM games
        WHERE home_team_score IS NOT NULL
        ORDER BY game_date DESC
        LIMIT ?
    """
    
    rows = execute_query(db_path, query, (limit,))
    
    if not rows:
        print("No completed games found")
        return
    
    for row in rows:
        result = "W" if row['home_win'] == 1 else "L"
        print(f"{row['game_date']} (Season {row['season']}): "
              f"Team {row['home_team_id']} {row['home_team_score']} - "
              f"{row['visitor_team_score']} Team {row['visitor_team_id']} "
              f"[{result}]")


def show_team_summary(db_path: Path):
    """Show summary statistics by team."""
    print_section("TEAM SUMMARY (top 10 by games played)")
    
    query = """
        SELECT 
            home_team_id as team_id,
            COUNT(*) as total_games,
            SUM(CASE WHEN home_win = 1 THEN 1 ELSE 0 END) as home_wins,
            SUM(CASE WHEN home_win = 0 THEN 1 ELSE 0 END) as home_losses
        FROM games
        WHERE home_team_score IS NOT NULL
        GROUP BY home_team_id
        ORDER BY total_games DESC
        LIMIT 10
    """
    
    rows = execute_query(db_path, query)
    
    if not rows:
        print("No team data found")
        return
    
    print(f"\n{'Team ID':<10} {'Games':<10} {'Home W':<10} {'Home L':<10} {'Win %':<10}")
    print("-" * 60)
    
    for row in rows:
        total = row['total_games']
        wins = row['home_wins'] or 0
        losses = row['home_losses'] or 0
        win_pct = (wins / total * 100) if total > 0 else 0
        
        print(f"{row['team_id']:<10} {total:<10} {wins:<10} {losses:<10} {win_pct:.1f}%")


def show_data_quality(db_path: Path):
    """Show data quality metrics."""
    print_section("DATA QUALITY")
    
    queries = {
        "Total rows": "SELECT COUNT(*) FROM games",
        "Rows with scores": "SELECT COUNT(*) FROM games WHERE home_team_score IS NOT NULL",
        "Rows without scores": "SELECT COUNT(*) FROM games WHERE home_team_score IS NULL",
        "Regular season": "SELECT COUNT(*) FROM games WHERE postseason = 0",
        "Postseason": "SELECT COUNT(*) FROM games WHERE postseason = 1",
        "Completed games": "SELECT COUNT(*) FROM games WHERE status = 'Final' OR status IS NULL",
    }
    
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    try:
        for label, query in queries.items():
            cursor.execute(query)
            count = cursor.fetchone()[0]
            print(f"{label:<25} {count:,}")
    finally:
        conn.close()


def main():
    """Main inspection function."""
    print("=" * 70)
    print("NBA PREDICTOR DATABASE INSPECTION")
    print("=" * 70)
    
    db_path = config.DB_PATH
    
    # Check if database exists
    if not show_database_info(db_path):
        print("\n✗ Cannot proceed - database not found")
        sys.exit(1)
    
    # Show schema
    if not show_schema_info(db_path):
        print("\n✗ Schema verification failed")
        sys.exit(1)
    
    # Show statistics
    show_statistics(db_path)
    
    # Show data quality
    show_data_quality(db_path)
    
    # Show sample data
    show_sample_data(db_path, limit=10)
    
    # Show recent games
    show_recent_games(db_path, limit=5)
    
    # Show team summary
    show_team_summary(db_path)
    
    print("\n" + "=" * 70)
    print("✓ Database inspection complete")
    print("=" * 70)


if __name__ == "__main__":
    main()

