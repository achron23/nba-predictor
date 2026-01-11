# NBA Predictor v2.0 Quick Start Guide

## What's New in v2.0

v2.0 replaces static parquet files with **real-time data from balldontlie.io API** stored in a SQLite database. Features are now computed on-demand for each prediction.

### Key Differences from v1.1

- **Data Source:** balldontlie.io API (vs static CSV)
- **Storage:** SQLite database (vs parquet files)
- **Features:** 15 features (vs 21) - NO shooting stats (FG%, FT%, 3P%)
- **Team IDs:** balldontlie team IDs (simpler, no mapping)
- **Feature Computation:** On-demand from database (vs pre-computed in DataFrame)
- **Data Updates:** Manual via CLI scripts (no automated batch jobs in MVP)

---

## Setup (One-Time)

### 1. Ensure API Key is Set

You've already added your API key to `.env.example`. Now rename it:

```bash
# Option 1: Rename .env.example to .env
cp .env.example .env

# Option 2: Export as environment variable
export BALLDONTLIE_API_KEY=3349c5cc-9592-4cfe-8fa8-5b5fa28a36e3
```

### 2. Run Automated Setup (Recommended)

This script will:
- Initialize SQLite database
- Fetch 2023 and 2024 seasons (~5-10 minutes per season due to rate limiting)
- Verify data ingestion

```bash
uv run python scripts/setup_v2_0.py
```

**OR** Do it manually:

```bash
# Initialize database
uv run python scripts/init_db.py

# Fetch historical data (2023 season)
uv run python scripts/fetch_games.py --season 2023

# Fetch historical data (2024 season)
uv run python scripts/fetch_games.py --season 2024
```

### 3. Train v2.0 Model

Once you have data in the database, train the v2.0 model:

```bash
uv run python -m nba_predictor.train_v2_0
```

This will:
- Query all completed games from SQLite
- Compute features on-demand for each game
- Train LogisticRegression model (same as v1.1 but with v2.0 features)
- Evaluate on time-based test split
- Save model to `models/model_v2_0.joblib`

**Expected output:**
```
Training v2.0 model from SQLite database
Fetched X completed games
Computing features for Y games...
Training LogisticRegression model...
Test accuracy: ~0.65-0.70
Test log loss: ~0.60-0.65
Model saved to models/model_v2_0.joblib
```

---

## Daily Usage

### Update Data with Recent Games

Run this daily/weekly to fetch new games:

```bash
# Fetch yesterday's games
uv run python scripts/fetch_games.py --yesterday

# Fetch specific date
uv run python scripts/fetch_games.py --date 2025-01-15

# Fetch date range
uv run python scripts/fetch_games.py --start-date 2025-01-01 --end-date 2025-01-31
```

### Make Predictions

The predictor now supports **both v1.1 and v2.0** models:

**Using v2.0 model (database features):**

```python
from nba_predictor.api.predictor import NBAPredictor
from nba_predictor import config

# Initialize with v2.0 model (default)
predictor = NBAPredictor(
    model_path=config.MODEL_V2_0_PATH,
    use_database=True  # Use database features
)

# Predict
result = predictor.predict(
    home_team_id=10,  # balldontlie team ID
    away_team_id=14,  # balldontlie team ID
    season=2024,      # Optional
    game_date="2025-01-15"  # Optional
)

print(f"Home win probability: {result['p_home_win']:.2%}")
print(f"Model version: {result['model_version']}")  # v2.0
```

**Using v1.1 model (parquet features):**

```python
# Initialize with v1.1 model
predictor = NBAPredictor(
    model_path=config.MODEL_V1_1_PATH,
    use_database=False  # Use parquet features
)

# Predict (same API)
result = predictor.predict(
    home_team_id=1610612744,  # NBA official team ID
    away_team_id=1610612754,
)
```

### Start API Server

The API automatically uses v2.0 model if available:

```bash
uv run python run_api.py
```

Test it:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team_id": 10, "away_team_id": 14}'
```

---

## Architecture Overview

```
balldontlie.io API
        ↓
  [Manual Script]
        ↓
  SQLite Database
   (games table)
        ↓
  [On-Demand Feature Computer]
        ↓
    v2.0 Model
        ↓
     Prediction
```

### Key Components

1. **SQLite Database** (`data/db/nba_predictor.db`)
   - Single `games` table
   - Stores raw game results (no pre-computed features)
   - Indexed for fast team-date queries

2. **balldontlie API Client** (`src/nba_predictor/data_fetcher/balldontlie_client.py`)
   - Fetches game data from balldontlie.io
   - Handles pagination and rate limiting (5 req/min free tier)
   - Transforms API response to database schema

3. **Manual Ingest Scripts** (`scripts/fetch_games.py`)
   - CLI tool to fetch games manually
   - INSERT OR IGNORE for duplicate handling

4. **On-Demand Feature Computer** (`src/nba_predictor/db/feature_computer.py`)
   - Computes features from database queries
   - Anti-leakage: `WHERE game_date < prediction_date`
   - Returns None for insufficient data (falls back to defaults)

5. **Updated Predictor** (`src/nba_predictor/api/predictor.py`)
   - Supports both v1.1 (parquet) and v2.0 (database)
   - Dual initialization paths based on `use_database` parameter

---

## v2.0 Feature Set

v2.0 uses **15 features** (vs 21 in v1.1):

### Baseline (3)
- `home_team_id` - balldontlie team ID
- `visitor_team_id` - balldontlie team ID
- `season` - Season year

### Rolling Record (2)
- `home_win_pct_L10` - Home team's last 10 games win %
- `away_win_pct_L10` - Away team's last 10 games win %

### Rolling Performance (2)
- `home_pt_diff_L10` - Home team's avg point differential (last 10)
- `away_pt_diff_L10` - Away team's avg point differential (last 10)

### Rest (4)
- `home_rest_days` - Days since home team's last game
- `away_rest_days` - Days since away team's last game
- `home_back_to_back` - 1 if home team played yesterday, 0 otherwise
- `away_back_to_back` - 1 if away team played yesterday, 0 otherwise

### Streak (2)
- `home_win_streak` - Home team's current win/loss streak (positive=wins, negative=losses)
- `away_win_streak` - Away team's current win/loss streak

### Head-to-Head (2)
- `home_h2h_win_pct` - Home team's win % in last 5 meetings
- `home_h2h_pt_diff` - Home team's avg point diff in last 5 meetings

**What's Missing from v1.1?**
- Shooting stats (FG%, FT%, 3P%) - Not available from `/games` endpoint
- Simpler and faster to compute

---

## Troubleshooting

### "API key not found"

Ensure you've set the API key:
```bash
# Check if .env file exists
cat .env

# Should show:
# BALLDONTLIE_API_KEY=3349c5cc-9592-4cfe-8fa8-5b5fa28a36e3

# Or export manually:
export BALLDONTLIE_API_KEY=3349c5cc-9592-4cfe-8fa8-5b5fa28a36e3
```

### "Database not found"

Initialize the database:
```bash
uv run python scripts/init_db.py
```

### "No games found in database"

Fetch data:
```bash
uv run python scripts/fetch_games.py --season 2024
```

### "Model not found: models/model_v2_0.joblib"

Train the model:
```bash
uv run python -m nba_predictor.train_v2_0
```

### Rate limiting errors

balldontlie.io free tier: 5 requests/min. The client automatically handles this with delays between requests. If you hit rate limits:
1. Wait a minute and retry
2. Consider upgrading to paid tier ($10/mo for 60 req/min)

---

## Next Steps (Optional Enhancements)

These are **nice-to-have** features for later:

- **Task 8:** Historical data migration from `games.csv`
- **Task 9:** Add tests for database and feature computer
- **Task 10:** Enhanced API response with data freshness metadata
- **Task 11:** Documentation (DATABASE_SETUP.md, DATA_INGESTION.md)

---

## Comparison: v1.1 vs v2.0

| Feature | v1.1 | v2.0 |
|---------|------|------|
| Data Source | Static CSV | balldontlie.io API |
| Storage | Parquet files | SQLite database |
| Features | 21 (with shooting stats) | 15 (no shooting stats) |
| Feature Computation | Pre-computed | On-demand |
| Team IDs | NBA official IDs | balldontlie IDs |
| Data Updates | Manual re-run scripts | CLI fetch tool |
| Prediction Latency | <10ms | ~60-100ms |

---

## Files Created/Modified in v2.0

### New Files
- `src/nba_predictor/db/__init__.py`
- `src/nba_predictor/db/schema.py` - Database schema
- `src/nba_predictor/db/connection.py` - DB utilities
- `src/nba_predictor/db/feature_computer.py` - On-demand features
- `src/nba_predictor/data_fetcher/__init__.py`
- `src/nba_predictor/data_fetcher/config.py` - API config
- `src/nba_predictor/data_fetcher/balldontlie_client.py` - API client
- `src/nba_predictor/train_v2_0.py` - v2.0 training script
- `scripts/init_db.py` - Database initialization
- `scripts/fetch_games.py` - Data ingestion
- `scripts/setup_v2_0.py` - Automated setup
- `.env.example` - API key template

### Modified Files
- `src/nba_predictor/config.py` - Added v2.0 config
- `src/nba_predictor/api/predictor.py` - Dual-mode support
- `pyproject.toml` - Added dependencies (python-dotenv, requests)

---

## Support

For questions or issues:
1. Check this guide first
2. Review plan file: `~/.claude/plans/memoized-wibbling-peacock.md`
3. Check API docs: https://docs.balldontlie.io

**Get your free API key:** https://www.balldontlie.io
