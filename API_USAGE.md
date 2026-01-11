# NBA Predictor API Usage Guide

## Starting the API

### Option 1: Using the run script
```bash
uv run python run_api.py
```

### Option 2: Using uvicorn directly (with auto-reload)
```bash
uv run uvicorn nba_predictor.api.app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

## API Documentation

Once the API is running, you can access:
- **Interactive API docs (Swagger)**: http://localhost:8000/docs
- **Alternative docs (ReDoc)**: http://localhost:8000/redoc

## Endpoints

### 1. Root - `GET /`
Get API information.

**Example:**
```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "name": "NBA Predictor API",
  "version": "1.0.0",
  "description": "Predict NBA game outcomes using machine learning",
  "endpoints": {
    "health": "/health",
    "predict": "/predict",
    "docs": "/docs"
  }
}
```

### 2. Health Check - `GET /health`
Check API and model status.

**Example:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "v1.1"
}
```

### 3. Predict - `POST /predict`
Predict the probability of home team winning.

**Request Body:**
```json
{
  "home_team_id": 1610612744,  // Required: Home team ID
  "away_team_id": 1610612747,  // Required: Away team ID
  "season": 2023,              // Optional: Season year
  "game_date": "2024-01-15"    // Optional: Game date (ISO format)
}
```

**Example using curl:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "home_team_id": 1610612744,
    "away_team_id": 1610612747,
    "season": 2023
  }'
```

**Example using Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "home_team_id": 1610612744,  # Warriors
        "away_team_id": 1610612747,  # Lakers
        "season": 2023
    }
)

result = response.json()
print(f"P(Home Win): {result['p_home_win']:.1%}")
```

**Response:**
```json
{
  "home_team_id": 1610612744,
  "away_team_id": 1610612747,
  "p_home_win": 0.702,
  "model_version": "v1.1",
  "features_used": 21
}
```

## Common NBA Team IDs

| Team | ID |
|------|------------|
| Atlanta Hawks | 1610612737 |
| Boston Celtics | 1610612738 |
| Brooklyn Nets | 1610612751 |
| Charlotte Hornets | 1610612766 |
| Chicago Bulls | 1610612741 |
| Cleveland Cavaliers | 1610612739 |
| Dallas Mavericks | 1610612742 |
| Denver Nuggets | 1610612743 |
| Detroit Pistons | 1610612765 |
| Golden State Warriors | 1610612744 |
| Houston Rockets | 1610612745 |
| Indiana Pacers | 1610612754 |
| LA Clippers | 1610612746 |
| Los Angeles Lakers | 1610612747 |
| Memphis Grizzlies | 1610612763 |
| Miami Heat | 1610612748 |
| Milwaukee Bucks | 1610612749 |
| Minnesota Timberwolves | 1610612750 |
| New Orleans Pelicans | 1610612740 |
| New York Knicks | 1610612752 |
| Oklahoma City Thunder | 1610612760 |
| Orlando Magic | 1610612753 |
| Philadelphia 76ers | 1610612755 |
| Phoenix Suns | 1610612756 |
| Portland Trail Blazers | 1610612757 |
| Sacramento Kings | 1610612758 |
| San Antonio Spurs | 1610612759 |
| Toronto Raptors | 1610612761 |
| Utah Jazz | 1610612762 |
| Washington Wizards | 1610612764 |

## Error Responses

### 400 Bad Request
Invalid input (e.g., same team for home and away).
```json
{
  "detail": "Home and away teams must be different"
}
```

### 503 Service Unavailable
Model not loaded.
```json
{
  "detail": "Predictor not loaded. Please check server logs."
}
```

### 500 Internal Server Error
Prediction failed due to server error.
```json
{
  "detail": "Prediction failed: [error message]"
}
```

## Testing

Run the test script to verify all endpoints:
```bash
uv run python test_api.py
```

## Notes

- The API uses the v1.1 model trained with engineered features
- For v2, team statistics are cached from recent historical data
- In production, this would be enhanced with live data integration
- All probabilities are between 0.0 (0% chance) and 1.0 (100% chance)
