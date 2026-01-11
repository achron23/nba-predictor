"""Test script for NBA Predictor API.

Tests the API endpoints programmatically without needing a running server.
"""

import sys
sys.path.insert(0, 'src')

from fastapi.testclient import TestClient
from nba_predictor.api.app import app

# Create test client with startup events enabled
client = TestClient(app)

# Trigger startup event manually
with client:
    pass  # Context manager triggers startup/shutdown

def test_root():
    """Test root endpoint."""
    print("Testing GET /...")
    response = client.get("/")
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")
    print()

def test_health():
    """Test health check endpoint."""
    print("Testing GET /health...")
    response = client.get("/health")
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")
    print()

def test_prediction():
    """Test prediction endpoint."""
    print("Testing POST /predict...")

    # Example: Lakers (1610612747) at Warriors (1610612744)
    payload = {
        "home_team_id": 1610612744,
        "away_team_id": 1610612747,
        "season": 2023
    }

    response = client.post("/predict", json=payload)
    print(f"  Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"  Home Team ID: {result['home_team_id']}")
        print(f"  Away Team ID: {result['away_team_id']}")
        print(f"  P(Home Win): {result['p_home_win']:.3f}")
        print(f"  Model Version: {result['model_version']}")
        print(f"  Features Used: {result['features_used']}")
    else:
        print(f"  Error: {response.json()}")
    print()

def test_invalid_prediction():
    """Test prediction with invalid input (same team)."""
    print("Testing POST /predict with invalid input...")

    payload = {
        "home_team_id": 1610612744,
        "away_team_id": 1610612744,  # Same as home
        "season": 2023
    }

    response = client.post("/predict", json=payload)
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("NBA Predictor API Tests")
    print("=" * 60)
    print()

    test_root()
    test_health()
    test_prediction()
    test_invalid_prediction()

    print("=" * 60)
    print("Tests completed!")
    print("=" * 60)
