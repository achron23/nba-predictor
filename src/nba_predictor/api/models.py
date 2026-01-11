"""Pydantic models for API request and response validation."""

from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    """Request model for game prediction endpoint.

    Attributes:
        home_team_id: NBA team ID for the home team (e.g., 1610612737 for Hawks)
        away_team_id: NBA team ID for the away team
        season: Optional season year (e.g., 2023 for 2023-24 season)
        game_date: Optional game date in ISO format (YYYY-MM-DD)
    """
    home_team_id: int = Field(
        ...,
        description="Home team ID",
        example=1610612737
    )
    away_team_id: int = Field(
        ...,
        description="Away team ID",
        example=1610612738
    )
    season: Optional[int] = Field(
        None,
        description="Season year (e.g., 2023 for 2023-24 season)",
        example=2023
    )
    game_date: Optional[str] = Field(
        None,
        description="Game date in ISO format (YYYY-MM-DD)",
        example="2024-01-15"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "home_team_id": 1610612737,
                "away_team_id": 1610612738,
                "season": 2023,
                "game_date": "2024-01-15"
            }
        }


class PredictionResponse(BaseModel):
    """Response model for game prediction endpoint.

    Attributes:
        home_team_id: Home team ID from request
        away_team_id: Away team ID from request
        p_home_win: Predicted probability that home team wins (0.0 to 1.0)
        model_version: Model version used for prediction
        features_used: Number of features used in prediction
    """
    home_team_id: int = Field(..., description="Home team ID")
    away_team_id: int = Field(..., description="Away team ID")
    p_home_win: float = Field(
        ...,
        description="Probability of home team winning",
        ge=0.0,
        le=1.0
    )
    model_version: str = Field(..., description="Model version identifier")
    features_used: int = Field(..., description="Number of features used")

    class Config:
        json_schema_extra = {
            "example": {
                "home_team_id": 1610612737,
                "away_team_id": 1610612738,
                "p_home_win": 0.567,
                "model_version": "v1.1",
                "features_used": 25
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint.

    Attributes:
        status: Health status ("healthy" or "unhealthy")
        model_loaded: Whether the prediction model is loaded
        model_version: Model version identifier
    """
    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Model loaded status")
    model_version: str = Field(..., description="Model version identifier")
