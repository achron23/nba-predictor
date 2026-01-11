"""FastAPI application for NBA game outcome predictions.

This module implements the main API endpoints for the NBA Predictor service.
Run with: uvicorn nba_predictor.api.app:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import PredictionRequest, PredictionResponse, HealthResponse
from .predictor import NBAPredictor
from .. import config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NBA Predictor API",
    description="Predict NBA game outcomes using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance (loaded on startup)
_predictor: NBAPredictor = None
_model_version = "v1.1"


@app.on_event("startup")
async def load_model():
    """Load the trained model on API startup."""
    global _predictor

    logger.info("Loading predictor on startup...")

    try:
        _predictor = NBAPredictor()
        logger.info(f"Predictor loaded successfully")
        logger.info(f"Model version: {_predictor.model_version}")
        logger.info(f"Features count: {len(_predictor.feature_cols)}")
        logger.info(f"Teams in cache: {len(_predictor.team_stats)}")

    except Exception as e:
        logger.error(f"Failed to load predictor: {e}")
        logger.warning("API will start but predictions will fail")
        _predictor = None


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "NBA Predictor API",
        "version": "1.0.0",
        "description": "Predict NBA game outcomes using machine learning",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint to verify service status and model availability.

    Returns:
        HealthResponse with service status and model information
    """
    model_loaded = _predictor is not None
    status = "healthy" if model_loaded else "unhealthy"

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        model_version=_model_version
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """Predict the probability of the home team winning.

    Args:
        request: PredictionRequest with home_team_id and away_team_id

    Returns:
        PredictionResponse with win probability and metadata

    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Predictor not loaded. Please check server logs."
        )

    logger.info(
        f"Prediction requested: Home={request.home_team_id}, "
        f"Away={request.away_team_id}, Season={request.season}"
    )

    try:
        # Make prediction
        result = _predictor.predict(
            home_team_id=request.home_team_id,
            away_team_id=request.away_team_id,
            season=request.season,
            game_date=request.game_date
        )

        logger.info(f"Prediction result: p_home_win={result['p_home_win']:.3f}")

        return PredictionResponse(**result)

    except ValueError as e:
        # Invalid input
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Server error
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
