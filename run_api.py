"""Convenience script to run the NBA Predictor API.

Run with: uv run python run_api.py

For development with auto-reload, use:
  uv run uvicorn nba_predictor.api.app:app --reload --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path

# Add src directory to Python path for direct execution
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import uvicorn

if __name__ == "__main__":
    # Run without reload for Windows compatibility
    # Use uvicorn CLI directly for reload functionality
    uvicorn.run(
        "nba_predictor.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
