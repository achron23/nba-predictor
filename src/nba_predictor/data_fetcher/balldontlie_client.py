"""balldontlie.io API client for fetching NBA game data.

This client handles:
- Authentication with API key
- Pagination (automatic cursor following)
- Rate limiting (respects free tier 5 req/min)
- Retry logic with exponential backoff
- Response parsing to database schema format
"""

import os
import time
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

from .config import (
    BALLDONTLIE_API_BASE_URL,
    BALLDONTLIE_API_TIMEOUT,
    BALLDONTLIE_API_MAX_RETRIES,
    BALLDONTLIE_FREE_TIER_RATE_LIMIT,
    MAX_PER_PAGE,
    RETRY_BACKOFF_FACTOR,
    RETRY_INITIAL_DELAY,
)

logger = logging.getLogger(__name__)


class BallDontLieClient:
    """Client for interacting with balldontlie.io API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the API client.

        Args:
            api_key: balldontlie.io API key. If None, reads from BALLDONTLIE_API_KEY env var.

        Raises:
            ValueError: If API key is not provided
        """
        self.api_key = api_key or os.environ.get("BALLDONTLIE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set BALLDONTLIE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.base_url = BALLDONTLIE_API_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        })

        # Rate limiting (for free tier: 5 req/min = 12 seconds between requests)
        self.min_request_interval = 60.0 / BALLDONTLIE_FREE_TIER_RATE_LIMIT
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting by sleeping if necessary."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None
    ) -> Dict:
        """Make an API request with retry logic.

        Args:
            endpoint: API endpoint path (e.g., "/games")
            params: Query parameters

        Returns:
            Response JSON as dictionary

        Raises:
            requests.RequestException: If request fails after retries
        """
        url = f"{self.base_url}{endpoint}"
        params = params or {}

        for attempt in range(BALLDONTLIE_API_MAX_RETRIES):
            try:
                # Rate limiting
                self._rate_limit()

                # Make request
                logger.debug(f"Request: {url} with params: {params}")
                response = self.session.get(
                    url,
                    params=params,
                    timeout=BALLDONTLIE_API_TIMEOUT
                )
                response.raise_for_status()

                return response.json()

            except requests.Timeout as e:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{BALLDONTLIE_API_MAX_RETRIES})")
                if attempt == BALLDONTLIE_API_MAX_RETRIES - 1:
                    raise

                # Exponential backoff
                delay = RETRY_INITIAL_DELAY * (RETRY_BACKOFF_FACTOR ** attempt)
                time.sleep(delay)

            except requests.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit exceeded
                    logger.warning(f"Rate limit exceeded, retrying after delay")
                    time.sleep(RETRY_INITIAL_DELAY * (RETRY_BACKOFF_FACTOR ** attempt))
                elif e.response.status_code >= 500:  # Server error, retry
                    logger.warning(f"Server error {e.response.status_code}, retrying")
                    time.sleep(RETRY_INITIAL_DELAY * (RETRY_BACKOFF_FACTOR ** attempt))
                else:
                    # Client error, don't retry
                    logger.error(f"HTTP error: {e}")
                    raise

            except requests.RequestException as e:
                logger.error(f"Request failed: {e}")
                if attempt == BALLDONTLIE_API_MAX_RETRIES - 1:
                    raise
                time.sleep(RETRY_INITIAL_DELAY * (RETRY_BACKOFF_FACTOR ** attempt))

    def fetch_games_by_date_range(
        self,
        start_date: str,
        end_date: str,
        season: Optional[int] = None
    ) -> List[Dict]:
        """Fetch all games within a date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            season: Optional season filter

        Returns:
            List of game dictionaries

        Raises:
            requests.RequestException: If API request fails
        """
        all_games = []
        cursor = None

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "per_page": MAX_PER_PAGE,
        }

        if season:
            params["seasons[]"] = season

        logger.info(f"Fetching games from {start_date} to {end_date}")

        while True:
            if cursor:
                params["cursor"] = cursor

            response = self._make_request("/games", params)

            # Check if response is valid
            if response is None:
                logger.error("API returned None response")
                break

            # Extract games
            games = response.get("data", [])
            all_games.extend(games)

            logger.info(f"Fetched {len(games)} games (total: {len(all_games)})")

            # Check for next page
            meta = response.get("meta", {})
            cursor = meta.get("next_cursor")

            if not cursor:
                break

        logger.info(f"Completed: fetched {len(all_games)} total games")
        return all_games

    def fetch_games_for_date(self, date: str) -> List[Dict]:
        """Fetch all games for a specific date.

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            List of game dictionaries
        """
        return self.fetch_games_by_date_range(date, date)

    def fetch_games_for_season(self, season: int) -> List[Dict]:
        """Fetch all games for a season.

        Args:
            season: Season year (e.g., 2024 for 2024-25 season)

        Returns:
            List of game dictionaries
        """
        # Season spans from October to June
        # Use a wide date range to capture all games
        start_date = f"{season}-10-01"
        end_date = f"{season + 1}-06-30"

        return self.fetch_games_by_date_range(start_date, end_date, season=season)

    def parse_game_to_db_format(self, game: Dict) -> Dict:
        """Transform balldontlie.io game response to database schema format.

        Args:
            game: Game dictionary from API response

        Returns:
            Dictionary matching database schema
        """
        # Extract team IDs directly (balldontlie team IDs are canonical in v2.0)
        home_team = game.get("home_team", {})
        visitor_team = game.get("visitor_team", {})

        home_team_id = home_team.get("id")
        visitor_team_id = visitor_team.get("id")

        # Extract scores
        home_score = game.get("home_team_score")
        visitor_score = game.get("visitor_team_score")

        # Compute home_win label
        home_win = None
        if home_score is not None and visitor_score is not None:
            home_win = 1 if home_score > visitor_score else 0

        # Extract date (convert datetime to date string)
        game_date = game.get("date", "")[:10]  # Extract YYYY-MM-DD

        # Extract season
        season = game.get("season")

        # Extract status
        status = game.get("status", "")

        # Extract postseason flag
        postseason = 1 if game.get("postseason") else 0

        return {
            "game_id": game.get("id"),
            "game_date": game_date,
            "season": season,
            "home_team_id": home_team_id,
            "visitor_team_id": visitor_team_id,
            "home_team_score": home_score,
            "visitor_team_score": visitor_score,
            "home_win": home_win,
            "postseason": postseason,
            "status": status,
        }
