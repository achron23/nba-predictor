"""Configuration for balldontlie.io API client."""

# API Configuration
BALLDONTLIE_API_BASE_URL = "https://api.balldontlie.io/v1"
BALLDONTLIE_API_TIMEOUT = 10  # seconds
BALLDONTLIE_API_MAX_RETRIES = 3
BALLDONTLIE_FREE_TIER_RATE_LIMIT = 5  # requests per minute

# Pagination
MAX_PER_PAGE = 100  # balldontlie.io max

# Retry configuration
RETRY_BACKOFF_FACTOR = 2  # Exponential backoff multiplier
RETRY_INITIAL_DELAY = 1  # Initial delay in seconds
