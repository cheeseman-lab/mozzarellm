"""
Global constants used across the project.
"""

# Random seed for reproducibility
SEED = 42

# Confidence score thresholds
SCORE_THRESHOLDS = {"high": 0.86, "medium": 0.81, "low": 0.0}

# API rate limits (requests per minute)
RATE_LIMITS = {"openai": 60, "anthropic": 50, "perplexity": 60, "gemini": 60}

# Token count estimates
CHARS_PER_TOKEN = {"openai": 4, "anthropic": 4.5, "perplexity": 4, "gemini": 4}

# Max gene set sizes
MAX_GENES_PER_ANALYSIS = 1000

# Default column names
DEFAULT_COLUMN_PREFIX = "model_default"
SCORE_COLUMN_SUFFIX = "Score"
NAME_COLUMN_SUFFIX = "Name"
ANALYSIS_COLUMN_SUFFIX = "Analysis"
BINS_COLUMN_SUFFIX = "Score bins"

# Score bin labels
SCORE_BIN_LABELS = [
    "Name not assigned",
    "Low Confidence",
    "Medium Confidence",
    "High Confidence",
]

# Score bin ranges
SCORE_BIN_RANGES = [
    -float("inf"),  # Start with negative infinity
    0,  # No score (name not assigned)
    0.81,  # Low confidence threshold
    0.86,  # Medium confidence threshold
    float("inf"),  # End with positive infinity
]
