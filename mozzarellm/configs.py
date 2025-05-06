# mozzarellm/configs.py

# Base configuration with shared settings
DEFAULT_CONFIG = {
    "MODEL": "",
    "CONTEXT": "You are an AI assistant specializing in genomics and systems biology with expertise in pathway analysis. Your task is to analyze gene clusters to identify biological pathways and potential novel pathway members based on published literature and gaps in knowledge of gene function.",
    "TEMP": 0.0,
    "MAX_TOKENS": 4000,
    "RATE_PER_TOKEN": 0.00001,
    "DOLLAR_LIMIT": 10.0,
    "LOG_NAME": "cluster_analysis",
}

# OpenAI configuration
DEFAULT_OPENAI_CONFIG = DEFAULT_CONFIG.copy()
DEFAULT_OPENAI_CONFIG.update(
    {
        "MODEL": "gpt-4o",
        "RATE_PER_TOKEN": 0.00001,
        "API_TYPE": "openai",
    }
)

# Anthropic configuration
DEFAULT_ANTHROPIC_CONFIG = DEFAULT_CONFIG.copy()
DEFAULT_ANTHROPIC_CONFIG.update(
    {
        "MODEL": "claude-3-7-sonnet-20250219",
        "RATE_PER_TOKEN": 0.000015,
        "API_TYPE": "anthropic",
    }
)

# Google configuration
DEFAULT_GEMINI_CONFIG = DEFAULT_CONFIG.copy()
DEFAULT_GEMINI_CONFIG.update(
    {
        "MODEL": "gemini-2.0-pro",
        "RATE_PER_TOKEN": 0.000005,
        "API_TYPE": "gemini",
    }
)
