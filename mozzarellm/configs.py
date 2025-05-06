# mozzarellm/configs.py

# Base configuration with shared settings
DEFAULT_CONFIG = {
    "MODEL": "",
    "CONTEXT": "You are an AI assistant specializing in genomics and systems biology with expertise in pathway analysis. Your task is to analyze gene clusters to identify biological pathways and potential novel pathway members based on published literature and gaps in knowledge of gene function.",
    "TEMP": 0.0,
    "MAX_TOKENS": 8000,
    "RATE_PER_TOKEN": 0.00001,
    "DOLLAR_LIMIT": 10.0,
    "LOG_NAME": "cluster_analysis",
}

# OpenAI configuration (standard models)
DEFAULT_OPENAI_CONFIG = DEFAULT_CONFIG.copy()
DEFAULT_OPENAI_CONFIG.update({
    "MODEL": "gpt-4o",
    "RATE_PER_TOKEN": 0.00001,
    "API_TYPE": "openai",
})

# OpenAI creative models config (o4-mini, o3-mini with higher temperature)
DEFAULT_OPENAI_REASONING_CONFIG = DEFAULT_OPENAI_CONFIG.copy()
DEFAULT_OPENAI_REASONING_CONFIG.update({
    "MODEL": "o4-mini",
    "TEMP": 1.0,
})

# Anthropic configuration 
DEFAULT_ANTHROPIC_CONFIG = DEFAULT_CONFIG.copy()
DEFAULT_ANTHROPIC_CONFIG.update({
    "MODEL": "claude-3-7-sonnet-20250219",
    "RATE_PER_TOKEN": 0.000015,
    "API_TYPE": "anthropic",
})

# Google configuration
DEFAULT_GEMINI_CONFIG = DEFAULT_CONFIG.copy()
DEFAULT_GEMINI_CONFIG.update({
    "MODEL": "gemini-2.5-pro-preview-03-25",
    "RATE_PER_TOKEN": 0.000005,
    "API_TYPE": "gemini",
})

# List of reasoning OpenAI models that use higher temperature
REASONING_OPENAI_MODELS = ["o4-mini", "o3-mini"]