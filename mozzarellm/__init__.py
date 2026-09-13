"""
mozzarellm: Gene cluster analysis using Large Language Models (LLMs)
"""

__version__ = "0.2.0"

from .clients.llm_api_clients import (
    AnthropicClient,
    GeminiClient,
    LLMClientBase,
    OpenAIClient,
    create_client,
)

# Prompts: the texts and their assembly
from .prompts import COMPONENTS, DEFAULT_ORDERS, assemble, default_order
from .schemas.mcp_schemas import (
    LiteraturePathwayRevision,
    LiteratureReclassification,
    LiteratureValidation,
    RelevantPaper,
)

# IO utils
from .utils.io import load_table, write_bundle

# Expose package-level API
__all__ = [
    # Providers
    "LLMClientBase",
    "OpenAIClient",
    "AnthropicClient",
    "GeminiClient",
    "create_client",
    # Prompts
    "COMPONENTS",
    "DEFAULT_ORDERS",
    "default_order",
    "assemble",
    # Literature validation schemas
    "RelevantPaper",
    "LiteratureValidation",
    "LiteratureReclassification",
    "LiteraturePathwayRevision",
    # IO utils
    "load_table",
    "write_bundle",
]
