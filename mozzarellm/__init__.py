"""
mozzarellm: Gene cluster analysis using Large Language Models (LLMs)
"""

__version__ = "0.2.0"

# New unified API
from .analyzer import ClusterAnalyzer
from .models import (
    AnalysisResult,
    ClusterInput,
    ClusterResult,
    GeneClassification,
    RetrievalContext,
)

# Prompt constants (preserved)
from .prompts import (
    CONCISE_COT_INSTRUCTIONS,
    ENHANCED_COT_INSTRUCTIONS,
    ROBUST_CLUSTER_PROMPT,
    ROBUST_SCREEN_CONTEXT,
)
from .providers import (
    AnthropicProvider,
    GeminiProvider,
    LLMProvider,
    OpenAIProvider,
    create_provider,
)

# Utility functions (preserved)
from .utils.cluster_utils import reshape_to_clusters

# Expose package-level API
__all__ = [
    # Main API
    "ClusterAnalyzer",
    # Models
    "ClusterResult",
    "AnalysisResult",
    "GeneClassification",
    "ClusterInput",
    "RetrievalContext",
    # Providers
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "create_provider",
    # Prompt constants
    "ENHANCED_COT_INSTRUCTIONS",
    "CONCISE_COT_INSTRUCTIONS",
    "ROBUST_SCREEN_CONTEXT",
    "ROBUST_CLUSTER_PROMPT",
    # Utilities
    "reshape_to_clusters",
]
