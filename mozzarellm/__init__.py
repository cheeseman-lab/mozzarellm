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

# Prompt components (modular)
from .prompts import (
    CLUSTER_ANALYSIS_TASK,
    CONCISE_COT_INSTRUCTIONS,
    DEFAULT_SCREEN_CONTEXT,
    ENHANCED_COT_INSTRUCTIONS,
    GENE_CLASSIFICATION_RULES,
    OUTPUT_FORMAT_JSON,
    PATHWAY_CONFIDENCE_CRITERIA,
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
    # Prompt components
    "CLUSTER_ANALYSIS_TASK",
    "GENE_CLASSIFICATION_RULES",
    "DEFAULT_SCREEN_CONTEXT",
    "PATHWAY_CONFIDENCE_CRITERIA",
    "OUTPUT_FORMAT_JSON",
    "ENHANCED_COT_INSTRUCTIONS",
    "CONCISE_COT_INSTRUCTIONS",
    # Utilities
    "reshape_to_clusters",
]
