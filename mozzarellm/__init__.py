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
)

# Prompt components (modular)
from .prompts import (
    CLUSTER_ANALYSIS_TASK,
    DEFAULT_SCREEN_CONTEXT,
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
    # Utilities
    "reshape_to_clusters",
]
