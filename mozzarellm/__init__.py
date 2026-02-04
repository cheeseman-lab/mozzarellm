"""
mozzarellm: Gene cluster analysis using Large Language Models (LLMs)
"""

__version__ = "0.2.0"

# New unified API
from .pipeline.analyzer import ClusterAnalyzer
from .schemas.analysis_output_schemas import (
    AnalysisResult,
    ClusterInput,
    ClusterResult,
    GeneClassification,
    RetrievalContext,
)

# Prompt components (modular)
from .prompt_components import (
    CLUSTER_ANALYSIS_TASK,
    CONCISE_COT_INSTRUCTIONS,
    ENHANCED_COT_INSTRUCTIONS,
    GENE_CLASSIFICATION_RULES,
    OUTPUT_FORMAT_JSON,
    PATHWAY_CONFIDENCE_CRITERIA,
)
from .clients.llm_api_clients import (
    AnthropicClient,
    GeminiClient,
    LLMClientBase,
    OpenAIClient,
    create_client,
)

# IO utils
from .utils.io import load_table, write_bundle

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
    "LLMClientBase",
    "OpenAIClient",
    "AnthropicClient",
    "GeminiClient",
    "create_client",
    # Prompt components
    "CLUSTER_ANALYSIS_TASK",
    "GENE_CLASSIFICATION_RULES",
    "DEFAULT_SCREEN_CONTEXT",
    "PATHWAY_CONFIDENCE_CRITERIA",
    "OUTPUT_FORMAT_JSON",
    "ENHANCED_COT_INSTRUCTIONS",
    "CONCISE_COT_INSTRUCTIONS",
    # IO utils
    "load_table",
    "write_bundle",
]
