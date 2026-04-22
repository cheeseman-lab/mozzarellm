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

# Prompt components (modular)
from .prompt_components import (
    CLUSTER_ANALYSIS_TASK,
    CLUSTER_ANALYSIS_TASK_MULTI,
    COT_STEPS_DEFAULT,
    GENE_CATEGORIZATION_RULES,
    OUTPUT_FORMAT_JSON,
    PATHWAY_CONFIDENCE_CRITERIA,
    assemble_cot_instructions,
)
from .schemas.analysis_output_schemas import (
    AnalysisResult,
    ClusterInput,
    ClusterResult,
    GeneClassification,
    RetrievalContext,
)
from .schemas.mcp_schemas import ClusterValidationResult

# IO utils
from .utils.io import load_table, write_bundle

# Expose package-level API
__all__ = [
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
    "CLUSTER_ANALYSIS_TASK_MULTI",
    "GENE_CATEGORIZATION_RULES",
    "PATHWAY_CONFIDENCE_CRITERIA",
    "OUTPUT_FORMAT_JSON",
    "COT_STEPS_DEFAULT",
    "assemble_cot_instructions",
    # Literature validation
    "ClusterValidationResult",
    # IO utils
    "load_table",
    "write_bundle",
]
