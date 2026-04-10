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
    CLUSTER_ANALYSIS_TASK_MULTI,
    COT_STEPS_DEFAULT,
    GENE_CATEGORIZATION_RULES,
    OUTPUT_FORMAT_JSON,
    PATHWAY_CONFIDENCE_CRITERIA,
    assemble_cot_instructions,
)
from .clients.llm_api_clients import (
    AnthropicClient,
    GeminiClient,
    LLMClientBase,
    OpenAIClient,
    create_client,
)

# Literature validation
from .pipeline.literature_mcp import (
    validate_and_amend_with_mcp,
    validate_and_amend_without_mcp,
)
from .schemas.mcp_schemas import ClusterValidationResult
from .utils.cluster_utils import aggregate_genes_by_cluster

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
    "CLUSTER_ANALYSIS_TASK_MULTI",
    "GENE_CATEGORIZATION_RULES",
    "PATHWAY_CONFIDENCE_CRITERIA",
    "OUTPUT_FORMAT_JSON",
    "COT_STEPS_DEFAULT",
    "assemble_cot_instructions",
    # Literature validation
    "validate_and_amend_with_mcp",
    "validate_and_amend_without_mcp",
    "ClusterValidationResult",
    "aggregate_genes_by_cluster",
    # IO utils
    "load_table",
    "write_bundle",
]
