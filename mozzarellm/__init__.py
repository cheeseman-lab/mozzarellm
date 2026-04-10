"""
mozzarellm: Gene cluster analysis using Large Language Models (LLMs)
"""

__version__ = "0.2.0"

# New unified API
# Literature validation
from .clients.pmc_client import LiteratureClient
from .clients.llm_api_clients import (
    AnthropicClient,
    GeminiClient,
    LLMClientBase,
    OpenAIClient,
    create_client,
)
from .pipeline.analyzer import ClusterAnalyzer
from .pipeline.literature_mcp import (
    validate_and_amend_with_mcp,
    validate_and_amend_without_mcp,
)

# Prompt components (modular)
from .prompt_components import (
    CLUSTER_ANALYSIS_TASK,
    CLUSTER_ANALYSIS_TASK_MULTI,
    COT_STEPS_DEFAULT,
    DIRECT_MCP_VALIDATION_PROMPT,
    GENE_CATEGORIZATION_RULES,
    LITERATURE_VALIDATION_OUTPUT_FORMAT,
    OUTPUT_FORMAT_JSON,
    PATHWAY_CONFIDENCE_CRITERIA,
    STRUCTURED_MCP_REFINEMENT_PROMPT,
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
from .utils.cluster_utils import aggregate_genes_by_cluster

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
    # Literature validation prompts
    "LITERATURE_VALIDATION_OUTPUT_FORMAT",
    "DIRECT_MCP_VALIDATION_PROMPT",
    "STRUCTURED_MCP_REFINEMENT_PROMPT",
    # Literature validation
    "LiteratureClient",
    "validate_and_amend_with_mcp",
    "validate_and_amend_without_mcp",
    "ClusterValidationResult",
    # IO utils
    "load_table",
    "write_bundle",
    "aggregate_genes_by_cluster",
]
