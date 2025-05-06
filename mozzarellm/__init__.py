"""
mozzarellm: Gene cluster analysis using Large Language Models (LLMs)
"""

__version__ = "0.2.0"

# Import configuration and prompt constants
from .configs import (
    DEFAULT_CONFIG,
    DEFAULT_OPENAI_CONFIG,
    DEFAULT_ANTHROPIC_CONFIG,
    DEFAULT_GEMINI_CONFIG,
)

from .prompts import (
    DEFAULT_CLUSTER_PROMPT,
    DEFAULT_BATCH_PROMPT,
)

# Import key functions
from .utils.cluster_analyzer import (
    analyze_gene_clusters,
    process_clusters,
    load_gene_annotations,
    load_screen_context,
)
from .utils.cluster_utils import reshape_to_clusters
from .utils.prompt_factory import (
    make_cluster_analysis_prompt,
    make_batch_cluster_analysis_prompt,
)
from .utils.llm_analysis_utils import process_cluster_response, save_cluster_analysis

# Expose package-level API
__all__ = [
    # Core functions
    "analyze_gene_clusters",
    "process_clusters",
    "reshape_to_clusters",
    # Configuration constants
    "DEFAULT_CONFIG",
    "DEFAULT_OPENAI_CONFIG",
    "DEFAULT_ANTHROPIC_CONFIG",
    "DEFAULT_GEMINI_CONFIG",
    # Prompt constants
    "DEFAULT_CLUSTER_PROMPT",
    "DEFAULT_BATCH_PROMPT",
    "CLUSTER_OUTPUT_FORMAT",
    "HELA_SCREEN_INFO",
    # Utility functions
    "make_cluster_analysis_prompt",
    "make_batch_cluster_analysis_prompt",
    "process_cluster_response",
    "save_cluster_analysis",
    "load_gene_annotations",
    "load_screen_context",
]
