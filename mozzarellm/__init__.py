"""
mozzarellm: Gene cluster analysis using Large Language Models (LLMs)
"""

__version__ = "0.1.0"

# Import key functions to make them available at package level
from .utils.cluster_analyzer import analyze_gene_clusters, process_clusters
from .utils.cluster_utils import reshape_to_clusters
from .utils.prompt_factory import (
    make_cluster_analysis_prompt,
    make_batch_cluster_analysis_prompt,
)
from .utils.llm_analysis_utils import process_cluster_response, save_cluster_analysis

# Expose package-level API
__all__ = [
    "analyze_gene_clusters",
    "process_clusters",
    "reshape_to_clusters",
    "make_cluster_analysis_prompt",
    "make_batch_cluster_analysis_prompt",
    "process_cluster_response",
    "save_cluster_analysis",
]
