"""The prompts: texts (components.py) and how they are assembled (assembly.py)."""

from .assembly import (
    DEFAULT_ORDERS,
    assemble,
    compose_stepwise_user_turns,
    default_order,
    make_cluster_analysis_system_prompt,
    make_single_cluster_analysis_user_prompt,
    render,
    strip_feature_fields,
    strip_source_fields,
)
from .components import COMPONENTS, EMBEDS

__all__ = [
    "COMPONENTS",
    "EMBEDS",
    "DEFAULT_ORDERS",
    "default_order",
    "render",
    "assemble",
    "make_cluster_analysis_system_prompt",
    "compose_stepwise_user_turns",
    "make_single_cluster_analysis_user_prompt",
    "strip_feature_fields",
    "strip_source_fields",
]
