"""
Prompt construction utilities for gene cluster analysis.

This module assembles prompts by concatenating modular components from prompts.py
in a standardized order. Components are organized in prompts.py to match assembly order.

Rules of thumb:
stable policy + format requirements = system prompt
cluster specific content = user prompt

Main components of any given system prompt:
  1. TASK SPECIFICATION (may be composed of multiple sub-tasks/prompt components)
  2. CONTEXT (e.g. from screen_context.json, universal context like gene annotations)
  3. OUTPUT FORMAT (response structure)

Main components of any given user prompt:
  1. CLUSTER SPECIFIC CONTENT (e.g. gene list + metadata, retrieved context)
"""

import os
import json
from pathlib import Path
from mozzarellm.utils.screen_context_utils import load_screen_context_json
from mozzarellm.prompt_components import (
    CLUSTER_ANALYSIS_TASK,
    GENE_CLASSIFICATION_RULES,
    PATHWAY_CONFIDENCE_CRITERIA,
    OUTPUT_FORMAT_JSON,
)


def make_cluster_analysis_system_prompt(
    *,
    screen_context_path: Path | None = None,
    override_screen_context: bool = False,  # testing utility
    template_path: Path | None = None,
    template_string: str | None = None,
):
    """
    Creates a system prompt for gene cluster analysis by assembling modular components.

    Assembly order:
    1. CLUSTER_ANALYSIS_TASK (discovery mission)
    2. SCREEN CONTEXT (from screen_context.json)
    3. GENE_CLASSIFICATION_RULES (framework for analysis)
    4. PATHWAY_CONFIDENCE_CRITERIA (assessment criteria)
    5. OUTPUT_FORMAT_JSON (response structure)

    Args:
        cluster_id: Identifier for the cluster
        genes: List of gene identifiers in the cluster
        gene_annotations_dict: Optional dict of gene functional annotations
        screen_context: Optional benchmark-specific experimental context
        template_path: Path to custom template file (escape hatch - full control)
        template_string: Custom template string (escape hatch - full control)
        retrieved_context: Optional dict with RAG evidence snippets
        cot_instructions: Optional chain-of-thought instructions string

    Returns:
        prompt: Fully assembled prompt string
    """
    try:
        screen_ctx_obj = load_screen_context_json(
            screen_context_path, override=override_screen_context
        )
    except Exception as e:
        raise ValueError(f"Failed to load screen context: {e}")

    SCREEN_CONTEXT_TEXT = json.dumps(
        screen_ctx_obj, ensure_ascii=False
    )  # minify JSON; has no effect on readability for LLMs + saves tokens

    # =========================================================================
    # ESCAPE HATCH: Custom template overrides everything
    # =========================================================================
    if template_string or template_path:
        prompt = _load_custom_template(template_path, template_string)
        return (
            prompt
            + "\n\n The following experimental context is provided: "
            + SCREEN_CONTEXT_TEXT
            + "\n\n"
        )

    # =========================================================================
    # DEFAULT PROMPT CONSTRUCTION
    # =========================================================================

    prompt = (
        CLUSTER_ANALYSIS_TASK
        + "\n\n"
        + SCREEN_CONTEXT_TEXT
        + "\n\n"
        + GENE_CLASSIFICATION_RULES
        + "\n\n"
        + PATHWAY_CONFIDENCE_CRITERIA
        + "\n\n"
        + OUTPUT_FORMAT_JSON
    )

    return prompt


def make_single_cluster_analysis_user_prompt(cluster_id, screen_name, cluster_to_bundle_path_map):
    BUNDLE_PATH = cluster_to_bundle_path_map[str(cluster_id)]

    # Build a user prompt from the bundle JSON
    bundle_obj = json.loads(Path(BUNDLE_PATH).read_text(encoding="utf-8"))
    bundle_text = json.dumps(bundle_obj, ensure_ascii=False)
    return (
        f"Here is the evidence bundle JSON for cluster {cluster_id}:\n\n```json\n{bundle_text}\n```"
    )


# =============================================================================
# Helper functions for custom templates and optional components
# =============================================================================


def _load_custom_template(template_path=None, template_string=None):
    """Load a custom template from file or string."""
    if template_string:
        print("Using provided custom template string")
        return template_string

    if template_path:
        try:
            p = Path(template_path)
        except Exception as e:
            raise ValueError(f"Invalid custom template path: {template_path!r}. Error: {e}")

        if not p.exists():
            raise ValueError(f"Custom template path does not exist: {str(p)!r}")

        allowed_suffixes = {".txt", ".md"}
        if p.suffix.lower() not in allowed_suffixes:
            raise ValueError(
                f"Custom template file must be one of: {allowed_suffixes}. Got: {str(p)!r}"
            )

        print(f"Loading custom template from: {p}")
        try:
            return p.read_text(encoding="utf-8")
        except Exception as e:
            raise ValueError(f"Failed to read custom template file: {str(p)!r}. Error: {e}")

    raise ValueError("Custom template path not provided and template string not provided")


def _format_retrieved_evidence(retrieved_context):
    """Format retrieved evidence snippets for RAG mode."""
    snippets = retrieved_context.get("snippets", [])
    if not snippets:
        return ""

    evidence_text = (
        "\nRETRIEVED EVIDENCE (ranked by relevance; cite by [number] in your reasoning):\n"
    )

    for i, sn in enumerate(snippets, 1):
        txt = sn.get("text", "").strip()
        src = sn.get("source", "")
        meta = sn.get("meta", {})
        relevance = sn.get("relevance_score", 0)

        # Format source tag based on source type
        if src == "knowledge_file":
            src_tag = f"File:{meta.get('path', '')} (score:{meta.get('score', 0)})"
        else:
            src_tag = str(src or "Unknown")

        evidence_text += f"[{i}] {src_tag} [relevance:{relevance}]: {txt}\n"

    # Add retrieval metadata summary
    ret_meta = retrieved_context.get("retrieval_metadata", {})
    if ret_meta:
        evidence_text += (
            f"\nRetrieval Summary: {ret_meta.get('annotations_found', 0)} gene annotations, "
        )
        evidence_text += f"{ret_meta.get('knowledge_snippets_found', 0)} knowledge snippets from {ret_meta.get('total_retrieved', 0)} total sources.\n"

        # Note genes without annotations
        genes_no_annot = ret_meta.get("genes_without_annotations", [])
        if genes_no_annot:
            evidence_text += f"\nNOTE: The following genes lack direct functional annotations in the provided data: {', '.join(genes_no_annot)}\n"
            evidence_text += "These genes should be carefully evaluated based on pathway context and knowledge base evidence.\n"

    return evidence_text
