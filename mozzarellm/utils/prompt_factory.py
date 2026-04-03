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
from datetime import datetime
from mozzarellm.utils.screen_context_utils import load_screen_context_json
from mozzarellm.prompt_components import (
    CLUSTER_ANALYSIS_TASK,
    GENE_CATEGORIZATION_RULES,
    PATHWAY_CONFIDENCE_CRITERIA,
    NOVEL_CLASSIFICATION_RULES,
    UNCHARACTERIZED_CLASSIFICATION_RULES,
    OUTPUT_FORMAT_JSON,
    COT_STEPS_DEFAULT,
    assemble_cot_instructions,
)


def make_cluster_analysis_system_prompt(
    *,
    screen_name: str,
    screen_context_path: Path | None = None,
    override_CoT_steps: list[str] | None = None,  # testing utility
    override_screen_context: bool = False,  # testing utility
    template_path: Path | None = None,
    template_string: str | None = None,
    CoT_mode: bool = False,
    output_dir: Path | None = None,
):
    """
    Creates a system prompt for gene cluster analysis by assembling modular components.

    Assembly order:
    - Standard mode: MAIN TASK + CONTEXT + RULES/CRITERIA + OUTPUT_FORMAT
    - CoT mode: COT_INSTRUCTIONS (pre-assembled with all components) + CONTEXT

    COT_INSTRUCTIONS is built from modular steps via assemble_cot_instructions().
    Use assemble_cot_instructions() with custom steps for prompt permutation testing.

    Args:
        screen_name: Name of the screen for output directory naming
        screen_context_path: Path to screen_context.json file
        override_screen_context: If True, use default context (testing utility)
        template_path: Path to custom template file (escape hatch - full control)
        template_string: Custom template string (escape hatch - full control)
        CoT_mode: If True, use chain-of-thought instructions
        output_dir: Optional output directory for saving prompts

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
    if CoT_mode and override_CoT_steps:
        prompt = assemble_cot_instructions(override_CoT_steps, screen_context=SCREEN_CONTEXT_TEXT)
    elif CoT_mode:
        prompt = assemble_cot_instructions(COT_STEPS_DEFAULT, screen_context=SCREEN_CONTEXT_TEXT)
    else:
        prompt = (
            CLUSTER_ANALYSIS_TASK
            + "\n\nThe following experimental context is provided: "
            + SCREEN_CONTEXT_TEXT
            + "\n\n"
            + GENE_CATEGORIZATION_RULES
            + "\n\n"
            + NOVEL_CLASSIFICATION_RULES
            + "\n\n"
            + UNCHARACTERIZED_CLASSIFICATION_RULES
            + "\n\n"
            + PATHWAY_CONFIDENCE_CRITERIA
            + "\n\n"
            + OUTPUT_FORMAT_JSON
        )
    if not output_dir:
        output_dir = Path(f"output/{screen_name}_analysis/prompts_used/")
    output_dir.mkdir(exist_ok=True)
    # save system prompt to file with timestamp
    with open(
        output_dir
        / f"cluster_analysis_phase1_system_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(prompt)
    return prompt


def make_single_cluster_analysis_user_prompt(cluster_id, screen_name, cluster_to_bundle_path_map):
    BUNDLE_PATH = cluster_to_bundle_path_map[str(cluster_id)]

    # Build a user prompt from the bundle JSON
    bundle_obj = json.loads(Path(BUNDLE_PATH).read_text(encoding="utf-8"))
    bundle_text = json.dumps(bundle_obj, ensure_ascii=False)

    output_dir = Path(f"output/{screen_name}_analysis/prompts_used/")
    output_dir.mkdir(exist_ok=True)
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


## from previous version; may be reindroduced later
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
