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

import json
from datetime import datetime
from pathlib import Path

from mozzarellm.prompt_components import (
    CLUSTER_ANALYSIS_TASK,
    COT_SCREEN_CONTEXT,
    COT_STEPS_DEFAULT,
    COT_STEPS_UNIFIED_MCP,
    GENE_CATEGORIZATION_RULES,
    NOVEL_CLASSIFICATION_RULES,
    OUTPUT_FORMAT_JSON,
    PATHWAY_CONFIDENCE_CRITERIA,
    STEP_LITERATURE_VALIDATION,
    UNCHARACTERIZED_CLASSIFICATION_RULES,
)
from mozzarellm.utils.screen_context_utils import load_screen_context_json

VALID_MODES = ("standard", "cot", "stepwise")


def compose_cot_steps(mcp: bool) -> list[str]:
    """Canonical CoT step list. mcp=True inserts the literature-validation step.

    Used by `mode="cot"` (squashed into a numbered chain in one API call) and
    `mode="stepwise"` (each step delivered as a separate user turn).
    `mode="standard"` does NOT use this list — it assembles a flat rules-based
    prompt from `CLUSTER_ANALYSIS_TASK` + `*_RULES` + `OUTPUT_FORMAT_JSON`.
    """
    return COT_STEPS_UNIFIED_MCP if mcp else COT_STEPS_DEFAULT


def _inject_screen_context(steps: list[str], screen_context: str) -> list[str]:
    """Substitute the SCREEN_CONTEXT placeholder with the actual JSON context."""
    return [
        f"{COT_SCREEN_CONTEXT}\n{screen_context}" if step == COT_SCREEN_CONTEXT else step
        for step in steps
    ]


def make_cluster_analysis_system_prompt(
    *,
    screen_name: str,
    screen_context_path: Path | None = None,
    mode: str = "standard",
    mcp: bool = False,
    override_CoT_steps: list[str]
    | None = None,  # testing utility for prompt-permutation experiments
    override_screen_context: bool = False,  # testing utility
    template_path: Path | None = None,
    template_string: str | None = None,
    output_dir: Path | None = None,
):
    """
    Creates a system prompt for gene cluster analysis.

    Three modes, each producing a structurally distinct prompt:

      - standard: flat rules-based prompt — TASK + SCREEN_CONTEXT + GENE_CATEGORIZATION_RULES
                  + NOVEL_CLASSIFICATION_RULES + UNCHARACTERIZED_CLASSIFICATION_RULES
                  + PATHWAY_CONFIDENCE_CRITERIA + OUTPUT_FORMAT_JSON. No step structure.
                  This is the historical default. With mcp=True, the literature-validation
                  step is inserted before OUTPUT_FORMAT_JSON.

      - cot:      numbered chain-of-thought — `STEP 1 - ..., STEP 2 - ...` from
                  `compose_cot_steps(mcp)`. Single API call.

      - stepwise: same canonical step list as cot, but delivered as separate API turns.
                  System prompt holds only TASK + SCREEN_CONTEXT; the runner walks the
                  remaining reasoning steps as user turns (multi-turn conversation).

    Args:
        screen_name: Used for output directory naming.
        screen_context_path: Path to screen_context.json.
        mode: One of "standard" / "cot" / "stepwise". Default "standard".
        mcp: When True, attach the literature-validation step (cot+stepwise insert it
             into the canonical list; standard appends it before OUTPUT_FORMAT_JSON).
        override_CoT_steps: Custom step list for permutation testing (cot/stepwise only).
        override_screen_context: Use default placeholder context (testing utility).
        template_path / template_string: Escape hatch — bypass step assembly.
        output_dir: Where to save the assembled prompt for inspection.

    Returns:
        Fully assembled system prompt string.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

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

    if mode == "standard":
        # Historical default: flat rules-based prompt. No step structure.
        components = [
            CLUSTER_ANALYSIS_TASK,
            f"The following experimental context is provided: {SCREEN_CONTEXT_TEXT}",
            GENE_CATEGORIZATION_RULES,
            NOVEL_CLASSIFICATION_RULES,
            UNCHARACTERIZED_CLASSIFICATION_RULES,
            PATHWAY_CONFIDENCE_CRITERIA,
        ]
        if mcp:
            components.append(STEP_LITERATURE_VALIDATION)
        components.append(OUTPUT_FORMAT_JSON)
        prompt = "\n\n".join(components)
    elif mode == "cot":
        # Numbered chain-of-thought; one API call.
        steps = override_CoT_steps or compose_cot_steps(mcp)
        steps = _inject_screen_context(steps, SCREEN_CONTEXT_TEXT)
        prompt = "\n\n".join(f"STEP {i + 1} - {s}" for i, s in enumerate(steps))
    else:  # stepwise — system holds task + context only; runner delivers reasoning steps as separate API calls
        steps = override_CoT_steps or compose_cot_steps(mcp)
        steps = _inject_screen_context(steps, SCREEN_CONTEXT_TEXT)
        prompt = "\n\n".join(steps[:2])
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
    output_dir.mkdir(parents=True, exist_ok=True)
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
