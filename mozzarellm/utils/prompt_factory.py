"""
Prompt construction utilities for gene cluster analysis.

This module assembles prompts by concatenating modular components from prompts.py
in a standardized order. Components are organized in prompts.py to match assembly order.
"""

import os


def make_cluster_analysis_prompt(
    cluster_id,
    genes,
    gene_annotations_dict=None,
    screen_context=None,
    template_path=None,
    template_string=None,
    retrieved_context=None,
    cot_instructions=None,
):
    """
    Create a prompt for gene cluster analysis by assembling modular components.

    Default assembly order (non-RAG, non-CoT):
    1. CLUSTER_ANALYSIS_TASK (formatted with cluster_id, genes)
    2. GENE_CLASSIFICATION_RULES
    3. SCREEN INFORMATION (benchmark-specific or default)
    4. PATHWAY_CONFIDENCE_CRITERIA (always included)
    5. Gene annotations (if provided)
    6. OUTPUT_FORMAT_JSON

    When RAG/CoT are enabled, retrieved evidence and CoT instructions are inserted
    between pathway confidence criteria and gene annotations.

    Custom templates (escape hatch):
    If template_path or template_string is provided, it completely replaces the
    modular assembly. Use {cluster_id} and {gene_list} as placeholders.

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
    gene_list = ", ".join(genes)

    # =========================================================================
    # ESCAPE HATCH: Custom template overrides everything
    # =========================================================================
    if template_string or template_path:
        prompt = _load_custom_template(template_path, template_string)
        prompt = prompt.format(cluster_id=str(cluster_id), gene_list=gene_list)

        # Still add optional components if provided
        if retrieved_context and isinstance(retrieved_context, dict):
            prompt += _format_retrieved_evidence(retrieved_context)

        if cot_instructions and isinstance(cot_instructions, str):
            prompt += f"\n\nREASONING STEPS:\n{cot_instructions}\n"

        if gene_annotations_dict:
            prompt += _format_gene_annotations(gene_annotations_dict, genes)

        return prompt

    # =========================================================================
    # DEFAULT: Assemble from modular components
    # =========================================================================
    from mozzarellm.prompts import (
        CLUSTER_ANALYSIS_TASK,
        DEFAULT_SCREEN_CONTEXT,
        GENE_CLASSIFICATION_RULES,
        OUTPUT_FORMAT_JSON,
        PATHWAY_CONFIDENCE_CRITERIA,
    )

    # =========================================================================
    # SECTION 1: Core task (formatted)
    # =========================================================================
    prompt = CLUSTER_ANALYSIS_TASK.format(cluster_id=str(cluster_id), gene_list=gene_list)

    # =========================================================================
    # SECTION 2: Gene classification rules
    # =========================================================================
    prompt += "\n\n" + GENE_CLASSIFICATION_RULES

    # =========================================================================
    # SECTION 3: Screen context (custom or default)
    # =========================================================================
    if screen_context:
        prompt += f"\n\nSCREEN INFORMATION:\n{screen_context}\n"
    else:
        prompt += f"\n\nSCREEN INFORMATION:\n{DEFAULT_SCREEN_CONTEXT}\n"

    # =========================================================================
    # SECTION 4: Pathway confidence criteria (always included)
    # =========================================================================
    prompt += "\n" + PATHWAY_CONFIDENCE_CRITERIA

    # =========================================================================
    # SECTION 5: Retrieved evidence (RAG mode)
    # =========================================================================
    if retrieved_context and isinstance(retrieved_context, dict):
        prompt += _format_retrieved_evidence(retrieved_context)

    # =========================================================================
    # SECTION 6: Chain-of-thought instructions (CoT mode)
    # =========================================================================
    if cot_instructions and isinstance(cot_instructions, str):
        prompt += f"\n\nREASONING STEPS (keep to brief bullet points; no long prose):\n{cot_instructions}\n"

    # =========================================================================
    # SECTION 7: Gene annotations
    # =========================================================================
    if gene_annotations_dict:
        prompt += _format_gene_annotations(gene_annotations_dict, genes)

    # =========================================================================
    # SECTION 8: Output format (always last)
    # =========================================================================
    prompt += "\n\n" + OUTPUT_FORMAT_JSON

    return prompt


# =============================================================================
# Helper functions for custom templates and optional components
# =============================================================================


def _load_custom_template(template_path=None, template_string=None):
    """Load a custom template from file or string."""
    if template_string:
        print("Using provided custom template string")
        return template_string

    if template_path and os.path.exists(template_path):
        print(f"Loading custom template from: {template_path}")
        with open(template_path) as f:
            return f.read()

    raise ValueError("Custom template path does not exist or template string not provided")


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
        if src == "annotations":
            src_tag = f"Gene:{meta.get('gene', '')}"
        elif src == "knowledge_file":
            src_tag = f"File:{meta.get('path', '')} (score:{meta.get('score', 0)})"
        else:
            src_tag = "Screen Context"

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


def _format_gene_annotations(gene_annotations_dict, genes):
    """Format gene functional annotations."""
    feature_text = "\nAdditional gene information:\n"
    relevant_feature_count = 0

    for gene in genes:
        if gene in gene_annotations_dict:
            feature_text += f"{gene}: {gene_annotations_dict[gene]}\n"
            relevant_feature_count += 1

    # Only add the feature section if we found relevant features
    if relevant_feature_count > 0:
        feature_explanation = """
IMPORTANT: The additional gene information provided above should be used to:
1. Better determine if genes are truly UNCHARACTERIZED
2. Evaluate potential pathway connections for NOVEL_ROLE genes
3. Identify ESTABLISHED genes for the dominant process
"""
        print(f"Added {relevant_feature_count} gene feature descriptions to prompt")
        return feature_text + feature_explanation
    else:
        print("No relevant gene features found for this cluster")
        return ""
