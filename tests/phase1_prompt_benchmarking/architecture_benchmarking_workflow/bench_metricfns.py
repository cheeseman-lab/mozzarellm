"""Metric computation functions for architecture benchmarking.

Computes structural, logical, efficiency, and robustness metrics per run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from mozzarellm.schemas.mcp_schemas import (
    LiteraturePathwayRevision,
    LiteratureReclassification,
    LiteratureValidation,
)

VALID_CONFIDENCE_VALUES = {"High", "Medium", "Low"}
VALID_NOVEL_SUBCLASSES = {
    "NO_EVIDENCE",
    "INDIRECT_EVIDENCE",
    "PARTIAL_EVIDENCE",
    "CONTRADICTORY_EVIDENCE",
}
VALID_UNCHARACTERIZED_SUBCLASSES = {
    "DARK_GENE",
    "NASCENT",
    "ANNOTATED_ONLY",
    "NON_HUMAN_CHARACTERIZED",
}


# =============================================================================
# STRUCTURAL METRICS
# =============================================================================


def compute_structural_metrics(
    parsed: dict | None,
    cluster_id: str,
    bundle_genes: list[str],
) -> dict[str, Any]:
    """Compute structural quality metrics for a parsed LLM output."""
    metrics: dict[str, Any] = {}

    # json_parse_success: was the response parseable at all?
    metrics["json_parse_success"] = parsed is not None
    if parsed is None:
        metrics["schema_compliance"] = False
        metrics["required_fields_present"] = False
        metrics["cluster_id_exact_match"] = False
        metrics["valid_confidence_value"] = False
        metrics["valid_subclass_values"] = False
        metrics["gene_completeness"] = 0.0
        metrics["gene_completeness_count"] = 0
        metrics["gene_total_count"] = len(bundle_genes)
        return metrics

    # required_fields_present
    required_fields = [
        "cluster_id",
        "dominant_process",
        "pathway_confidence",
        "established_genes",
        "uncharacterized_genes",
        "novel_role_genes",
        "summary",
    ]
    present = [f for f in required_fields if f in parsed]
    metrics["required_fields_present"] = len(present) == len(required_fields)
    metrics["required_fields_missing"] = [f for f in required_fields if f not in parsed]

    # cluster_id_exact_match
    metrics["cluster_id_exact_match"] = str(parsed.get("cluster_id", "")) == str(cluster_id)

    # valid_confidence_value
    confidence = parsed.get("pathway_confidence", "")
    metrics["valid_confidence_value"] = confidence in VALID_CONFIDENCE_VALUES

    # valid_subclass_values (check subclass values on novel/uncharacterized genes)
    subclass_valid = True
    for gene_entry in parsed.get("novel_role_genes", []):
        if isinstance(gene_entry, dict):
            cls = gene_entry.get("class", "")
            if cls and cls not in VALID_NOVEL_SUBCLASSES:
                subclass_valid = False
                break
    for gene_entry in parsed.get("uncharacterized_genes", []):
        if isinstance(gene_entry, dict):
            cls = gene_entry.get("class", "")
            if cls and cls not in VALID_UNCHARACTERIZED_SUBCLASSES:
                subclass_valid = False
                break
    metrics["valid_subclass_values"] = subclass_valid

    # gene_completeness: what fraction of input genes appear in the output?
    output_genes = set()
    for g in parsed.get("established_genes", []):
        if isinstance(g, str):
            output_genes.add(g)
    for g in parsed.get("novel_role_genes", []):
        if isinstance(g, dict):
            output_genes.add(g.get("gene", ""))
        elif isinstance(g, str):
            output_genes.add(g)
    for g in parsed.get("uncharacterized_genes", []):
        if isinstance(g, dict):
            output_genes.add(g.get("gene", ""))
        elif isinstance(g, str):
            output_genes.add(g)
    output_genes.discard("")

    bundle_gene_set = set(bundle_genes)
    matched = output_genes & bundle_gene_set
    metrics["gene_completeness"] = len(matched) / len(bundle_gene_set) if bundle_gene_set else 1.0
    metrics["gene_completeness_count"] = len(matched)
    metrics["gene_total_count"] = len(bundle_gene_set)

    # schema_compliance: all structural checks pass
    metrics["schema_compliance"] = (
        metrics["required_fields_present"]
        and metrics["cluster_id_exact_match"]
        and metrics["valid_confidence_value"]
        and metrics["valid_subclass_values"]
    )

    return metrics


# =============================================================================
# MCP-SPECIFIC METRICS
# =============================================================================


def compute_mcp_metrics(
    parsed: dict | None,
    raw_outputs: dict,
    mcp_enabled: bool,
) -> dict[str, Any]:
    """Compute MCP-specific metrics. Returns nulls for non-MCP routes."""
    metrics: dict[str, Any] = {}
    metrics["mcp_enabled"] = mcp_enabled

    if not mcp_enabled:
        metrics["mcp_preflight_status"] = None
        metrics["n_mcp_tool_calls"] = None
        metrics["mcp_servers_used"] = None
        metrics["literature_schema_warnings"] = None
        return metrics

    tool_calls = raw_outputs.get("tool_calls", [])
    metrics["n_mcp_tool_calls"] = len(tool_calls)
    metrics["mcp_servers_used"] = list(
        {tc.get("server_name", "unknown") for tc in tool_calls if isinstance(tc, dict)}
    )
    metrics["literature_schema_warnings"] = raw_outputs.get("schema_warnings", [])
    metrics["mcp_preflight_status"] = "passed"  # if we got here, preflight passed

    # Validate literature blocks via Pydantic schemas
    if parsed:
        lit_warnings = _validate_literature_output(parsed)
        if lit_warnings:
            existing = metrics["literature_schema_warnings"] or []
            metrics["literature_schema_warnings"] = existing + lit_warnings

    return metrics


def _validate_literature_output(parsed: dict) -> list[str]:
    """Soft-validate literature-specific fields using Pydantic schemas."""
    warnings = []

    # Validate literature_informed_pathway_revision
    revision = parsed.get("literature_informed_pathway_revision")
    if revision:
        try:
            LiteraturePathwayRevision.model_validate(revision)
        except ValidationError as e:
            warnings.append(f"literature_informed_pathway_revision: {e.error_count()} error(s)")

    # Validate literature_informed_reclassifications
    reclassifications = parsed.get("literature_informed_reclassifications", [])
    for i, entry in enumerate(reclassifications):
        try:
            LiteratureReclassification.model_validate(entry)
        except ValidationError as e:
            warnings.append(
                f"literature_informed_reclassifications[{i}]: {e.error_count()} error(s)"
            )

    # Validate per-gene literature_validation blocks
    for category_key in ("novel_role_genes", "uncharacterized_genes"):
        for i, gene_entry in enumerate(parsed.get(category_key, [])):
            if not isinstance(gene_entry, dict):
                continue
            lit_val = gene_entry.get("literature_validation")
            if lit_val:
                try:
                    LiteratureValidation.model_validate(lit_val)
                except ValidationError as e:
                    gene_name = gene_entry.get("gene", f"index_{i}")
                    warnings.append(
                        f"{category_key}.{gene_name}.literature_validation: {e.error_count()} error(s)"
                    )

    return warnings


# =============================================================================
# LOGICAL CONSISTENCY METRICS
# =============================================================================


def compute_logical_metrics(parsed: dict | None) -> dict[str, Any]:
    """Compute logical consistency checks on the parsed output."""
    metrics: dict[str, Any] = {}

    if parsed is None:
        metrics["no_duplicate_genes"] = None
        metrics["categories_mutually_exclusive"] = None
        metrics["summary_present"] = None
        return metrics

    # Collect all genes across categories
    established = set()
    for g in parsed.get("established_genes", []):
        if isinstance(g, str):
            established.add(g)
    novel = set()
    for g in parsed.get("novel_role_genes", []):
        if isinstance(g, dict):
            novel.add(g.get("gene", ""))
        elif isinstance(g, str):
            novel.add(g)
    uncharacterized = set()
    for g in parsed.get("uncharacterized_genes", []):
        if isinstance(g, dict):
            uncharacterized.add(g.get("gene", ""))
        elif isinstance(g, str):
            uncharacterized.add(g)

    # Remove blanks
    established.discard("")
    novel.discard("")
    uncharacterized.discard("")

    all_genes = list(established) + list(novel) + list(uncharacterized)
    metrics["no_duplicate_genes"] = len(all_genes) == len(set(all_genes))

    # categories_mutually_exclusive: no gene appears in more than one category
    overlaps = (established & novel) | (established & uncharacterized) | (novel & uncharacterized)
    metrics["categories_mutually_exclusive"] = len(overlaps) == 0
    if overlaps:
        metrics["overlapping_genes"] = sorted(overlaps)

    # summary_present and non-trivial
    summary = parsed.get("summary", "")
    metrics["summary_present"] = bool(summary and len(str(summary).strip()) > 10)

    return metrics


# =============================================================================
# EFFICIENCY METRICS
# =============================================================================


def compute_efficiency_metrics(raw_outputs: dict) -> dict[str, Any]:
    """Compute cost/latency/token metrics from raw_outputs."""
    return {
        "latency_seconds": raw_outputs.get("elapsed_s", 0.0),
        "input_tokens": raw_outputs.get("input_tokens", 0),
        "output_tokens": raw_outputs.get("output_tokens", 0),
        "total_tokens": (raw_outputs.get("input_tokens", 0) + raw_outputs.get("output_tokens", 0)),
        "estimated_cost_usd": raw_outputs.get("cost_usd", 0.0),
        "pricing_warning": raw_outputs.get("pricing_warning"),
    }


# =============================================================================
# AGGREGATE
# =============================================================================


def compute_all_metrics(
    parsed: dict | None,
    raw_outputs: dict,
    cluster_id: str,
    bundle_genes: list[str],
    mcp_enabled: bool,
) -> dict[str, Any]:
    """Compute all metric categories and return a flat metrics dict."""
    metrics = {}
    metrics.update(compute_structural_metrics(parsed, cluster_id, bundle_genes))
    metrics.update(compute_mcp_metrics(parsed, raw_outputs, mcp_enabled))
    metrics.update(compute_logical_metrics(parsed))
    metrics.update(compute_efficiency_metrics(raw_outputs))
    return metrics
