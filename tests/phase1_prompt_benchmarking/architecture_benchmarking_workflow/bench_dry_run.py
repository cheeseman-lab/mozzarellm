"""Dry-run mock output generation for architecture benchmarking.

Produces deterministic mock parsed outputs and raw_outputs so the benchmark
infrastructure can be tested end-to-end without API credentials or cost.
"""

from __future__ import annotations

import json
from pathlib import Path

from .arch_bench_routes import Route


def _load_bundle_genes(bundle_path: Path) -> list[str]:
    """Extract gene symbols from an evidence bundle JSON."""
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    return [g["gene_symbol"] for g in bundle.get("cluster_genes", []) if g.get("gene_symbol")]


def generate_mock_parsed_output(
    cluster_id: str,
    genes: list[str],
    route: Route,
) -> dict:
    """Generate a deterministic mock parsed output.

    Classifies all genes as ESTABLISHED so gene-completeness checks pass.
    """
    parsed = {
        "cluster_id": cluster_id,
        "dominant_process": "Mock pathway",
        "pathway_confidence": "Low",
        "established_genes": genes,
        "uncharacterized_genes": [],
        "novel_role_genes": [],
        "summary": "Mock output for dry-run validation of benchmark infrastructure.",
    }

    if route.mcp:
        parsed["literature_informed_reclassifications"] = []
        parsed["literature_informed_pathway_revision"] = {
            "pre_literature_pathway": "Mock pathway",
            "post_literature_pathway": "Mock pathway",
            "pathway_changed": False,
            "rationale": "Dry-run mock — no literature validation performed.",
        }

    parsed["_validation_metadata"] = {
        "mode": route.mode_tag,
        "model": "dry-run",
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "time_seconds": 0.0,
        "tool_calls": 0,
    }
    return parsed


def generate_mock_raw_outputs(route: Route) -> dict:
    """Generate deterministic mock raw_outputs dict."""
    raw = {
        "response_text": "[dry-run] No API call made.",
        "tool_calls": [],
        "input_tokens": 0,
        "output_tokens": 0,
        "elapsed_s": 0.0,
        "cost_usd": 0.0,
        "pricing_warning": None,
        "schema_warnings": [],
        "error": None,
        "steps": [],
    }
    return raw
