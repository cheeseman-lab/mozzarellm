"""Unit tests for bench_metricfns.py — structural, logical, MCP, efficiency metrics."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Bypass mozzarellm.__init__ (heavy SDK deps) — expose only the schemas sub-package.
_PKG_DIR = _REPO_ROOT / "mozzarellm"
for _name, _subdir in [
    ("mozzarellm", _PKG_DIR),
    ("mozzarellm.schemas", _PKG_DIR / "schemas"),
]:
    if _name not in sys.modules:
        _m = types.ModuleType(_name)
        _m.__path__ = [str(_subdir)]
        _m.__package__ = _name
        sys.modules[_name] = _m

from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_metricfns import (
    VALID_CONFIDENCE_VALUES,
    VALID_NOVEL_SUBCLASSES,
    VALID_UNCHARACTERIZED_SUBCLASSES,
    compute_all_metrics,
    compute_efficiency_metrics,
    compute_logical_metrics,
    compute_mcp_metrics,
    compute_structural_metrics,
)


def _make_valid_parsed(cluster_id: str, genes: list[str]) -> dict:
    """Build a fully valid parsed output dict with all required fields."""
    established = genes[:1] if genes else []
    novel = genes[1:2] if len(genes) > 1 else []
    uncharacterized = genes[2:] if len(genes) > 2 else []

    return {
        "cluster_id": cluster_id,
        "dominant_process": "cell signaling",
        "pathway_confidence": "High",
        "established_genes": established,
        "novel_role_genes": [
            {"gene": g, "class": "NO_EVIDENCE", "rationale": "test"} for g in novel
        ],
        "uncharacterized_genes": [
            {"gene": g, "class": "DARK_GENE", "rationale": "test"} for g in uncharacterized
        ],
        "summary": "This cluster is involved in cell signaling pathways.",
    }


# ── Structural metrics ──────────────────────────────────────────────────────


class TestStructuralMetrics:
    def test_valid_output_all_pass(self):
        genes = ["GeneA", "GeneB", "GeneC"]
        parsed = _make_valid_parsed("21", genes)
        m = compute_structural_metrics(parsed, "21", genes)

        assert m["json_parse_success"] is True
        assert m["schema_compliance"] is True
        assert m["required_fields_present"] is True
        assert m["gene_completeness"] == 1.0

    def test_parsed_none(self):
        genes = ["A", "B", "C"]
        m = compute_structural_metrics(None, "1", genes)

        assert m["json_parse_success"] is False
        assert m["schema_compliance"] is False
        assert m["gene_completeness"] == 0.0
        assert m["gene_total_count"] == 3

    def test_missing_required_field(self):
        parsed = _make_valid_parsed("5", ["X"])
        del parsed["summary"]
        m = compute_structural_metrics(parsed, "5", ["X"])

        assert m["required_fields_present"] is False
        assert "summary" in m["required_fields_missing"]

    def test_wrong_cluster_id(self):
        parsed = _make_valid_parsed("99", ["A"])
        m = compute_structural_metrics(parsed, "21", ["A"])

        assert m["cluster_id_exact_match"] is False
        assert m["schema_compliance"] is False

    def test_invalid_confidence(self):
        parsed = _make_valid_parsed("1", ["A"])
        parsed["pathway_confidence"] = "Very High"
        m = compute_structural_metrics(parsed, "1", ["A"])

        assert m["valid_confidence_value"] is False

    def test_invalid_novel_subclass(self):
        parsed = _make_valid_parsed("1", ["A", "B"])
        parsed["novel_role_genes"] = [{"gene": "B", "class": "INVALID"}]
        m = compute_structural_metrics(parsed, "1", ["A", "B"])

        assert m["valid_subclass_values"] is False

    def test_gene_completeness_partial(self):
        bundle = ["A", "B", "C", "D", "E"]
        parsed = _make_valid_parsed("1", ["A", "B", "C"])
        m = compute_structural_metrics(parsed, "1", bundle)

        assert m["gene_completeness"] == pytest.approx(0.6)
        assert m["gene_completeness_count"] == 3
        assert m["gene_total_count"] == 5

    def test_gene_completeness_extra_genes(self):
        bundle = ["A", "B"]
        parsed = _make_valid_parsed("1", ["A", "B", "EXTRA"])
        m = compute_structural_metrics(parsed, "1", bundle)

        assert m["gene_completeness"] == 1.0
        assert m["gene_completeness_count"] == 2
        assert m["gene_total_count"] == 2

    def test_empty_bundle(self):
        parsed = _make_valid_parsed("1", [])
        m = compute_structural_metrics(parsed, "1", [])

        assert m["gene_completeness"] == 1.0


# ── Logical metrics ─────────────────────────────────────────────────────────


class TestLogicalMetrics:
    def test_no_duplicates_exclusive(self):
        parsed = _make_valid_parsed("1", ["A", "B", "C"])
        m = compute_logical_metrics(parsed)

        assert m["no_duplicate_genes"] is True
        assert m["categories_mutually_exclusive"] is True

    def test_duplicate_across_categories(self):
        parsed = _make_valid_parsed("1", ["A", "B", "C"])
        parsed["novel_role_genes"].append({"gene": "A", "class": "NO_EVIDENCE"})
        m = compute_logical_metrics(parsed)

        assert m["categories_mutually_exclusive"] is False
        assert "A" in m["overlapping_genes"]

    def test_same_gene_established_and_novel(self):
        parsed = {
            "established_genes": ["X"],
            "novel_role_genes": [{"gene": "X", "class": "NO_EVIDENCE"}],
            "uncharacterized_genes": [],
            "summary": "A sufficiently long summary for testing purposes.",
        }
        m = compute_logical_metrics(parsed)

        assert m["categories_mutually_exclusive"] is False
        assert "X" in m["overlapping_genes"]

    def test_empty_summary(self):
        parsed = _make_valid_parsed("1", ["A"])
        parsed["summary"] = ""
        m = compute_logical_metrics(parsed)

        assert m["summary_present"] is False

    def test_parsed_none_logical(self):
        m = compute_logical_metrics(None)

        assert m["no_duplicate_genes"] is None
        assert m["categories_mutually_exclusive"] is None
        assert m["summary_present"] is None


# ── MCP metrics ──────────────────────────────────────────────────────────────


class TestMcpMetrics:
    def test_non_mcp_route(self):
        m = compute_mcp_metrics(None, {}, mcp_enabled=False)

        assert m["mcp_enabled"] is False
        assert m["n_mcp_tool_calls"] is None
        assert m["mcp_servers_used"] is None
        assert m["literature_schema_warnings"] is None

    def test_mcp_with_tool_calls(self):
        raw = {
            "tool_calls": [
                {"server_name": "pubmed"},
                {"server_name": "pubmed"},
            ]
        }
        parsed = _make_valid_parsed("1", ["A"])
        m = compute_mcp_metrics(parsed, raw, mcp_enabled=True)

        assert m["n_mcp_tool_calls"] == 2
        assert m["mcp_servers_used"] == ["pubmed"]

    def test_mcp_no_literature_warnings(self):
        parsed = _make_valid_parsed("1", ["A"])
        m = compute_mcp_metrics(parsed, {}, mcp_enabled=True)

        assert m["literature_schema_warnings"] == []

    def test_mcp_invalid_literature_revision(self):
        parsed = _make_valid_parsed("1", ["A"])
        parsed["literature_informed_pathway_revision"] = {"invalid": True}
        m = compute_mcp_metrics(parsed, {}, mcp_enabled=True)

        assert len(m["literature_schema_warnings"]) > 0


# ── Efficiency metrics ───────────────────────────────────────────────────────


class TestEfficiencyMetrics:
    def test_standard_raw_outputs(self):
        raw = {
            "elapsed_s": 1.5,
            "input_tokens": 100,
            "output_tokens": 200,
            "cost_usd": 0.01,
        }
        m = compute_efficiency_metrics(raw)

        assert m["latency_seconds"] == 1.5
        assert m["input_tokens"] == 100
        assert m["output_tokens"] == 200
        assert m["total_tokens"] == 300
        assert m["estimated_cost_usd"] == 0.01

    def test_missing_keys(self):
        m = compute_efficiency_metrics({})

        assert m["latency_seconds"] == 0.0
        assert m["input_tokens"] == 0
        assert m["output_tokens"] == 0
        assert m["total_tokens"] == 0
        assert m["estimated_cost_usd"] == 0.0
        assert m["pricing_warning"] is None


# ── Integration ──────────────────────────────────────────────────────────────


class TestIntegration:
    def test_compute_all_valid(self):
        genes = ["A", "B", "C"]
        parsed = _make_valid_parsed("10", genes)
        raw = {
            "elapsed_s": 2.0,
            "input_tokens": 50,
            "output_tokens": 100,
            "cost_usd": 0.005,
            "tool_calls": [{"server_name": "pubmed"}],
        }
        m = compute_all_metrics(parsed, raw, "10", genes, mcp_enabled=True)

        assert m["json_parse_success"] is True
        assert m["mcp_enabled"] is True
        assert m["n_mcp_tool_calls"] == 1
        assert m["latency_seconds"] == 2.0
        assert m["no_duplicate_genes"] is True

    def test_compute_all_none_no_mcp(self):
        m = compute_all_metrics(None, {}, "1", ["A"], mcp_enabled=False)

        assert m["json_parse_success"] is False
        assert m["mcp_enabled"] is False
        assert m["n_mcp_tool_calls"] is None
        assert m["no_duplicate_genes"] is None
        assert m["latency_seconds"] == 0.0
