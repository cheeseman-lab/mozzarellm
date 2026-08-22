"""Unit tests for bench_dry_run.py.

No API calls; uses tmp_path and synthetic data only.
"""

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_dry_run import (
    _load_bundle_genes,
    generate_mock_parsed_output,
    generate_mock_raw_outputs,
)
from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_routes import (
    ROUTE_REGISTRY,
)


class TestMockOutputs:
    def test_mock_parsed_non_mcp(self):
        genes = ["A", "B", "C", "D", "E"]
        parsed = generate_mock_parsed_output("21", genes, ROUTE_REGISTRY["single_call"])
        assert parsed["cluster_id"] == "21"
        assert parsed["pathway_confidence"] == "Low"
        assert set(parsed["established_genes"]) == set(genes)
        assert len(parsed["established_genes"]) == 5

    def test_mock_parsed_mcp(self):
        genes = ["X", "Y"]
        parsed = generate_mock_parsed_output("5", genes, ROUTE_REGISTRY["single_call_mcp"])
        assert "literature_informed_reclassifications" in parsed
        assert "literature_informed_pathway_revision" in parsed

    def test_mock_raw_outputs(self):
        raw = generate_mock_raw_outputs(ROUTE_REGISTRY["single_call"])
        for key in ("response_text", "tool_calls", "input_tokens", "elapsed_s", "error", "steps"):
            assert key in raw

    def test_load_bundle_genes(self, tmp_path):
        bundle = {
            "cluster_genes": [
                {"gene_symbol": "X"},
                {"gene_symbol": "Y"},
            ]
        }
        bundle_path = tmp_path / "bundle.json"
        bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
        result = _load_bundle_genes(bundle_path)
        assert result == ["X", "Y"]

    def test_load_bundle_genes_empty(self, tmp_path):
        bundle = {"cluster_genes": []}
        bundle_path = tmp_path / "bundle.json"
        bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
        result = _load_bundle_genes(bundle_path)
        assert result == []
