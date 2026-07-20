"""Unit tests for bench_reportgen.py.

No API calls; uses tmp_path and synthetic data only.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_reportgen import (
    _build_report_markdown,
    _summarize_route,
    _write_aggregate_csv,
    generate_report,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_record(
    route: str,
    metrics: dict,
    timing: dict | None = None,
    error: str | None = None,
    order_variant: str = "none",
) -> dict:
    """Build a minimal record dict for report-gen tests."""
    rec = {
        "run_id": f"{route}__test__cluster_1__rep_1",
        "experiment_id": "test_exp",
        "route": route,
        "mode": "standard",
        "mcp": False,
        "delivery": "single_call",
        "screen_name": "denali",
        "cluster_id": "1",
        "replicate": 1,
        "model_name": "test-model",
        "temperature": 0.2,
        "system_prompt_hash": "abc123",
        "user_prompt_hash": "def456",
        "evidence_bundle_path": "/fake/bundle.json",
        "screen_context_path": "/fake/ctx.json",
        "parsed_output": {},
        "metrics": metrics,
        "timing": timing or {},
        "trace_path": "/fake/trace.json",
        "latency_seconds": 1.0,
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "estimated_cost_usd": 0.01,
        "schema_warnings": [],
        "mcp_tool_calls": [],
        "error": error,
        "base_route": route,
        "order_variant": order_variant,
        "order_hypothesis": None,
        "component_order": [],
        "system_components": [],
        "user_turns_order": [],
    }
    return rec


def _base_metrics(
    schema_compliance: bool = True,
    gene_completeness: float = 1.0,
    json_parse_success: bool = True,
) -> dict:
    return {
        "schema_compliance": schema_compliance,
        "gene_completeness": gene_completeness,
        "json_parse_success": json_parse_success,
        "cluster_id_exact_match": True,
        "valid_confidence_value": True,
        "valid_subclass_values": True,
        "no_duplicate_genes": True,
        "categories_mutually_exclusive": True,
        "latency_seconds": 1.0,
        "input_tokens": 100,
        "output_tokens": 50,
        "estimated_cost_usd": 0.01,
    }


_MINIMAL_CONFIG = {
    "experiment_id": "test_exp",
    "model": {"model_name": "test-model"},
    "run": {"num_replicates": 1, "dry_run": True},
}


# ═════════════════════════════════════════════════════════════════════════════
# TestSummarizeRoute
# ═════════════════════════════════════════════════════════════════════════════


class TestSummarizeRoute:
    def test_three_compliant_records(self):
        records = [
            _make_record("single_call", _base_metrics()),
            _make_record("single_call", _base_metrics()),
            _make_record("single_call", _base_metrics()),
        ]
        summary = _summarize_route(records)
        assert summary["n_runs"] == 3
        assert summary["error_rate"] == 0
        assert summary["schema_compliance_rate"] == 1.0

    def test_one_error_record(self):
        records = [
            _make_record("single_call", _base_metrics()),
            _make_record("single_call", _base_metrics(), error="fail"),
            _make_record("single_call", _base_metrics()),
        ]
        summary = _summarize_route(records)
        assert summary["n_errors"] == 1
        assert summary["error_rate"] == pytest.approx(1 / 3)

    def test_empty_list(self):
        summary = _summarize_route([])
        assert summary == {}

    def test_timing_stats(self):
        records = [
            _make_record(
                "single_call",
                _base_metrics(),
                timing={"full_run_time_seconds": 1.0, "n_api_calls": 1},
            ),
            _make_record(
                "single_call",
                _base_metrics(),
                timing={"full_run_time_seconds": 2.0, "n_api_calls": 1},
            ),
            _make_record(
                "single_call",
                _base_metrics(),
                timing={"full_run_time_seconds": 3.0, "n_api_calls": 1},
            ),
        ]
        summary = _summarize_route(records)
        assert summary["mean_full_run_time_seconds"] == pytest.approx(2.0)
        assert summary["median_full_run_time_seconds"] == pytest.approx(2.0)

    def test_derived_seconds_per_schema_ok(self):
        records = [
            _make_record(
                "single_call",
                _base_metrics(schema_compliance=True),
                timing={"full_run_time_seconds": 5.0},
            ),
            _make_record(
                "single_call",
                _base_metrics(schema_compliance=True),
                timing={"full_run_time_seconds": 5.0},
            ),
        ]
        summary = _summarize_route(records)
        assert summary["seconds_per_schema_compliant_output"] == pytest.approx(5.0)


# ═════════════════════════════════════════════════════════════════════════════
# TestReportGeneration
# ═════════════════════════════════════════════════════════════════════════════


class TestReportGeneration:
    def test_generate_report_creates_files(self, tmp_path):
        records = [
            _make_record("single_call", _base_metrics()),
            _make_record("cot", _base_metrics()),
        ]
        report_path = generate_report(records, _MINIMAL_CONFIG, tmp_path)
        assert report_path.exists()
        assert (tmp_path / "aggregate_summary.json").exists()
        assert (tmp_path / "aggregate_summary.csv").exists()
        assert report_path.stat().st_size > 0
        assert (tmp_path / "aggregate_summary.json").stat().st_size > 0
        assert (tmp_path / "aggregate_summary.csv").stat().st_size > 0

    def test_report_contains_headers(self, tmp_path):
        records = [_make_record("single_call", _base_metrics())]
        report_path = generate_report(records, _MINIMAL_CONFIG, tmp_path)
        text = report_path.read_text(encoding="utf-8")
        assert "Route Comparison" in text
        assert "Structural Quality" in text
        assert "Efficiency" in text
        assert "Timing" in text

    def test_order_variant_section_present(self, tmp_path):
        records = [
            _make_record(
                "single_call_O1_late_screen_context",
                _base_metrics(),
                order_variant="late_screen_context",
            ),
        ]
        report_path = generate_report(records, _MINIMAL_CONFIG, tmp_path)
        text = report_path.read_text(encoding="utf-8")
        assert "Order Variant Comparison" in text

    def test_order_variant_section_absent(self, tmp_path):
        records = [
            _make_record("single_call", _base_metrics(), order_variant="none"),
        ]
        report_path = generate_report(records, _MINIMAL_CONFIG, tmp_path)
        text = report_path.read_text(encoding="utf-8")
        assert "Order Variant Comparison" not in text

    def test_csv_rows_match_routes(self, tmp_path):
        records = [
            _make_record("single_call", _base_metrics()),
            _make_record("cot", _base_metrics()),
            _make_record("cot", _base_metrics()),
        ]
        generate_report(records, _MINIMAL_CONFIG, tmp_path)
        csv_path = tmp_path / "aggregate_summary.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        route_names = {row["route"] for row in rows}
        assert route_names == {"single_call", "cot"}
        assert len(rows) == 2
