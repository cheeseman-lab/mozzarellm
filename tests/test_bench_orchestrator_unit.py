"""Unit tests for bench_orchestrator helper functions.

No API calls; uses tmp_path and synthetic data only.

Inline-MCP route tests (3a_mcp) were removed when MCP became the
single_call_mcp enrichment; no remaining mode has route.mcp=True.
"""

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import pandas as pd
    from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_orchestrator import (
        _build_run_id,
        _build_timing_dict,
        _filter_clusters,
        _resolve_bundle_path,
        _resolve_screen_context_path,
        execute_single_run,
    )
    from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_configparse import (
        BenchmarkConfig,
        ClusterFilter,
        PathsConfig,
        RunConfig,
        TimingConfig,
    )
    from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_routes import (
        MODE_REGISTRY,
        Route,
    )

    _IMPORTS_OK = True
except (ImportError, OSError):
    _IMPORTS_OK = False

pytestmark = pytest.mark.skipif(not _IMPORTS_OK, reason="pandas or orchestrator unavailable")


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_clusters_df(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Build a minimal DataFrame with screen_name/cluster_id columns."""
    return pd.DataFrame(pairs, columns=["screen_name", "cluster_id"])


def _config_with_filters(
    screens_include="all",
    clusters_include="all",
) -> BenchmarkConfig:
    cfg = BenchmarkConfig()
    cfg.screens_include = screens_include
    cfg.clusters_include = clusters_include
    return cfg


# ═════════════════════════════════════════════════════════════════════════════
# TestHelpers
# ═════════════════════════════════════════════════════════════════════════════


class TestHelpers:
    def test_build_run_id(self):
        result = _build_run_id("exp", "single_call", "denali", "21", 1)
        assert result == "exp__single_call__denali__cluster_21__rep_1"

    def test_resolve_bundle_path_exact_match(self, tmp_path):
        bundle = tmp_path / "denali__cluster_21__bundle.json"
        bundle.write_text("{}", encoding="utf-8")
        result = _resolve_bundle_path(tmp_path, "denali", "21")
        assert result == bundle

    def test_resolve_bundle_path_no_match(self, tmp_path):
        result = _resolve_bundle_path(tmp_path, "denali", "21")
        assert result is None

    def test_resolve_screen_context_path_exists(self, tmp_path):
        ctx = tmp_path / "denali_screen_context.json"
        ctx.write_text("{}", encoding="utf-8")
        result = _resolve_screen_context_path(tmp_path, "denali")
        assert result == ctx

    def test_resolve_screen_context_path_missing(self, tmp_path):
        result = _resolve_screen_context_path(tmp_path, "denali")
        assert result is None

    def test_filter_clusters_all_screens(self):
        df = _make_clusters_df([("denali", "21"), ("aconcagua", "5")])
        cfg = _config_with_filters(screens_include="all", clusters_include="all")
        pairs = _filter_clusters(df, cfg)
        assert set(pairs) == {("denali", "21"), ("aconcagua", "5")}

    def test_filter_clusters_screen_filter(self):
        df = _make_clusters_df([("denali", "21"), ("aconcagua", "5")])
        cfg = _config_with_filters(screens_include=["denali"])
        pairs = _filter_clusters(df, cfg)
        assert pairs == [("denali", "21")]

    def test_filter_clusters_explicit_cluster_list(self):
        df = _make_clusters_df([("denali", "21"), ("denali", "22"), ("aconcagua", "5")])
        cfg = _config_with_filters(clusters_include=[ClusterFilter("denali", "21")])
        pairs = _filter_clusters(df, cfg)
        assert pairs == [("denali", "21")]


# ═════════════════════════════════════════════════════════════════════════════
# TestTimingDict
# ═════════════════════════════════════════════════════════════════════════════


class TestTimingDict:
    def test_all_flags_true(self):
        route = MODE_REGISTRY["single_call"]
        timing_cfg = TimingConfig(
            track_full_run=True,
            track_prompt_construction=True,
            track_model_latency=True,
            track_metrics=True,
            track_io=True,
            track_step_latencies=True,
            track_mcp_tool_latency=True,
        )
        result = _build_timing_dict(
            route=route,
            t_prompt=0.1,
            t_model=1.5,
            t_metrics=0.05,
            t_io=0.02,
            t_full_run=2.0,
            raw_outputs={"steps": []},
            timing_cfg=timing_cfg,
        )
        assert result["full_run_time_seconds"] == 2.0
        assert result["prompt_construction_seconds"] == 0.1
        assert result["model_latency_seconds"] == 1.5
        assert result["metrics_time_seconds"] == 0.05
        assert result["io_write_time_seconds"] == 0.02
        assert result["n_api_calls"] == 1

    def test_all_flags_false(self):
        route = MODE_REGISTRY["single_call"]
        timing_cfg = TimingConfig(
            track_full_run=False,
            track_prompt_construction=False,
            track_model_latency=False,
            track_metrics=False,
            track_io=False,
            track_step_latencies=False,
            track_mcp_tool_latency=False,
        )
        result = _build_timing_dict(
            route=route,
            t_prompt=0.1,
            t_model=1.5,
            t_metrics=0.05,
            t_io=0.02,
            t_full_run=2.0,
            raw_outputs={"steps": []},
            timing_cfg=timing_cfg,
        )
        assert result["full_run_time_seconds"] is None
        assert result["prompt_construction_seconds"] is None
        assert result["model_latency_seconds"] is None
        assert result["metrics_time_seconds"] is None
        assert result["io_write_time_seconds"] is None
        assert result["n_api_calls"] == 1

    def test_multi_turn_with_steps(self):
        route = MODE_REGISTRY["stepwise"]
        raw_outputs = {
            "steps": [
                {"elapsed_s": 0.5, "component": "cPH"},
                {"elapsed_s": 1.0, "component": "cGCR"},
            ],
        }
        timing_cfg = TimingConfig(track_step_latencies=True)
        result = _build_timing_dict(
            route=route,
            t_prompt=0.0,
            t_model=0.0,
            t_metrics=0.0,
            t_io=0.0,
            t_full_run=0.0,
            raw_outputs=raw_outputs,
            timing_cfg=timing_cfg,
        )
        assert result["n_api_calls"] == 2
        assert result["step_latencies_seconds"] is not None
        assert len(result["step_latencies_seconds"]) == 2
        assert result["slowest_step_component"] == "cGCR"

    def test_single_call_route(self):
        route = MODE_REGISTRY["single_call"]
        result = _build_timing_dict(
            route=route,
            t_prompt=0.0,
            t_model=0.0,
            t_metrics=0.0,
            t_io=0.0,
            t_full_run=0.0,
            raw_outputs={"steps": []},
        )
        assert result["n_api_calls"] == 1
        assert result["step_latencies_seconds"] is None


# ═════════════════════════════════════════════════════════════════════════════
# TestDryRunExecution — exercises execute_single_run in dry_run mode
# ═════════════════════════════════════════════════════════════════════════════


def _make_dry_run_config(output_dir: Path) -> BenchmarkConfig:
    """Build a BenchmarkConfig suitable for a dry-run test."""
    cfg = BenchmarkConfig()
    cfg.experiment_id = "test_exp"
    cfg.run = RunConfig(
        dry_run=True,
        workflow_testing=True,
        overwrite_outputs=False,
        save_prompts=True,
        save_raw_outputs=True,
        save_parsed_outputs=True,
        save_traces=True,
    )
    cfg.paths = PathsConfig(output_dir=output_dir)
    return cfg


def _setup_bundle_and_context(tmp_path: Path) -> tuple[Path, Path]:
    """Write minimal evidence bundle and screen context JSONs for dry-run."""
    bundle_path = tmp_path / "denali__cluster_21__bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                "cluster_genes": [
                    {"gene_symbol": "GeneA"},
                    {"gene_symbol": "GeneB"},
                    {"gene_symbol": "GeneC"},
                ]
            }
        ),
        encoding="utf-8",
    )
    ctx_path = tmp_path / "denali_screen_context.json"
    ctx_path.write_text(
        json.dumps(
            {
                "screen_name": "denali",
                "assay_type": "genetic screen",
                "target_phenotype": "cell division",
                "organism": "Danio rerio",
                "cell_line_or_system": "zebrafish embryo",
                "perturbation": {
                    "type": "CRISPR knockout",
                    "library_or_reagent": "GeCKO v2",
                },
                "readout": {
                    "measurement": "fluorescence",
                    "instrument_or_platform": "FACS",
                    "primary_metric": "fold change",
                },
                "clustering": {
                    "method": "hierarchical",
                    "parameters": {"n_clusters": 10},
                },
                "controls": {
                    "negative_controls": "non-targeting sgRNA",
                    "positive_controls": "essential gene sgRNA",
                },
                "provenance": {
                    "dataset_name": "denali",
                    "citation": "Cheesman Lab 2024",
                    "data_source": "in-house",
                },
            }
        ),
        encoding="utf-8",
    )
    return bundle_path, ctx_path


class TestDryRunExecution:
    def test_dry_run_returns_record_without_error(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config = _make_dry_run_config(output_dir)
        bundle_path, ctx_path = _setup_bundle_and_context(tmp_path)

        record = execute_single_run(
            route=MODE_REGISTRY["single_call"],
            screen_name="denali",
            cluster_id="21",
            bundle_path=bundle_path,
            screen_context_path=ctx_path,
            replicate=1,
            config=config,
            client=None,
            output_dir=output_dir,
        )
        assert record["error"] is None
        assert record["parsed_output"] is not None

    def test_dry_run_record_has_expected_keys(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config = _make_dry_run_config(output_dir)
        bundle_path, ctx_path = _setup_bundle_and_context(tmp_path)

        record = execute_single_run(
            route=MODE_REGISTRY["single_call"],
            screen_name="denali",
            cluster_id="21",
            bundle_path=bundle_path,
            screen_context_path=ctx_path,
            replicate=1,
            config=config,
            client=None,
            output_dir=output_dir,
        )
        expected_keys = {
            "run_id",
            "route",
            "mode",
            "mcp",
            "parsed_output",
            "metrics",
            "timing",
            "component_order",
            "screen_name",
            "cluster_id",
            "replicate",
        }
        assert expected_keys.issubset(record.keys())

    def test_dry_run_metrics_populated(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config = _make_dry_run_config(output_dir)
        bundle_path, ctx_path = _setup_bundle_and_context(tmp_path)

        record = execute_single_run(
            route=MODE_REGISTRY["single_call"],
            screen_name="denali",
            cluster_id="21",
            bundle_path=bundle_path,
            screen_context_path=ctx_path,
            replicate=1,
            config=config,
            client=None,
            output_dir=output_dir,
        )
        metrics = record["metrics"]
        assert metrics["json_parse_success"] is True
        assert metrics["schema_compliance"] is True
        assert metrics["gene_completeness"] == 1.0

    def test_dry_run_writes_trace_file(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config = _make_dry_run_config(output_dir)
        bundle_path, ctx_path = _setup_bundle_and_context(tmp_path)

        record = execute_single_run(
            route=MODE_REGISTRY["single_call"],
            screen_name="denali",
            cluster_id="21",
            bundle_path=bundle_path,
            screen_context_path=ctx_path,
            replicate=1,
            config=config,
            client=None,
            output_dir=output_dir,
        )
        trace_path = Path(record["trace_path"])
        assert trace_path.exists()
        trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
        assert trace_data["route"] == "single_call"
        assert trace_data["parsed_output"] is not None

    def test_dry_run_with_wording_overrides(self, tmp_path):
        """Phase 3: component_overrides and condition_name flow through to the record."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config = _make_dry_run_config(output_dir)
        bundle_path, ctx_path = _setup_bundle_and_context(tmp_path)

        extra_fields = {
            "wording_source": "wording_v1",
            "wording_target_id": "W3",
            "wording_hypothesis": "test hypothesis",
        }
        record = execute_single_run(
            route=MODE_REGISTRY["single_call"],
            screen_name="denali",
            cluster_id="21",
            bundle_path=bundle_path,
            screen_context_path=ctx_path,
            replicate=1,
            config=config,
            client=None,
            output_dir=output_dir,
            component_overrides={"GCR": "Override text for GCR."},
            condition_name="single_call_w_wording_v1.W3",
            extra_record_fields=extra_fields,
        )
        assert record["route"] == "single_call_w_wording_v1.W3"
        assert record["wording_target_id"] == "W3"
        assert record["wording_source"] == "wording_v1"
        assert "single_call_w_wording_v1.W3" in record["run_id"]

    def test_dry_run_order_variant_metadata(self, tmp_path):
        """Phase 2: order variant metadata appears in the record."""
        from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_order import (
            apply_order_variant,
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config = _make_dry_run_config(output_dir)
        bundle_path, ctx_path = _setup_bundle_and_context(tmp_path)

        order_route = apply_order_variant(MODE_REGISTRY["single_call"], "O1")
        record = execute_single_run(
            route=order_route,
            screen_name="denali",
            cluster_id="21",
            bundle_path=bundle_path,
            screen_context_path=ctx_path,
            replicate=1,
            config=config,
            client=None,
            output_dir=output_dir,
        )
        assert record["base_route"] == "single_call"
        assert record["order_variant"] == "late_screen_context"
        assert record["order_hypothesis"] is not None
        assert record["component_order"] != list(MODE_REGISTRY["single_call"].component_order)

    def test_filter_clusters_empty_df(self):
        df = _make_clusters_df([])
        cfg = _config_with_filters()
        pairs = _filter_clusters(df, cfg)
        assert pairs == []

    def test_filter_clusters_deduplication(self):
        df = _make_clusters_df(
            [
                ("denali", "21"),
                ("denali", "21"),
                ("aconcagua", "5"),
            ]
        )
        cfg = _config_with_filters()
        pairs = _filter_clusters(df, cfg)
        assert len(pairs) == 2
        assert set(pairs) == {("denali", "21"), ("aconcagua", "5")}
