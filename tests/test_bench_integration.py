"""Integration tests for run_benchmark — exercises the full dispatch + loop.

Calls run_benchmark end-to-end with dry_run=True for each phase (architecture,
order, wording). Verifies that the unified runner correctly dispatches, builds
RunSpecs, and produces records/manifests/traces without any live API calls.

Inline-MCP routes (3a_mcp, 3b_mcp, 3c_mcp) were removed when MCP became the
single_call_mcp enrichment; integration_test.yaml's architecture_benchmark
now lists only the 3 live modes.
"""

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pd = pytest.importorskip("pandas")

from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_configparse import (
    load_config,
)
from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_orchestrator import (
    run_benchmark,
)

_CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "phase1_prompt_benchmarking"
    / "configs"
    / "integration_test.yaml"
)

_SCREEN_CONTEXT = {
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


def _setup_fixture_dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create the minimal on-disk fixtures for a dry-run benchmark."""
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    bundles_dir = tmp_path / "bundles"
    bundles_dir.mkdir()

    clusters_csv = inputs_dir / "benchmark_clusters.csv"
    clusters_csv.write_text(
        "screen_name,cluster_id,gene_symbol\n"
        "denali,21,AURKA\n"
        "denali,21,PLK1\n"
        "denali,77,FANCG\n"
        "denali,77,GCLC\n",
        encoding="utf-8",
    )

    (inputs_dir / "denali_screen_context.json").write_text(
        json.dumps(_SCREEN_CONTEXT), encoding="utf-8"
    )

    (bundles_dir / "denali__cluster_21__bundle.json").write_text(
        json.dumps(
            {
                "cluster_genes": [
                    {"gene_symbol": "AURKA"},
                    {"gene_symbol": "PLK1"},
                ]
            }
        ),
        encoding="utf-8",
    )

    (bundles_dir / "denali__cluster_77__bundle.json").write_text(
        json.dumps(
            {
                "cluster_genes": [
                    {"gene_symbol": "FANCG"},
                    {"gene_symbol": "GCLC"},
                ]
            }
        ),
        encoding="utf-8",
    )

    return inputs_dir, bundles_dir, clusters_csv


def _load_and_patch(tmp_path: Path) -> "BenchmarkConfig":
    """Load the integration config and point all paths at tmp_path fixtures."""
    config = load_config(_CONFIG_PATH)
    inputs_dir, bundles_dir, clusters_csv = _setup_fixture_dirs(tmp_path)

    config.paths.benchmark_inputs_dir = inputs_dir
    config.paths.benchmark_clusters_csv = clusters_csv
    config.paths.evidence_bundles_dir = bundles_dir
    config.paths.output_dir = tmp_path / "output"
    return config


# ============================================================================
# Phase 1: Architecture
# ============================================================================


class TestPhase1Architecture:
    # 3 routes × 2 clusters × 1 rep = 6 records
    EXPECTED_ROUTES = {"single_call", "cot", "stepwise"}
    EXPECTED_CLUSTERS = {"21", "77"}
    EXPECTED_COUNT = len(EXPECTED_ROUTES) * len(EXPECTED_CLUSTERS)  # 6

    def test_produces_records(self, tmp_path):
        config = _load_and_patch(tmp_path)
        records = run_benchmark(config)

        assert len(records) == self.EXPECTED_COUNT
        observed_routes = {r["route"] for r in records}
        assert observed_routes == self.EXPECTED_ROUTES
        observed_clusters = {r["cluster_id"] for r in records}
        assert observed_clusters == self.EXPECTED_CLUSTERS
        for rec in records:
            assert rec.get("error") is None

    def test_writes_manifest(self, tmp_path):
        config = _load_and_patch(tmp_path)
        run_benchmark(config)

        manifest_path = config.experiment_output_dir / "run_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["total_runs"] == self.EXPECTED_COUNT
        assert manifest["phase"] == "architecture"

    def test_writes_config_snapshot(self, tmp_path):
        config = _load_and_patch(tmp_path)
        run_benchmark(config)

        snap_path = config.experiment_output_dir / "config_snapshot.yaml"
        assert snap_path.exists()
        snap = json.loads(snap_path.read_text(encoding="utf-8"))
        assert "architecture_benchmark" in snap
        assert "order_benchmark" not in snap
        assert "wording_benchmark" not in snap

    def test_writes_trace(self, tmp_path):
        config = _load_and_patch(tmp_path)
        run_benchmark(config)

        traces = list(config.experiment_output_dir.rglob("*.json"))
        trace_files = [
            t for t in traces if t.name != "run_manifest.json" and t.name != "config_snapshot.yaml"
        ]
        assert len(trace_files) >= self.EXPECTED_COUNT


# ============================================================================
# Phase 2: Order
# ============================================================================


class TestPhase2Order:
    # O + O1 + O2 = 3 variants × 2 clusters × 1 rep = 6 records
    EXPECTED_COUNT = 6

    def test_produces_records_with_order_metadata(self, tmp_path):
        config = _load_and_patch(tmp_path)
        config.order_benchmark.enabled = True

        records = run_benchmark(config)

        assert len(records) == self.EXPECTED_COUNT
        route_names = {r["route"] for r in records}
        assert any("O_canonical" in n for n in route_names)
        assert any("O1_" in n for n in route_names)
        assert any("O2_" in n for n in route_names)
        observed_clusters = {r["cluster_id"] for r in records}
        assert observed_clusters == {"21", "77"}

    def test_order_metadata_on_records(self, tmp_path):
        config = _load_and_patch(tmp_path)
        config.order_benchmark.enabled = True

        records = run_benchmark(config)
        for rec in records:
            assert rec.get("error") is None
            assert "order_variant" in rec
            assert "base_route" in rec
            assert rec["base_route"] == "single_call"

    def test_manifest_has_phase(self, tmp_path):
        config = _load_and_patch(tmp_path)
        config.order_benchmark.enabled = True
        run_benchmark(config)

        manifest = json.loads(
            (config.experiment_output_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["phase"] == "order"
        assert manifest["total_runs"] == self.EXPECTED_COUNT

    def test_config_snapshot_has_order_section(self, tmp_path):
        config = _load_and_patch(tmp_path)
        config.order_benchmark.enabled = True
        run_benchmark(config)

        snap = json.loads(
            (config.experiment_output_dir / "config_snapshot.yaml").read_text(encoding="utf-8")
        )
        assert "order_benchmark" in snap
        assert "wording_benchmark" not in snap


# ============================================================================
# Phase 3: Wording
# ============================================================================


class TestPhase3Wording:
    # W0 + W1 = 2 conditions × 2 clusters × 1 rep = 4 records
    EXPECTED_COUNT = 4

    def test_produces_records_with_wording_metadata(self, tmp_path):
        config = _load_and_patch(tmp_path)
        config.wording_benchmark.enabled = True

        records = run_benchmark(config)

        assert len(records) == self.EXPECTED_COUNT
        target_ids = {r.get("wording_target_id") for r in records}
        assert target_ids == {"W0", "W1"}
        observed_clusters = {r["cluster_id"] for r in records}
        assert observed_clusters == {"21", "77"}

    def test_wording_route_name_format(self, tmp_path):
        config = _load_and_patch(tmp_path)
        config.wording_benchmark.enabled = True

        records = run_benchmark(config)
        for rec in records:
            assert rec.get("error") is None
            route = rec["route"]
            assert route.startswith("single_call_")
            assert "__" not in route

    def test_manifest_has_phase(self, tmp_path):
        config = _load_and_patch(tmp_path)
        config.wording_benchmark.enabled = True
        run_benchmark(config)

        manifest = json.loads(
            (config.experiment_output_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["phase"] == "wording"
        assert manifest["total_runs"] == self.EXPECTED_COUNT

    def test_config_snapshot_has_wording_section(self, tmp_path):
        config = _load_and_patch(tmp_path)
        config.wording_benchmark.enabled = True
        run_benchmark(config)

        snap = json.loads(
            (config.experiment_output_dir / "config_snapshot.yaml").read_text(encoding="utf-8")
        )
        assert "wording_benchmark" in snap
        assert "order_benchmark" not in snap
