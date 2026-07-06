"""Unit tests for bench_configparse.py — config loading, defaults, path resolution."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_configparse import (
    ArchitectureBenchmarkConfig,
    BenchmarkConfig,
    ClusterFilter,
    ModelConfig,
    OrderBenchmarkConfig,
    PathsConfig,
    RunConfig,
    TimingConfig,
    WordingBenchmarkConfig,
    load_config,
)


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    """Write *data* to ``tmp_path / configs / test.yaml`` and return the path."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = cfg_dir / "test.yaml"
    cfg_file.write_text(yaml.dump(data), encoding="utf-8")
    return cfg_file


# ── Config loading ───────────────────────────────────────────────────────────


class TestConfigLoading:
    def test_minimal_config(self, tmp_path):
        cfg_file = _write_yaml(tmp_path, {"experiment_id": "min_test"})
        cfg = load_config(cfg_file)

        assert cfg.experiment_id == "min_test"
        assert cfg.model.provider == ModelConfig().provider
        assert cfg.run.num_replicates == RunConfig().num_replicates
        assert cfg.timing.track_full_run == TimingConfig().track_full_run

    def test_full_config(self, tmp_path):
        data = {
            "experiment_id": "full_test",
            "model": {
                "provider": "openai",
                "model_name": "gpt-4o",
                "temperature": 0.5,
                "max_tokens": 8000,
            },
            "run": {
                "num_replicates": 5,
                "max_workers": 8,
                "dry_run": True,
                "workflow_testing": True,
            },
            "paths": {
                "output_dir": "my_outputs",
            },
            "timing": {
                "track_full_run": False,
                "track_io": False,
            },
        }
        cfg_file = _write_yaml(tmp_path, data)
        cfg = load_config(cfg_file)

        assert cfg.experiment_id == "full_test"
        assert cfg.model.provider == "openai"
        assert cfg.model.model_name == "gpt-4o"
        assert cfg.model.temperature == 0.5
        assert cfg.model.max_tokens == 8000
        assert cfg.run.num_replicates == 5
        assert cfg.run.max_workers == 8
        assert cfg.run.dry_run is True
        assert cfg.run.workflow_testing is True
        assert cfg.timing.track_full_run is False
        assert cfg.timing.track_io is False

    def test_relative_paths_resolved(self, tmp_path):
        data = {
            "experiment_id": "paths_test",
            "paths": {
                "output_dir": "results/arch",
                "evidence_bundles_dir": "bundles",
            },
        }
        cfg_file = _write_yaml(tmp_path, data)
        cfg = load_config(cfg_file)
        base_dir = cfg_file.parent.parent  # configs/ → tmp_path

        assert cfg.paths.output_dir == base_dir / "results" / "arch"
        assert cfg.paths.evidence_bundles_dir == base_dir / "bundles"

    def test_missing_config_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_screens_include_all(self, tmp_path):
        cfg_file = _write_yaml(
            tmp_path,
            {
                "experiment_id": "scr_all",
                "screens": {"include": "all"},
            },
        )
        cfg = load_config(cfg_file)
        assert cfg.screens_include == "all"

    def test_screens_include_list(self, tmp_path):
        cfg_file = _write_yaml(
            tmp_path,
            {
                "experiment_id": "scr_list",
                "screens": {"include": ["denali", "whitney"]},
            },
        )
        cfg = load_config(cfg_file)
        assert cfg.screens_include == ["denali", "whitney"]

    def test_clusters_include_all(self, tmp_path):
        cfg_file = _write_yaml(
            tmp_path,
            {
                "experiment_id": "cl_all",
                "clusters": {"include": "all"},
            },
        )
        cfg = load_config(cfg_file)
        assert cfg.clusters_include == "all"

    def test_clusters_include_explicit(self, tmp_path):
        cfg_file = _write_yaml(
            tmp_path,
            {
                "experiment_id": "cl_explicit",
                "clusters": {
                    "include": [
                        {"screen_name": "denali", "cluster_id": 21},
                        {"screen_name": "whitney", "cluster_id": 5},
                    ]
                },
            },
        )
        cfg = load_config(cfg_file)
        assert isinstance(cfg.clusters_include, list)
        assert len(cfg.clusters_include) == 2
        assert isinstance(cfg.clusters_include[0], ClusterFilter)
        assert cfg.clusters_include[0].screen_name == "denali"
        assert cfg.clusters_include[0].cluster_id == "21"

    def test_architecture_benchmark_section(self, tmp_path):
        cfg_file = _write_yaml(
            tmp_path,
            {
                "experiment_id": "ab_test",
                "architecture_benchmark": {
                    "enabled": True,
                    "base_routes": ["3a", "3b"],
                },
            },
        )
        cfg = load_config(cfg_file)
        assert cfg.architecture_benchmark.enabled is True
        assert cfg.architecture_benchmark.base_routes == ["3a", "3b"]

    def test_architecture_benchmark_defaults(self, tmp_path):
        cfg_file = _write_yaml(tmp_path, {"experiment_id": "ab_def"})
        cfg = load_config(cfg_file)
        assert cfg.architecture_benchmark.enabled is True
        assert len(cfg.architecture_benchmark.base_routes) == 6

    def test_legacy_routes_key_maps_to_architecture_benchmark(self, tmp_path):
        cfg_file = _write_yaml(
            tmp_path,
            {
                "experiment_id": "legacy",
                "routes": {"include": ["3a", "3c"]},
            },
        )
        cfg = load_config(cfg_file)
        assert cfg.architecture_benchmark.enabled is True
        assert cfg.architecture_benchmark.base_routes == ["3a", "3c"]

    def test_order_benchmark_section(self, tmp_path):
        cfg_file = _write_yaml(
            tmp_path,
            {
                "experiment_id": "ob_test",
                "order_benchmark": {
                    "enabled": True,
                    "base_routes": ["3a", "3b"],
                    "variants": ["reverse", "shuffle"],
                },
            },
        )
        cfg = load_config(cfg_file)
        assert cfg.order_benchmark.enabled is True
        assert cfg.order_benchmark.base_routes == ["3a", "3b"]
        assert cfg.order_benchmark.variants == ["reverse", "shuffle"]

    def test_wording_benchmark_section(self, tmp_path):
        cfg_file = _write_yaml(
            tmp_path,
            {
                "experiment_id": "wb_test",
                "wording_benchmark": {
                    "enabled": True,
                    "base_routes": ["3a"],
                    "targets": ["W1", "W3"],
                    "default_source": "gpt4o",
                },
            },
        )
        cfg = load_config(cfg_file)
        assert cfg.wording_benchmark.enabled is True
        assert cfg.wording_benchmark.base_routes == ["3a"]
        assert cfg.wording_benchmark.targets == ["W1", "W3"]
        assert cfg.wording_benchmark.default_source == "gpt4o"

    def test_experiment_output_dir_no_workflow_testing(self, tmp_path):
        cfg_file = _write_yaml(
            tmp_path,
            {
                "experiment_id": "exp1",
                "run": {"workflow_testing": False},
                "paths": {"output_dir": "out"},
            },
        )
        cfg = load_config(cfg_file)
        base_dir = cfg_file.parent.parent

        # Non-workflow runs get bundle_source + timestamp: exp1_uniprot_YYYYMMDD_HHMMSS
        output_dir = cfg.experiment_output_dir
        assert output_dir.parent == base_dir / "out"
        assert output_dir.name.startswith("exp1_uniprot_")
        # Check format: exp1 (4) + _uniprot_ (9) + YYYYMMDD (8) + _ (1) + HHMMSS (6) = 28 chars
        assert len(output_dir.name) == 28

    def test_experiment_output_dir_with_workflow_testing(self, tmp_path):
        cfg_file = _write_yaml(
            tmp_path,
            {
                "experiment_id": "exp2",
                "run": {"workflow_testing": True},
                "paths": {"output_dir": "out"},
            },
        )
        cfg = load_config(cfg_file)
        base_dir = cfg_file.parent.parent

        assert cfg.experiment_output_dir == base_dir / "out" / "_workflow_testing" / "exp2"

    def test_experiment_output_dir_with_overwrite_outputs(self, tmp_path):
        cfg_file = _write_yaml(
            tmp_path,
            {
                "experiment_id": "exp3",
                "run": {"workflow_testing": False, "overwrite_outputs": True},
                "paths": {"output_dir": "out"},
            },
        )
        cfg = load_config(cfg_file)
        base_dir = cfg_file.parent.parent

        # overwrite_outputs uses consistent path (no timestamp)
        assert cfg.experiment_output_dir == base_dir / "out" / "exp3"


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestConfigEdgeCases:
    def test_unknown_yaml_keys_ignored(self, tmp_path):
        cfg_file = _write_yaml(
            tmp_path,
            {
                "experiment_id": "unk",
                "some_future_key": {"nested": True},
                "another_unknown": 42,
            },
        )
        cfg = load_config(cfg_file)
        assert cfg.experiment_id == "unk"

    def test_timing_partial(self, tmp_path):
        cfg_file = _write_yaml(
            tmp_path,
            {
                "experiment_id": "tp",
                "timing": {"track_full_run": False},
            },
        )
        cfg = load_config(cfg_file)

        assert cfg.timing.track_full_run is False
        assert cfg.timing.track_prompt_construction == TimingConfig().track_prompt_construction
        assert cfg.timing.track_model_latency == TimingConfig().track_model_latency

    def test_wording_targets_string_vs_list(self, tmp_path):
        cfg_str = _write_yaml(
            tmp_path,
            {
                "experiment_id": "wt_str",
                "wording_benchmark": {"enabled": True, "targets": "all"},
            },
        )
        cfg1 = load_config(cfg_str)
        assert cfg1.wording_benchmark.targets == "all"

        cfg_list = _write_yaml(
            tmp_path,
            {
                "experiment_id": "wt_list",
                "wording_benchmark": {"enabled": True, "targets": ["W1"]},
            },
        )
        cfg2 = load_config(cfg_list)
        assert cfg2.wording_benchmark.targets == ["W1"]
