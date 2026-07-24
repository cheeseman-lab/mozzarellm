"""Tests for the wording alternate-text registry and wording config parsing.

Covers:
  - alternate component text lives outside the orchestrator
  - alternate override keys are real component-registry keys and differ from canonical
  - the wording_benchmark config section parses (enabled / base_routes / targets / source)
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mozzarellm.prompt_components import COMPONENT_REGISTRY
from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_wording_alternates import (
    WORDING_ALTERNATE_SET_REGISTRY,
)

# ============================================================================
# Alternate source registry
# ============================================================================


class TestAlternateRegistry:
    def test_sources_exist(self):
        assert "wording_v1" in WORDING_ALTERNATE_SET_REGISTRY
        assert "wording_v2" in WORDING_ALTERNATE_SET_REGISTRY

    def test_v1_covers_w1_and_w3_w5_components(self):
        v1 = WORDING_ALTERNATE_SET_REGISTRY["wording_v1"]
        for key in ("CAT", "GCR", "NPR", "UPR", "PCC"):
            assert key in v1
            assert isinstance(v1[key], str) and v1[key].strip()

    def test_alternate_keys_are_known_components(self):
        # Every override key must be a real component-registry key.
        for source, mapping in WORDING_ALTERNATE_SET_REGISTRY.items():
            for key in mapping:
                assert key in COMPONENT_REGISTRY, f"{source}:{key} not in COMPONENT_REGISTRY"

    def test_alternate_text_differs_from_canonical(self):
        v1 = WORDING_ALTERNATE_SET_REGISTRY["wording_v1"]
        for key, text in v1.items():
            assert text != COMPONENT_REGISTRY[key], f"{key} alternate equals canonical"


# ============================================================================
# Wording config parsing
# ============================================================================


class TestWordingBenchmarkConfig:
    def test_parse_minimal(self, tmp_path):
        from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_configparse import (
            load_config,
        )

        cfg_dir = tmp_path / "configs"
        cfg_dir.mkdir()
        cfg_file = cfg_dir / "wb.yaml"
        cfg_file.write_text(
            "experiment_id: wb_test\n"
            "wording_benchmark:\n"
            "  enabled: true\n"
            "  base_routes:\n"
            "    - single_call\n"
            "  targets: W1-W3\n"
            "  default_source: wording_v1\n",
            encoding="utf-8",
        )
        cfg = load_config(cfg_file)
        assert cfg.wording_benchmark.enabled is True
        assert cfg.wording_benchmark.base_routes == ["single_call"]
        assert cfg.wording_benchmark.targets == "W1-W3"
        assert cfg.wording_benchmark.default_source == "wording_v1"
        assert cfg.wording_benchmark.force_source is None

    def test_default_disabled(self, tmp_path):
        from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_configparse import (
            load_config,
        )

        cfg_dir = tmp_path / "configs"
        cfg_dir.mkdir()
        cfg_file = cfg_dir / "wb.yaml"
        cfg_file.write_text("experiment_id: wb_test\n", encoding="utf-8")
        cfg = load_config(cfg_file)
        assert cfg.wording_benchmark.enabled is False
