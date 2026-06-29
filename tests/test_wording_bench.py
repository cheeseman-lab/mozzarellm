"""Tests for the Phase 3 wording benchmark (hypothesis-driven design).

Covers the acceptance criteria from the wording benchmark spec:
  - alternate text lives outside the orchestrator
  - targets live outside the orchestrator and the YAML config
  - targets may declare their own default source
  - config can specify targets: all / list / range
  - default_source and force_source resolution
  - canonical (W0) is always included
  - no combinatorial explosion (only explicitly selected targets)
  - output records identify source/target/hypothesis/components
"""

import sys
from pathlib import Path

import pytest

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.arch_bench_routes import (
    ROUTE_REGISTRY,
)
from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.wording_bench_alternates import (
    WORDING_ALTERNATE_SET_REGISTRY,
)
from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.wording_bench_targets import (
    WORDING_OVERRIDE_TARGET_REGISTRY,
    WordingOverrideRun,
    WordingOverrideTarget,
    build_wording_override_runs,
    resolve_source,
    resolve_target_ids,
)
from mozzarellm.prompt_components import COMPONENT_REGISTRY
from mozzarellm.utils.prompt_factory import make_cluster_analysis_system_prompt


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
# Target registry
# ============================================================================


class TestTargetRegistry:
    def test_w0_through_w5_exist(self):
        assert set(WORDING_OVERRIDE_TARGET_REGISTRY) >= {"W0", "W1", "W2", "W3", "W4", "W5"}

    def test_w0_is_canonical_no_components(self):
        w0 = WORDING_OVERRIDE_TARGET_REGISTRY["W0"]
        assert w0.components == ()
        assert w0.source is None

    def test_non_canonical_targets_have_components_and_source(self):
        for tid, t in WORDING_OVERRIDE_TARGET_REGISTRY.items():
            if tid == "W0":
                continue
            assert t.components, f"{tid} must override at least one component"
            assert t.source is not None, f"{tid} must declare a default source"

    def test_w4_groups_npr_upr(self):
        assert WORDING_OVERRIDE_TARGET_REGISTRY["W4"].components == ("NPR", "UPR")


# ============================================================================
# Source resolution
# ============================================================================


class TestResolveSource:
    def test_canonical_resolves_none(self):
        w0 = WORDING_OVERRIDE_TARGET_REGISTRY["W0"]
        assert resolve_source(w0) is None

    def test_target_source_used(self):
        w1 = WORDING_OVERRIDE_TARGET_REGISTRY["W1"]
        assert resolve_source(w1) == "wording_v1"

    def test_force_source_wins(self):
        w1 = WORDING_OVERRIDE_TARGET_REGISTRY["W1"]
        assert (
            resolve_source(w1, default_source="wording_v1", force_source="wording_v2")
            == "wording_v2"
        )

    def test_default_source_used_when_target_none(self):
        t = WordingOverrideTarget(
            id="WX", name="x", hypothesis="h", description="d", components=("CAT",), source=None
        )
        assert resolve_source(t, default_source="wording_v1") == "wording_v1"

    def test_error_when_no_source(self):
        t = WordingOverrideTarget(
            id="WX", name="x", hypothesis="h", description="d", components=("CAT",), source=None
        )
        with pytest.raises(ValueError):
            resolve_source(t)


# ============================================================================
# Target selection (all / list / range)
# ============================================================================


class TestResolveTargetIds:
    def test_all(self):
        ids = resolve_target_ids("all")
        assert ids[0] == "W0"
        assert set(ids) == set(WORDING_OVERRIDE_TARGET_REGISTRY)

    def test_range(self):
        ids = resolve_target_ids("W1-W3")
        assert ids == ["W0", "W1", "W2", "W3"]

    def test_explicit_list_always_includes_w0(self):
        ids = resolve_target_ids(["W1", "W3", "W5"])
        assert ids == ["W0", "W1", "W3", "W5"]

    def test_single_string(self):
        assert resolve_target_ids("W2") == ["W0", "W2"]

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            resolve_target_ids(["W99"])

    def test_no_duplicate_w0(self):
        ids = resolve_target_ids(["W0", "W1"])
        assert ids.count("W0") == 1


# ============================================================================
# Builder
# ============================================================================


class TestBuildWordingOverrideRuns:
    def test_canonical_always_included(self):
        runs = build_wording_override_runs(["3a"], ["W1"])
        canon = [r for r in runs if r.target_id == "W0"]
        assert len(canon) == 1
        assert canon[0].component_overrides == {}
        assert canon[0].source is None

    def test_no_combinatorial_explosion(self):
        # 1 base route x (W0 + W1 + W3) = 3 runs, not 2**n combinations.
        runs = build_wording_override_runs(["3a"], ["W1", "W3"])
        assert len(runs) == 3
        assert {r.target_id for r in runs} == {"W0", "W1", "W3"}

    def test_overrides_resolved_from_source(self):
        runs = build_wording_override_runs(["3a"], ["W4"])
        w4 = next(r for r in runs if r.target_id == "W4")
        v1 = WORDING_ALTERNATE_SET_REGISTRY["wording_v1"]
        assert w4.component_overrides == {"NPR": v1["NPR"], "UPR": v1["UPR"]}

    def test_force_source_applies(self):
        # force_source=wording_v2 only defines CAT; W1 needs CAT -> ok.
        runs = build_wording_override_runs(["3a"], ["W1"], force_source="wording_v2")
        w1 = next(r for r in runs if r.target_id == "W1")
        assert w1.source == "wording_v2"
        assert w1.component_overrides == {
            "CAT": WORDING_ALTERNATE_SET_REGISTRY["wording_v2"]["CAT"]
        }

    def test_force_source_missing_component_raises(self):
        # wording_v2 has no GCR; forcing it for W3 (GCR) must fail validation.
        with pytest.raises(ValueError):
            build_wording_override_runs(["3a"], ["W3"], force_source="wording_v2")

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError):
            build_wording_override_runs(["3a"], ["W1"], force_source="does_not_exist")

    def test_unknown_base_route_raises(self):
        with pytest.raises(ValueError):
            build_wording_override_runs(["nope"], ["W1"])

    def test_multiple_base_routes(self):
        runs = build_wording_override_runs(["3a", "3b"], ["W1"])
        # (W0 + W1) x 2 routes = 4
        assert len(runs) == 4
        assert {r.base_route.name for r in runs} == {"3a", "3b"}


# ============================================================================
# Run-route naming
# ============================================================================


class TestRunRouteName:
    def test_canonical_name(self):
        runs = build_wording_override_runs(["3a"], ["W1"])
        canon = next(r for r in runs if r.target_id == "W0")
        assert canon.run_route_name == "3a_canonical_W0_canonical"

    def test_override_name(self):
        runs = build_wording_override_runs(["3a"], ["W4"])
        w4 = next(r for r in runs if r.target_id == "W4")
        assert w4.run_route_name == "3a_wording_v1_W4_terse_prioritization_rules"


# ============================================================================
# Prompt construction hook (overrides actually change the prompt)
# ============================================================================


class TestComponentOverridesPropagate:
    def test_override_changes_standard_prompt(self):
        runs = build_wording_override_runs(["3a"], ["W3"])
        w3 = next(r for r in runs if r.target_id == "W3")

        base = make_cluster_analysis_system_prompt(
            screen_name="denali", mode="standard", mcp=False, override_screen_context=True
        )
        overridden = make_cluster_analysis_system_prompt(
            screen_name="denali",
            mode="standard",
            mcp=False,
            override_screen_context=True,
            component_overrides=w3.component_overrides,
        )
        assert base != overridden
        assert w3.component_overrides["GCR"].strip()[:40] in overridden

    def test_canonical_matches_no_override(self):
        base = make_cluster_analysis_system_prompt(
            screen_name="denali", mode="standard", mcp=False, override_screen_context=True
        )
        canon = make_cluster_analysis_system_prompt(
            screen_name="denali",
            mode="standard",
            mcp=False,
            override_screen_context=True,
            component_overrides={},
        )
        assert base == canon


# ============================================================================
# Config parsing
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
            "    - 3a\n"
            "  targets: W1-W3\n"
            "  default_source: wording_v1\n",
            encoding="utf-8",
        )
        cfg = load_config(cfg_file)
        assert cfg.wording_benchmark.enabled is True
        assert cfg.wording_benchmark.base_routes == ["3a"]
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


# ============================================================================
# Supplements — edge-case coverage
# ============================================================================


class TestWordingBenchSupplements:
    def test_cot_route_with_gcr_override_silent(self):
        # Currently no warning — override silently has no effect on CoT prompt
        # because CoT uses cGCR.
        runs = build_wording_override_runs(["3b"], ["W3"])
        w3 = next(r for r in runs if r.target_id == "W3")
        assert w3.component_overrides  # override was resolved without error

    def test_run_route_name_round_trips_through_regex(self):
        from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_trace_parser import (
            TRACE_FILENAME_RE,
        )

        runs = build_wording_override_runs(["3a"], ["W3"])
        w3 = next(r for r in runs if r.target_id == "W3")
        filename = f"exp__{w3.run_route_name}__denali__cluster_21__rep_1.json"
        match = TRACE_FILENAME_RE.match(filename)
        assert match is not None, f"TRACE_FILENAME_RE failed to parse: {filename}"
        assert match.group("route") == w3.run_route_name
