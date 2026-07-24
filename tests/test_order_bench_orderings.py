"""Tests for order_bench_orderings — Phase 2 order variant generation and validation.

Inline-MCP route tests (3c_mcp) were removed when MCP became the
single_call_mcp enrichment; no remaining mode has route.mcp=True.
"""

import sys
from pathlib import Path

import pytest

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.arch_bench_routes import (
    Route,
    StepwiseTurn,
    build_routes_from_config,
    MODE_REGISTRY,
)
from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.order_bench_orderings import (
    ORDER_VARIANTS,
    _ORDER_SPECS,
    OrderVariant,
    apply_order_variant,
    build_order_benchmark_routes,
    resolve_order_variant_ids,
    validate_order_variant_names,
)
from mozzarellm.prompt_components import COMPONENT_REGISTRY


# ============================================================================
# Variant registry tests
# ============================================================================


class TestOrderVariantsRegistry:
    def test_all_five_variants_exist(self):
        expected = {"O", "O1", "O2", "O3", "O4"}
        assert set(ORDER_VARIANTS.keys()) == expected

    def test_all_variants_have_required_fields(self):
        for key, variant in ORDER_VARIANTS.items():
            assert isinstance(variant, OrderVariant)
            assert isinstance(variant.name, str) and len(variant.name) > 0
            assert isinstance(variant.description, str) and len(variant.description) > 0

    def test_canonical_has_no_hypothesis(self):
        assert ORDER_VARIANTS["O"].hypothesis is None

    def test_perturbation_variants_have_hypotheses(self):
        for key, variant in ORDER_VARIANTS.items():
            if key != "O":
                assert variant.hypothesis is not None, f"{key} should have a hypothesis"
                assert variant.hypothesis.startswith("H"), f"{key} hypothesis should start with H"


# ============================================================================
# Order specs coverage
# ============================================================================


class TestOrderSpecs:
    """Every variant must have specs for all 6 mode×mcp combinations."""

    ALL_MODE_MCP = [
        ("standard", False),
        ("standard", True),
        ("cot", False),
        ("cot", True),
        ("stepwise", False),
        ("stepwise", True),
    ]

    def test_all_variants_cover_all_modes(self):
        for variant_key in ORDER_VARIANTS:
            assert variant_key in _ORDER_SPECS, f"Missing specs for {variant_key}"
            for mode, mcp in self.ALL_MODE_MCP:
                assert (mode, mcp) in _ORDER_SPECS[variant_key], (
                    f"Missing spec: variant={variant_key}, mode={mode}, mcp={mcp}"
                )

    def test_all_specs_have_component_order(self):
        for variant_key, mode_specs in _ORDER_SPECS.items():
            for (mode, mcp), spec in mode_specs.items():
                assert "component_order" in spec, (
                    f"Missing component_order: {variant_key}/{mode}/{mcp}"
                )
                order = spec["component_order"]
                assert isinstance(order, tuple) and len(order) > 0

    def test_stepwise_specs_have_system_and_turns(self):
        for variant_key, mode_specs in _ORDER_SPECS.items():
            for (mode, mcp), spec in mode_specs.items():
                if mode == "stepwise":
                    assert "system_components" in spec, (
                        f"Missing system_components: {variant_key}/{mode}/{mcp}"
                    )
                    assert "user_turn_keys" in spec, (
                        f"Missing user_turn_keys: {variant_key}/{mode}/{mcp}"
                    )
                    assert len(spec["system_components"]) > 0
                    assert len(spec["user_turn_keys"]) > 0

    def test_component_keys_are_valid(self):
        """All component keys in specs resolve to COMPONENT_REGISTRY or SC."""
        valid_keys = set(COMPONENT_REGISTRY.keys()) | {"SC"}
        for variant_key, mode_specs in _ORDER_SPECS.items():
            for (mode, mcp), spec in mode_specs.items():
                for key in spec["component_order"]:
                    assert key in valid_keys, f"Invalid key {key!r} in {variant_key}/{mode}/{mcp}"

    def test_mcp_specs_include_lit(self):
        """MCP variants always include LIT in their component order."""
        for variant_key, mode_specs in _ORDER_SPECS.items():
            for (mode, mcp), spec in mode_specs.items():
                if mcp:
                    assert "LIT" in spec["component_order"], (
                        f"LIT missing from MCP spec: {variant_key}/{mode}/{mcp}"
                    )
                else:
                    assert "LIT" not in spec["component_order"], (
                        f"LIT should not be in non-MCP spec: {variant_key}/{mode}/{mcp}"
                    )

    def test_canonical_matches_phase1_routes(self):
        """Canonical order specs should match Phase 1 MODE_REGISTRY orders."""
        for route_name, route in MODE_REGISTRY.items():
            mode, mcp = route.mode, route.mcp
            canonical_spec = _ORDER_SPECS["O"][(mode, mcp)]
            assert canonical_spec["component_order"] == route.component_order, (
                f"Canonical order mismatch for route {route_name}: "
                f"spec={canonical_spec['component_order']} vs route={route.component_order}"
            )

    def test_stepwise_system_plus_turns_cover_full_order(self):
        """For stepwise specs, system_components ∪ user_turn component keys == component_order."""
        for variant_key, mode_specs in _ORDER_SPECS.items():
            for (mode, mcp), spec in mode_specs.items():
                if mode != "stepwise":
                    continue
                turn_component_keys = {key for key, _mcp_flag in spec["user_turn_keys"]}
                combined = set(spec["system_components"]) | turn_component_keys
                expected = set(spec["component_order"])
                assert combined == expected, (
                    f"Stepwise split mismatch: {variant_key}/{mode}/mcp={mcp} — "
                    f"system∪turns={sorted(combined)} vs order={sorted(expected)}"
                )

    def test_all_perturbations_use_same_component_set(self):
        """Every perturbation variant uses the same component set as canonical."""
        for variant_key, mode_specs in _ORDER_SPECS.items():
            if variant_key == "O":
                continue
            for (mode, mcp), spec in mode_specs.items():
                canonical_order = _ORDER_SPECS["O"][(mode, mcp)]["component_order"]
                assert set(spec["component_order"]) == set(canonical_order), (
                    f"Component set mismatch: {variant_key}/{mode}/mcp={mcp} — "
                    f"perturbed={sorted(spec['component_order'])} vs "
                    f"canonical={sorted(canonical_order)}"
                )


# ============================================================================
# Validation
# ============================================================================


class TestValidation:
    def test_valid_names_pass(self):
        names = ["O", "O1"]
        assert validate_order_variant_names(names) == names

    def test_invalid_names_raise(self):
        with pytest.raises(ValueError, match="Unknown order variant"):
            validate_order_variant_names(["O", "nonexistent"])

    def test_empty_list_is_valid(self):
        assert validate_order_variant_names([]) == []


# ============================================================================
# Selector (resolve_order_variant_ids)
# ============================================================================


class TestResolveOrderVariantIds:
    def test_all(self):
        ids = resolve_order_variant_ids("all")
        assert ids[0] == "O"
        assert set(ids) == set(ORDER_VARIANTS)

    def test_range(self):
        ids = resolve_order_variant_ids("O1-O3")
        assert ids == ["O", "O1", "O2", "O3"]

    def test_explicit_list_always_includes_canonical(self):
        ids = resolve_order_variant_ids(["O1", "O3"])
        assert ids == ["O", "O1", "O3"]

    def test_single_string(self):
        assert resolve_order_variant_ids("O2") == ["O", "O2"]

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown order variant"):
            resolve_order_variant_ids(["O99"])

    def test_no_duplicate_canonical(self):
        ids = resolve_order_variant_ids(["O", "O1"])
        assert ids.count("O") == 1

    def test_all_returns_full_registry(self):
        ids = resolve_order_variant_ids("all")
        assert len(ids) == len(ORDER_VARIANTS)


# ============================================================================
# apply_order_variant
# ============================================================================


class TestApplyOrderVariant:
    def test_produces_new_route_with_correct_name(self):
        base = MODE_REGISTRY["cot"]
        result = apply_order_variant(base, "O1")
        assert result.name == "cot_O1_late_screen_context"

    def test_preserves_base_fields(self):
        base = MODE_REGISTRY["single_call"]
        result = apply_order_variant(base, "O")
        assert result.mode == base.mode
        assert result.mcp == base.mcp
        assert result.delivery == base.delivery

    def test_sets_order_metadata(self):
        base = MODE_REGISTRY["cot"]
        result = apply_order_variant(base, "O3")
        assert result.base_route == "cot"
        assert result.order_variant == "early_output_format"
        assert result.order_hypothesis == "H4"

    def test_canonical_variant_has_empty_hypothesis(self):
        base = MODE_REGISTRY["single_call"]
        result = apply_order_variant(base, "O")
        assert result.order_hypothesis == ""

    def test_component_order_is_tuple(self):
        base = MODE_REGISTRY["cot"]
        result = apply_order_variant(base, "O4")
        assert isinstance(result.component_order, tuple)
        assert len(result.component_order) > 0

    def test_stepwise_route_has_user_turns(self):
        base = MODE_REGISTRY["stepwise"]
        result = apply_order_variant(base, "O1")
        assert result.delivery == "multi_turn"
        assert len(result.system_components) > 0
        assert len(result.user_turns) > 0
        assert all(isinstance(t, StepwiseTurn) for t in result.user_turns)

    def test_route_is_frozen(self):
        base = MODE_REGISTRY["single_call"]
        result = apply_order_variant(base, "O")
        with pytest.raises(AttributeError):
            result.name = "modified"

    def test_unknown_variant_raises(self):
        base = MODE_REGISTRY["single_call"]
        with pytest.raises(ValueError, match="Unknown order variant"):
            apply_order_variant(base, "nonexistent")

    def test_perturbation_changes_order(self):
        """Non-canonical variant should produce different order from canonical."""
        base = MODE_REGISTRY["cot"]
        canonical = apply_order_variant(base, "O")
        perturbed = apply_order_variant(base, "O4")
        assert canonical.component_order != perturbed.component_order

    def test_same_components_different_order(self):
        """Perturbed variant should have same components, just reordered."""
        base = MODE_REGISTRY["single_call"]
        canonical = apply_order_variant(base, "O")
        perturbed = apply_order_variant(base, "O1")
        assert set(canonical.component_order) == set(perturbed.component_order)
        assert canonical.component_order != perturbed.component_order


# ============================================================================
# build_order_benchmark_routes
# ============================================================================


class TestBuildOrderBenchmarkRoutes:
    def test_returns_correct_count(self):
        base_routes = build_routes_from_config(["single_call", "cot"])
        result = build_order_benchmark_routes(base_routes, ["O", "O1"])
        assert len(result) == 4  # 2 routes × 2 variants

    def test_all_selector_string(self):
        """'all' selector produces routes for every variant."""
        base_routes = build_routes_from_config(["single_call"])
        result = build_order_benchmark_routes(base_routes, "all")
        assert all(isinstance(r, Route) for r in result)
        assert len(result) == 5  # 1 route × 5 variants

    def test_range_selector_string(self):
        """Range selector like 'O1-O3' expands and includes canonical."""
        base_routes = build_routes_from_config(["single_call"])
        result = build_order_benchmark_routes(base_routes, "O1-O3")
        assert len(result) == 4  # O + O1 + O2 + O3

    def test_invalid_variants_raise(self):
        base_routes = build_routes_from_config(["single_call"])
        with pytest.raises(ValueError):
            build_order_benchmark_routes(base_routes, ["O", "invalid"])

    def test_all_base_routes(self):
        """Build order variants for every Phase 1 mode in the registry."""
        base_routes = build_routes_from_config(list(MODE_REGISTRY.keys()))
        result = build_order_benchmark_routes(base_routes, ["O", "O1"])
        assert len(result) == len(MODE_REGISTRY) * 2  # N routes × 2 variants
