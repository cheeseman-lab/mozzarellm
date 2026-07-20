"""Tests for order-variant prompt construction in the benchmark orchestrator.

Verifies that construct_prompts() correctly passes component_order to the
prompt factory for Phase 2 order-variant routes, and that stepwise routes
build user turns from the route's user_turns field.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.arch_bench_routes import (
    Route,
    StepwiseTurn,
    MODE_REGISTRY,
)
from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.order_bench_orderings import (
    apply_order_variant,
)
from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_orchestrator import (
    construct_prompts,
)
from mozzarellm.prompt_components import COMPONENT_REGISTRY


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_bundle_path(tmp_path):
    """Create a minimal evidence bundle JSON for testing."""
    bundle = {
        "cluster_id": "42",
        "screen_name": "test_screen",
        "genes": [
            {"gene_symbol": "GENE1", "accession": "P12345"},
            {"gene_symbol": "GENE2", "accession": "Q67890"},
        ],
    }
    path = tmp_path / "test_screen_cluster_42_bundle.json"
    import json

    path.write_text(json.dumps(bundle), encoding="utf-8")
    return path


@pytest.fixture
def mock_screen_context_path(tmp_path):
    """Create a minimal screen context JSON for testing."""
    ctx = {
        "screen_name": "test_screen",
        "lab": "Test Lab",
        "perturbation": {"type": "CRISPR", "description": "Test"},
        "readout": {"type": "imaging", "channels": ["DNA"]},
        "clustering": {"method": "leiden", "resolution": 10, "n_clusters": 100},
        "controls": {"positive": "TP53", "negative": "non-targeting"},
        "provenance": {"lab": "Test Lab", "screen_name": "test_screen"},
    }
    path = tmp_path / "test_screen_screen_context.json"
    import json

    path.write_text(json.dumps(ctx), encoding="utf-8")
    return path


# ============================================================================
# Phase 1 backward compatibility
# ============================================================================


class TestPhase1BackwardCompatibility:
    """Phase 1 routes should NOT pass component_order (preserving existing behavior)."""

    def test_phase1_route_does_not_pass_component_order(
        self, tmp_path, mock_bundle_path, mock_screen_context_path
    ):
        """Phase 1 route should use mode-based assembly (no component_order)."""
        route = MODE_REGISTRY["single_call"]
        assert route.order_variant == ""  # Phase 1

        with patch(
            "tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow"
            ".bench_orchestrator.make_cluster_analysis_system_prompt"
        ) as mock_make:
            mock_make.return_value = "mock prompt"

            construct_prompts(
                route=route,
                screen_name="test_screen",
                screen_context_path=mock_screen_context_path,
                cluster_id="42",
                bundle_path=mock_bundle_path,
                output_dir=tmp_path,
            )

            # component_order should NOT be passed for Phase 1 routes
            call_kwargs = mock_make.call_args[1]
            assert "component_order" not in call_kwargs or call_kwargs["component_order"] is None


# ============================================================================
# Phase 2 component_order passing
# ============================================================================


class TestPhase2ComponentOrder:
    """Phase 2 order-variant routes should pass component_order to the factory."""

    def test_single_call_route_passes_component_order(
        self, tmp_path, mock_bundle_path, mock_screen_context_path
    ):
        base = MODE_REGISTRY["single_call"]
        order_route = apply_order_variant(base, "O1")

        with patch(
            "tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow"
            ".bench_orchestrator.make_cluster_analysis_system_prompt"
        ) as mock_make:
            mock_make.return_value = "mock prompt"

            construct_prompts(
                route=order_route,
                screen_name="test_screen",
                screen_context_path=mock_screen_context_path,
                cluster_id="42",
                bundle_path=mock_bundle_path,
                output_dir=tmp_path,
            )

            call_kwargs = mock_make.call_args[1]
            assert call_kwargs["component_order"] is not None
            assert call_kwargs["component_order"] == list(order_route.component_order)

    def test_cot_route_passes_component_order(
        self, tmp_path, mock_bundle_path, mock_screen_context_path
    ):
        base = MODE_REGISTRY["cot"]
        order_route = apply_order_variant(base, "O3")

        with patch(
            "tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow"
            ".bench_orchestrator.make_cluster_analysis_system_prompt"
        ) as mock_make:
            mock_make.return_value = "mock prompt"

            construct_prompts(
                route=order_route,
                screen_name="test_screen",
                screen_context_path=mock_screen_context_path,
                cluster_id="42",
                bundle_path=mock_bundle_path,
                output_dir=tmp_path,
            )

            call_kwargs = mock_make.call_args[1]
            assert call_kwargs["component_order"] == list(order_route.component_order)

    def test_stepwise_route_passes_system_components(
        self, tmp_path, mock_bundle_path, mock_screen_context_path
    ):
        """Stepwise order-variant should pass system_components, not full order."""
        base = MODE_REGISTRY["stepwise"]
        order_route = apply_order_variant(base, "O1")

        with (
            patch(
                "tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow"
                ".bench_orchestrator.make_cluster_analysis_system_prompt"
            ) as mock_make,
            patch(
                "tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow"
                ".bench_orchestrator.compose_stepwise_turns_from_route"
            ) as mock_turns,
        ):
            mock_make.return_value = "mock system prompt"
            mock_turns.return_value = [{"content": "STEP 1 - test", "mcp": False}]

            result = construct_prompts(
                route=order_route,
                screen_name="test_screen",
                screen_context_path=mock_screen_context_path,
                cluster_id="42",
                bundle_path=mock_bundle_path,
                output_dir=tmp_path,
            )

            # System prompt should be built from system_components
            call_kwargs = mock_make.call_args[1]
            assert call_kwargs["component_order"] == list(order_route.system_components)
            # mode="standard" to avoid step numbering in system prompt
            assert call_kwargs["mode"] == "standard"

            # compose_stepwise_turns_from_route should have been called
            mock_turns.assert_called_once()
            assert result["stepwise_turns"] is not None

    def test_phase1_stepwise_uses_canonical_turns(
        self, tmp_path, mock_bundle_path, mock_screen_context_path
    ):
        """Phase 1 stepwise route should use compose_stepwise_user_turns (not from_route)."""
        route = MODE_REGISTRY["stepwise"]

        with (
            patch(
                "tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow"
                ".bench_orchestrator.make_cluster_analysis_system_prompt"
            ) as mock_make,
            patch(
                "tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow"
                ".bench_orchestrator.compose_stepwise_user_turns"
            ) as mock_canonical_turns,
            patch(
                "tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow"
                ".bench_orchestrator.compose_stepwise_turns_from_route"
            ) as mock_order_turns,
        ):
            mock_make.return_value = "mock prompt"
            mock_canonical_turns.return_value = [{"content": "STEP 1", "mcp": False}]

            construct_prompts(
                route=route,
                screen_name="test_screen",
                screen_context_path=mock_screen_context_path,
                cluster_id="42",
                bundle_path=mock_bundle_path,
                output_dir=tmp_path,
            )

            mock_canonical_turns.assert_called_once_with(route.mcp)
            mock_order_turns.assert_not_called()


# ============================================================================
# Component order content verification
# ============================================================================


class TestOrderVariantPromptContent:
    """Verify that different order variants produce different prompt content."""

    def test_late_screen_context_moves_sc(
        self, tmp_path, mock_bundle_path, mock_screen_context_path
    ):
        """late_screen_context should place SC after rules, not right after CAT."""
        base = MODE_REGISTRY["single_call"]
        canonical = apply_order_variant(base, "O")
        perturbed = apply_order_variant(base, "O1")

        # In canonical: SC is position 1 (after CAT)
        assert canonical.component_order[1] == "SC"
        # In late_screen_context: SC should be later
        sc_idx = list(perturbed.component_order).index("SC")
        assert sc_idx > 1

    def test_early_output_format_moves_o(self):
        """early_output_format should place O near the start."""
        base = MODE_REGISTRY["single_call"]
        canonical = apply_order_variant(base, "O")
        perturbed = apply_order_variant(base, "O3")

        canonical_o_idx = list(canonical.component_order).index("O")
        perturbed_o_idx = list(perturbed.component_order).index("O")
        assert perturbed_o_idx < canonical_o_idx

    def test_delayed_task_anchor_moves_cat(self):
        """delayed_task_anchor should move CAT away from position 0."""
        base = MODE_REGISTRY["cot"]
        canonical = apply_order_variant(base, "O")
        perturbed = apply_order_variant(base, "O4")

        assert canonical.component_order[0] == "CAT"
        assert perturbed.component_order[0] != "CAT"
