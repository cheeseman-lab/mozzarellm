"""Route definitions for prompt-architecture benchmarking.

Each Route is a frozen dataclass encoding a unique prompt-delivery configuration:
mode (standard / cot / stepwise), MCP toggle, delivery mechanism, and the ordered list of prompt components.
More details on each component can be found in the README and prompt-assembly-routes-info.md.

ROUTE_REGISTRY maps route names (single_call, single_call_mcp, cot, ...) to Route objects used by
the orchestrator for prompt construction, model dispatch, and output tagging.

Phase 2 order-benchmarking extends Route with metadata fields (base_route,
order_variant, order_hypothesis) so perturbed orderings can be traced back
to their defined baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class StepwiseTurn:
    """A single turn in a stepwise (multi-turn) route."""

    component: str
    mcp: bool = False


@dataclass(frozen=True)
class Route:
    """Immutable specification of one benchmark route."""

    name: str
    mode: Literal["standard", "cot", "stepwise"]
    mcp: bool
    delivery: Literal["single_call", "multi_turn"]
    component_order: tuple[str, ...]
    # For stepwise routes only — system prompt components vs user turns.
    system_components: tuple[str, ...] = ()
    user_turns: tuple[StepwiseTurn, ...] = ()
    description: str = ""
    # Phase 2 order benchmarking metadata (empty for Phase 1 routes).
    base_route: str = ""
    order_variant: str = ""
    order_hypothesis: str = ""

    @property
    def mode_tag(self) -> str:
        return f"{self.mode}_mcp" if self.mcp else self.mode


# =============================================================================
# ROUTE REGISTRY
# =============================================================================

ROUTE_REGISTRY: dict[str, Route] = {
    "single_call": Route(
        name="single_call",
        mode="standard",
        mcp=False,
        delivery="single_call",
        component_order=("CAT", "SC", "GCR", "NPR", "UPR", "PCC", "O"),
        description="Standard flat prompt, no MCP.",
    ),
    "single_call_mcp": Route(
        name="single_call_mcp",
        mode="standard",
        mcp=True,
        delivery="single_call",
        component_order=("CAT", "SC", "GCR", "NPR", "UPR", "PCC", "LIT", "O"),
        description="Standard flat prompt with PubMed MCP literature validation.",
    ),
    "cot": Route(
        name="cot",
        mode="cot",
        mcp=False,
        delivery="single_call",
        component_order=("CAT", "SC", "cPH", "cGCR", "cPri", "cPSC", "cVer", "cO"),
        description="Chain-of-thought numbered steps, single call, no MCP.",
    ),
    "cot_mcp": Route(
        name="cot_mcp",
        mode="cot",
        mcp=True,
        delivery="single_call",
        component_order=("CAT", "SC", "cPH", "cGCR", "cPri", "LIT", "cPSC", "cVer", "cO"),
        description="Chain-of-thought numbered steps with PubMed MCP.",
    ),
    "stepwise": Route(
        name="stepwise",
        mode="stepwise",
        mcp=False,
        delivery="multi_turn",
        component_order=("CAT", "SC", "cPH", "cGCR", "cPri", "cPSC", "cVer", "cO"),
        system_components=("CAT", "SC"),
        user_turns=(
            StepwiseTurn("cPH", mcp=False),
            StepwiseTurn("cGCR", mcp=False),
            StepwiseTurn("cPri", mcp=False),
            StepwiseTurn("cPSC", mcp=False),
            StepwiseTurn("cVer", mcp=False),
            StepwiseTurn("cO", mcp=False),
        ),
        description="Stepwise multi-turn, no MCP.",
    ),
    "stepwise_mcp": Route(
        name="stepwise_mcp",
        mode="stepwise",
        mcp=True,
        delivery="multi_turn",
        component_order=("CAT", "SC", "cPH", "cGCR", "cPri", "LIT", "cPSC", "cVer", "cO"),
        system_components=("CAT", "SC"),
        user_turns=(
            StepwiseTurn("cPH", mcp=False),
            StepwiseTurn("cGCR", mcp=False),
            StepwiseTurn("cPri", mcp=False),
            StepwiseTurn("LIT", mcp=True),
            StepwiseTurn("cPSC", mcp=False),
            StepwiseTurn("cVer", mcp=False),
            StepwiseTurn("cO", mcp=False),
        ),
        description="Stepwise multi-turn with MCP on literature turn only.",
    ),
}


def validate_route_names(names: list[str]) -> list[str]:
    """Validate that all requested route names exist in the registry.

    Returns the validated list unchanged if all are valid; raises ValueError otherwise.
    """
    invalid = [n for n in names if n not in ROUTE_REGISTRY]
    if invalid:
        raise ValueError(
            f"Unknown route(s): {invalid}. Valid routes: {sorted(ROUTE_REGISTRY.keys())}"
        )
    return names


def build_routes_from_config(route_names: list[str]) -> list[Route]:
    """Build an ordered list of Route objects from a list of route names."""
    validate_route_names(route_names)
    return [ROUTE_REGISTRY[name] for name in route_names]
