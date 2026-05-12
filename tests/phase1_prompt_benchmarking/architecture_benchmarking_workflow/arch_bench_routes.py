"""Route library for architecture benchmarking.

Defines the six LLM-analysis routes as first-class objects. Each route carries
all metadata needed by the orchestrator to construct prompts, call the model,
and tag outputs. Routes are constructed once from config and passed around —
keeping the orchestrator DRY and making new routes trivial to add.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

    @property
    def mode_tag(self) -> str:
        return f"{self.mode}_mcp" if self.mcp else self.mode


# =============================================================================
# ROUTE REGISTRY 
# =============================================================================

ROUTE_REGISTRY: dict[str, Route] = {
    "3a": Route(
        name="3a",
        mode="standard",
        mcp=False,
        delivery="single_call",
        component_order=("CAT", "SC", "GCR", "NPR", "UPR", "PCC", "O"),
        description="Standard flat prompt, no MCP.",
    ),
    "3a_mcp": Route(
        name="3a_mcp",
        mode="standard",
        mcp=True,
        delivery="single_call",
        component_order=("CAT", "SC", "GCR", "NPR", "UPR", "PCC", "LIT", "O"),
        description="Standard flat prompt with PubMed MCP literature validation.",
    ),
    "3b": Route(
        name="3b",
        mode="cot",
        mcp=False,
        delivery="single_call",
        component_order=("CAT", "SC", "cPH", "cGCR", "cPri", "cPSC", "cVer", "cO"),
        description="Chain-of-thought numbered steps, single call, no MCP.",
    ),
    "3b_mcp": Route(
        name="3b_mcp",
        mode="cot",
        mcp=True,
        delivery="single_call",
        component_order=("CAT", "SC", "cPH", "cGCR", "cPri", "LIT", "cPSC", "cVer", "cO"),
        description="Chain-of-thought numbered steps with PubMed MCP.",
    ),
    "3c": Route(
        name="3c",
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
    "3c_mcp": Route(
        name="3c_mcp",
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
