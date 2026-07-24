"""Order-variant definitions for Phase 2 prompt-order sensitivity benchmarking.


Hypotheses this tests (more detail in MLLM Benchmarking Plan_2_10_26.docx --- see google drive):
    H1: CAT first anchors the task  -->  tested by ``delayed_task_anchor``
    H2: GCR before NPR/UPR          -->  tested by ``prioritization_before_classification``
    H3: SC early improves grounding  -->  tested by ``late_screen_context``
    H4: O late preserves reasoning   -->  tested by ``early_output_format``
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mozzarellm.prompt_components import COMPONENT_REGISTRY, STEP_LITERATURE_VALIDATION
from mozzarellm.utils.screen_context_utils import load_screen_context_json

from .bench_routes import Route, StepwiseTurn

# ============================================================================
# Order variant descriptor
# ============================================================================

PerturbationType = Literal[
    "canonical",
    "late_screen_context",
    "prioritization_before_classification",
    "early_output_format",
    "delayed_task_anchor",
]


@dataclass(frozen=True)
class OrderVariant:
    """Metadata for a single order perturbation."""

    name: str
    hypothesis: str | None
    description: str
    perturbation_type: PerturbationType


ORDER_VARIANTS: dict[str, OrderVariant] = {
    "O": OrderVariant(
        name="canonical",
        hypothesis=None,
        description="Dependency-respecting baseline order.",
        perturbation_type="canonical",
    ),
    "O1": OrderVariant(
        name="late_screen_context",
        hypothesis="H3",
        description=(
            "Move SC later. Tests whether delaying screen context increases "
            "the risk of generic or under-contextualized interpretations."
        ),
        perturbation_type="late_screen_context",
    ),
    "O2": OrderVariant(
        name="prioritization_before_classification",
        hypothesis="H2",
        description=(
            "Move NPR/UPR before GCR (standard) or cPri before cGCR (CoT/stepwise). "
            "Tests whether prioritization is conditional on prior gene categorization."
        ),
        perturbation_type="prioritization_before_classification",
    ),
    "O3": OrderVariant(
        name="early_output_format",
        hypothesis="H4",
        description=(
            "Move O/cO to near the start. Tests whether premature formatting "
            "bias reduces content quality."
        ),
        perturbation_type="early_output_format",
    ),
    "O4": OrderVariant(
        name="delayed_task_anchor",
        hypothesis="H1",
        description=(
            "Move CAT later, after context/rules. Tests whether losing the "
            "early task anchor degrades pathway grounding."
        ),
        perturbation_type="delayed_task_anchor",
    ),
}


# ============================================================================
# Order specifications per (variant, mode, mcp) combination
# ============================================================================

# Each entry maps to a dict with:
#   "component_order": tuple  -- full ordered component list
#   "system_components": tuple -- stepwise-only system prompt keys
#   "user_turn_keys": tuple of (component_key, mcp_flag) -- stepwise-only turns
#
# For single-call routes only "component_order" is used.
# For stepwise routes all three are used.


def _stepwise_spec(
    component_order: tuple[str, ...],
    system_components: tuple[str, ...],
    user_turn_keys: tuple[tuple[str, bool], ...],
) -> dict:
    return {
        "component_order": component_order,
        "system_components": system_components,
        "user_turn_keys": user_turn_keys,
    }


def _single_call_spec(component_order: tuple[str, ...]) -> dict:
    return {"component_order": component_order}


# fmt: off
_ORDER_SPECS: dict[str, dict[tuple[str, bool], dict]] = {
    # ------------------------------------------------------------------
    # O: CANONICAL
    # ------------------------------------------------------------------
    "O": {
        ("standard", False): _single_call_spec(("CAT", "SC", "GCR", "NPR", "UPR", "PCC", "O")),
        ("standard", True):  _single_call_spec(("CAT", "SC", "GCR", "NPR", "UPR", "PCC", "LIT", "O")),
        ("cot", False):      _single_call_spec(("CAT", "SC", "cPH", "cGCR", "cPri", "cPSC", "cVer", "cO")),
        ("cot", True):       _single_call_spec(("CAT", "SC", "cPH", "cGCR", "cPri", "LIT", "cPSC", "cVer", "cO")),
        ("stepwise", False): _stepwise_spec(
            component_order=("CAT", "SC", "cPH", "cGCR", "cPri", "cPSC", "cVer", "cO"),
            system_components=("CAT", "SC"),
            user_turn_keys=(("cPH", False), ("cGCR", False), ("cPri", False), ("cPSC", False), ("cVer", False), ("cO", False)),
        ),
        ("stepwise", True):  _stepwise_spec(
            component_order=("CAT", "SC", "cPH", "cGCR", "cPri", "LIT", "cPSC", "cVer", "cO"),
            system_components=("CAT", "SC"),
            user_turn_keys=(("cPH", False), ("cGCR", False), ("cPri", False), ("LIT", True), ("cPSC", False), ("cVer", False), ("cO", False)),
        ),
    },

    # ------------------------------------------------------------------
    # O1: Late Screen Context  (H3)
    # ------------------------------------------------------------------
    "O1": {
        ("standard", False): _single_call_spec(("CAT", "GCR", "NPR", "UPR", "PCC", "SC", "O")),
        ("standard", True):  _single_call_spec(("CAT", "GCR", "NPR", "UPR", "PCC", "LIT", "SC", "O")),
        ("cot", False):      _single_call_spec(("CAT", "cPH", "cGCR", "cPri", "cPSC", "SC", "cVer", "cO")),
        ("cot", True):       _single_call_spec(("CAT", "cPH", "cGCR", "cPri", "LIT", "cPSC", "SC", "cVer", "cO")),
        ("stepwise", False): _stepwise_spec(
            component_order=("CAT", "cPH", "cGCR", "cPri", "cPSC", "SC", "cVer", "cO"),
            system_components=("CAT",),
            user_turn_keys=(("cPH", False), ("cGCR", False), ("cPri", False), ("cPSC", False), ("SC", False), ("cVer", False), ("cO", False)),
        ),
        ("stepwise", True):  _stepwise_spec(
            component_order=("CAT", "cPH", "cGCR", "cPri", "LIT", "cPSC", "SC", "cVer", "cO"),
            system_components=("CAT",),
            user_turn_keys=(("cPH", False), ("cGCR", False), ("cPri", False), ("LIT", True), ("cPSC", False), ("SC", False), ("cVer", False), ("cO", False)),
        ),
    },

    # ------------------------------------------------------------------
    # O2: Prioritization Before Classification  (H2)
    # ------------------------------------------------------------------
    "O2": {
        ("standard", False): _single_call_spec(("CAT", "SC", "NPR", "UPR", "GCR", "PCC", "O")),
        ("standard", True):  _single_call_spec(("CAT", "SC", "NPR", "UPR", "GCR", "PCC", "LIT", "O")),
        ("cot", False):      _single_call_spec(("CAT", "SC", "cPH", "cPri", "cGCR", "cPSC", "cVer", "cO")),
        ("cot", True):       _single_call_spec(("CAT", "SC", "cPH", "cPri", "cGCR", "LIT", "cPSC", "cVer", "cO")),
        ("stepwise", False): _stepwise_spec(
            component_order=("CAT", "SC", "cPH", "cPri", "cGCR", "cPSC", "cVer", "cO"),
            system_components=("CAT", "SC"),
            user_turn_keys=(("cPH", False), ("cPri", False), ("cGCR", False), ("cPSC", False), ("cVer", False), ("cO", False)),
        ),
        ("stepwise", True):  _stepwise_spec(
            component_order=("CAT", "SC", "cPH", "cPri", "cGCR", "LIT", "cPSC", "cVer", "cO"),
            system_components=("CAT", "SC"),
            user_turn_keys=(("cPH", False), ("cPri", False), ("cGCR", False), ("LIT", True), ("cPSC", False), ("cVer", False), ("cO", False)),
        ),
    },

    # ------------------------------------------------------------------
    # O3: Early Output Format  (H4)
    # ------------------------------------------------------------------
    "O3": {
        ("standard", False): _single_call_spec(("CAT", "O", "SC", "GCR", "NPR", "UPR", "PCC")),
        ("standard", True):  _single_call_spec(("CAT", "O", "SC", "GCR", "NPR", "UPR", "PCC", "LIT")),
        ("cot", False):      _single_call_spec(("CAT", "cO", "SC", "cPH", "cGCR", "cPri", "cPSC", "cVer")),
        ("cot", True):       _single_call_spec(("CAT", "cO", "SC", "cPH", "cGCR", "cPri", "LIT", "cPSC", "cVer")),
        ("stepwise", False): _stepwise_spec(
            component_order=("CAT", "SC", "cO", "cPH", "cGCR", "cPri", "cPSC", "cVer"),
            system_components=("CAT", "SC"),
            user_turn_keys=(("cO", False), ("cPH", False), ("cGCR", False), ("cPri", False), ("cPSC", False), ("cVer", False)),
        ),
        ("stepwise", True):  _stepwise_spec(
            component_order=("CAT", "SC", "cO", "cPH", "cGCR", "cPri", "LIT", "cPSC", "cVer"),
            system_components=("CAT", "SC"),
            user_turn_keys=(("cO", False), ("cPH", False), ("cGCR", False), ("cPri", False), ("LIT", True), ("cPSC", False), ("cVer", False)),
        ),
    },

    # ------------------------------------------------------------------
    # O4: Delayed Task Anchor  (H1)
    # ------------------------------------------------------------------
    "O4": {
        ("standard", False): _single_call_spec(("SC", "GCR", "NPR", "UPR", "PCC", "CAT", "O")),
        ("standard", True):  _single_call_spec(("SC", "GCR", "NPR", "UPR", "PCC", "LIT", "CAT", "O")),
        ("cot", False):      _single_call_spec(("SC", "cPH", "cGCR", "cPri", "cPSC", "CAT", "cVer", "cO")),
        ("cot", True):       _single_call_spec(("SC", "cPH", "cGCR", "cPri", "LIT", "cPSC", "CAT", "cVer", "cO")),
        ("stepwise", False): _stepwise_spec(
            component_order=("SC", "cPH", "cGCR", "cPri", "cPSC", "CAT", "cVer", "cO"),
            system_components=("SC",),
            user_turn_keys=(("cPH", False), ("cGCR", False), ("cPri", False), ("cPSC", False), ("CAT", False), ("cVer", False), ("cO", False)),
        ),
        ("stepwise", True):  _stepwise_spec(
            component_order=("SC", "cPH", "cGCR", "cPri", "LIT", "cPSC", "CAT", "cVer", "cO"),
            system_components=("SC",),
            user_turn_keys=(("cPH", False), ("cGCR", False), ("cPri", False), ("LIT", True), ("cPSC", False), ("CAT", False), ("cVer", False), ("cO", False)),
        ),
    },
}
# fmt: on


# ============================================================================
# Validation
# ============================================================================


def validate_order_variant_names(names: list[str]) -> list[str]:
    """Validate that all requested variant names exist.

    Returns the list unchanged if valid; raises ValueError otherwise.
    """
    invalid = [n for n in names if n not in ORDER_VARIANTS]
    if invalid:
        raise ValueError(
            f"Unknown order variant(s): {invalid}. Valid variants: {sorted(ORDER_VARIANTS.keys())}"
        )
    return names


# ============================================================================
# Generic variant selector (shared across benchmark axes)
# ============================================================================


def resolve_variant_ids(
    selector: str | list[str],
    *,
    prefix: str,
    registry: dict[str, object],
    canonical_key: str,
    label: str = "variant",
) -> list[str]:
    """Expand a variant selector into an ordered, de-duplicated list of ids.

    Supports:
        - ``"all"``             -> every key in *registry* (insertion order)
        - ``"{prefix}1-{prefix}4"`` -> inclusive numeric range
        - ``["{prefix}1", "{prefix}3"]`` -> explicit list
        - ``"{prefix}2"``       -> single id

    *canonical_key* is always included first.
    """
    range_re = re.compile(rf"^{re.escape(prefix)}(\d+)-{re.escape(prefix)}(\d+)$")

    if selector == "all":
        ids = list(registry.keys())
    elif isinstance(selector, str):
        m = range_re.match(selector.strip())
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if hi < lo:
                raise ValueError(f"Invalid {label} range (hi < lo): {selector!r}")
            ids = [f"{prefix}{i}" for i in range(lo, hi + 1)]
        else:
            ids = [selector.strip()]
    else:
        ids = [str(t).strip() for t in selector]

    unknown = [t for t in ids if t not in registry]
    if unknown:
        raise ValueError(f"Unknown {label} id(s): {unknown}. Known: {sorted(registry)}")

    ordered = [canonical_key] + [t for t in ids if t != canonical_key]
    seen: set[str] = set()
    result: list[str] = []
    for t in ordered:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def resolve_order_variant_ids(selector: str | list[str]) -> list[str]:
    """Expand an order-variant selector into a de-duplicated list of O-keys.

    Always includes ``"O"`` (canonical) first.  See :func:`resolve_variant_ids`
    for supported selector formats.
    """
    return resolve_variant_ids(
        selector,
        prefix="O",
        registry=ORDER_VARIANTS,
        canonical_key="O",
        label="order variant",
    )


# ============================================================================
# Apply variant to a Route
# ============================================================================


def apply_order_variant(route: Route, variant_key: str) -> Route:
    """Create a new Route with the specified order variant applied.

    Args:
        route: Base Route to apply the variant to.
        variant_key: Registry key (e.g. "O", "O1", "O2", "O3", "O4").

    Returns a frozen Route whose ``component_order``, ``system_components``,
    ``user_turns``, and order-metadata fields are set to the variant's spec.
    The ``mode``, ``mcp``, and ``delivery`` are preserved from the base route.
    """
    if variant_key not in ORDER_VARIANTS:
        raise ValueError(
            f"Unknown order variant: {variant_key!r}. "
            f"Valid variants: {sorted(ORDER_VARIANTS.keys())}"
        )

    variant = ORDER_VARIANTS[variant_key]
    spec_key = (route.mode, route.mcp)

    if spec_key not in _ORDER_SPECS[variant_key]:
        raise ValueError(
            f"No order spec for variant={variant_key!r}, mode={route.mode!r}, mcp={route.mcp}"
        )

    spec = _ORDER_SPECS[variant_key][spec_key]
    new_component_order = spec["component_order"]

    # Build stepwise fields if present
    new_system_components: tuple[str, ...] = ()
    new_user_turns: tuple[StepwiseTurn, ...] = ()

    if route.delivery == "multi_turn":
        new_system_components = spec.get("system_components", ())
        turn_keys = spec.get("user_turn_keys", ())
        new_user_turns = tuple(
            StepwiseTurn(component=comp, mcp=mcp_flag) for comp, mcp_flag in turn_keys
        )

    return Route(
        name=f"{route.name}_{variant_key}_{variant.name}",
        mode=route.mode,
        mcp=route.mcp,
        delivery=route.delivery,
        component_order=new_component_order,
        system_components=new_system_components,
        user_turns=new_user_turns,
        description=f"{route.description} [order variant: {variant_key} {variant.name}]",
        base_route=route.name,
        order_variant=variant.name,
        order_hypothesis=variant.hypothesis or "",
    )


def build_order_benchmark_routes(
    base_routes: list[Route],
    order_variants: str | list[str],
) -> list[Route]:
    """Build all order-variant routes from the given base routes and variant selector.

    *order_variants* supports the same selector syntax as wording targets:
    ``"all"``, ``"O1-O3"``, ``["O", "O1"]``, or a single ``"O2"``.
    Canonical ``"O"`` is always included.

    Returns a flat list of Route objects: one per (base_route, variant) pair.
    """
    variant_keys = resolve_order_variant_ids(order_variants)
    routes: list[Route] = []
    for base in base_routes:
        for key in variant_keys:
            routes.append(apply_order_variant(base, key))
    return routes


# ============================================================================
# Stepwise turn builder for order-variant routes
# ============================================================================


def compose_stepwise_turns_from_route(
    route: Route,
    screen_context_path: Path,
) -> list[dict]:
    """Build stepwise user turns from a Route's ``user_turns`` field.

    Resolves each ``StepwiseTurn.component`` to its text via
    ``COMPONENT_REGISTRY``, with special handling for ``SC`` (screen context)
    and ``LIT`` (literature validation).

    Returns the same ``[{"content": str, "mcp": bool}, ...]`` format as
    ``compose_stepwise_user_turns``.
    """
    ctx_obj = load_screen_context_json(screen_context_path)
    screen_context_text = json.dumps(ctx_obj, ensure_ascii=False)

    registry = dict(COMPONENT_REGISTRY)
    # LIT is now in the registry, but ensure it here for safety
    if "LIT" not in registry:
        registry["LIT"] = STEP_LITERATURE_VALIDATION

    turns: list[dict] = []
    for i, turn in enumerate(route.user_turns):
        if turn.component == "SC":
            content = "The following experimental context is provided: " + screen_context_text
        elif turn.component in registry:
            content = registry[turn.component]
        else:
            raise ValueError(
                f"Unknown component in user_turns: {turn.component!r}. "
                f"Valid keys: {sorted(registry.keys())} + 'SC'"
            )

        turns.append(
            {
                "content": f"STEP {i + 1} - {content}",
                "mcp": turn.mcp,
            }
        )

    return turns
