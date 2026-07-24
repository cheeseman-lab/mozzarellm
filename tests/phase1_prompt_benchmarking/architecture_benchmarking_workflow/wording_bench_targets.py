"""Hypothesis-driven override-target definitions for Phase 3 wording benchmarking.

Each ``WordingOverrideTarget`` is a named, W-numbered experiment that specifies
which prompt component(s) to override together, and why. It does NOT contain the
alternate text itself (see ``wording_bench_alternates.py``)and it does NOT contain
the runner (see ``bench_orchestrator.py``).

Every non-canonical target needs an alternate-text *source*. The source is
resolved per target via ``resolve_source`` with this precedence:
    force_source (config)  >  target.source  >  default_source (config)
If none is available for a target that requires overrides, a ``ValueError`` is
raised.

Adding a new hypothesis-driven override target requires editing ONLY this file:
    1. Add a ``WordingOverrideTarget`` entry to ``WORDING_OVERRIDE_TARGET_REGISTRY``.
    2. Ensure the referenced source defines the required components
       (in ``wording_bench_alternates.py``) or none.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .arch_bench_routes import MODE_REGISTRY, Route
from .order_bench_orderings import resolve_variant_ids
from .wording_bench_alternates import WORDING_ALTERNATE_SET_REGISTRY

# =============================================================================
# Target descriptor
# =============================================================================


@dataclass(frozen=True)
class WordingOverrideTarget:
    """A named, hypothesis-driven wording experiment.

    Attributes:
        id: W-number, e.g. "W0" (canonical), "W1", ...
        name: short snake_case label used in run/route names.
        hypothesis: the scientific claim being tested.
        description: human-readable note on what is overridden.
        components: component-registry keys overridden together. Empty for canonical.
        source: default alternate-text source name (None -> use config default/force).
    """

    id: str
    name: str
    hypothesis: str
    description: str
    components: tuple[str, ...]
    source: str | None = None


# =============================================================================
# Target registry — W-numbered experiments
# =============================================================================

WORDING_OVERRIDE_TARGET_REGISTRY: dict[str, WordingOverrideTarget] = {
    "W0": WordingOverrideTarget(
        id="W0",
        name="canonical",
        hypothesis="Current prompt wording.",
        description="No prompt component overrides. Baseline run.",
        components=(),
        source=None,
    ),
    "W1": WordingOverrideTarget(
        id="W1",
        name="concise_task_framing",
        hypothesis=(
            "Removing the philosophy/framing sentence from the task prompt changes "
            "how strongly the model treats pathway identification as a lens for discovery."
        ),
        description="Override only the cluster analysis task component (CAT).",
        components=("CAT",),
        source="wording_v1",
    ),
    "W2": WordingOverrideTarget(
        id="W2",
        name="multi_pathway_task_framing",
        hypothesis=(
            "Explicitly allowing 1-3 pathways reduces over-forcing of a single dominant "
            "pathway in mixed clusters."
        ),
        description=(
            "Override the task component (CAT) with multi-pathway wording. Uses "
            "wording_v2 because it needs a different CAT text than W1."
        ),
        components=("CAT",),
        source="wording_v2",
    ),
    "W3": WordingOverrideTarget(
        id="W3",
        name="imperative_gene_classification",
        hypothesis=(
            "A more imperative decision-tree version of the gene classification rules "
            "improves adherence to ESTABLISHED / NOVEL_ROLE / UNCHARACTERIZED boundaries."
        ),
        description="Override only the gene classification rules (GCR).",
        components=("GCR",),
        source="wording_v1",
    ),
    "W4": WordingOverrideTarget(
        id="W4",
        name="terse_prioritization_rules",
        hypothesis=(
            "Terser prioritization rules improve prioritization consistency by reducing "
            "distracting prose."
        ),
        description=(
            "Override novel-role and uncharacterized prioritization rules together "
            "(NPR + UPR) because they jointly define prioritization behavior."
        ),
        components=("NPR", "UPR"),
        source="wording_v1",
    ),
    "W5": WordingOverrideTarget(
        id="W5",
        name="qualitative_confidence_rubric",
        hypothesis=(
            "Removing percentage thresholds from the confidence rubric changes confidence "
            "calibration relative to the canonical rubric."
        ),
        description="Override only the pathway confidence criteria (PCC).",
        components=("PCC",),
        source="wording_v1",
    ),
    # =========================================================================
    # Sequential build-up program (W6-W24). One target per stage arm.
    #   Stage 0 (S0): W6            — schema-only floor
    #   Stage 1 (S1): W7, W8, W9    — CAT (pathway / mechanism / guarded)
    #                                 winner: W9 cat_guarded
    #   Stage 2 (S2): W10, W11, W12 — GCR on CAT=guarded
    #                                 winner: W11 gcr_decision_tree
    #   Stage 3 (S3): W13, W14, W15 — NPR on GCR=decision_tree
    #                                 winner: W13 npr_simple
    #   Stage 4 (S4): W16, W17, W18 — UPR on NPR=simple
    #                                 winner: W16 upr_simple
    #   Stage 5 (S5): W19, W20, W21 — PCC on UPR=simple
    #                                 winner: W19 pcc_simple
    #   Stage 6 (S6): W22, W23, W24 — VER on PCC=simple
    #                                 winner: W23 ver_precondition (= FINAL)
    # =========================================================================
    "W6": WordingOverrideTarget(
        id="W6",
        name="schema_only",
        hypothesis=(
            "Schema-only floor: excise every prose component (CAT, GCR, NPR, UPR, PCC) "
            "but keep OUTPUT_FORMAT_JSON. Anchors the build-up program."
        ),
        description="Empty-string overrides for CAT/GCR/NPR/UPR/PCC; OUTPUT_FORMAT_JSON intact.",
        components=("CAT", "GCR", "NPR", "UPR", "PCC"),
        source="wording_v4_floor",
    ),
    # ---- Stage 1 — CAT framings on the floor (W7-W9) -----------------------
    "W7": WordingOverrideTarget(
        id="W7",
        name="cat_pathway",
        hypothesis="CAT only, pathway frame — does identifying the dominant pathway as the goal help?",
        description="W6 + CAT (pathway frame); wording_v4_cat_pathway.",
        components=("CAT", "GCR", "NPR", "UPR", "PCC"),
        source="wording_v4_cat_pathway",
    ),
    "W8": WordingOverrideTarget(
        id="W8",
        name="cat_mechanism",
        hypothesis="CAT only, mechanism frame — does reframing as molecular mechanism change behavior?",
        description="W6 + CAT (mechanism frame); wording_v4_cat_mechanism.",
        components=("CAT", "GCR", "NPR", "UPR", "PCC"),
        source="wording_v4_cat_mechanism",
    ),
    "W9": WordingOverrideTarget(
        id="W9",
        name="cat_guarded",
        hypothesis="CAT only, guarded frame — does noting 'cluster may be noise' reduce fabrication?",
        description="W6 + CAT (guarded frame); wording_v4_cat_guarded.",
        components=("CAT", "GCR", "NPR", "UPR", "PCC"),
        source="wording_v4_cat_guarded",
    ),
    # ---- Stage 2 — GCR framings on CAT=cat_guarded (W10-W12) ---------------
    "W10": WordingOverrideTarget(
        id="W10",
        name="stage2_gcr_simple",
        hypothesis="W9 + GCR_simple — does adding bare GCR labels improve classification on top of locked CAT?",
        description="Stage 2 arm A: cat_guarded + gcr_simple; wording_v6_stage2_gcr_simple.",
        components=("CAT", "GCR", "NPR", "UPR", "PCC"),
        source="wording_v6_stage2_gcr_simple",
    ),
    "W11": WordingOverrideTarget(
        id="W11",
        name="stage2_gcr_decision_tree",
        hypothesis="W9 + GCR_decision_tree — does the explicit gating procedure improve classification in context?",
        description="Stage 2 arm B: cat_guarded + gcr_decision_tree; wording_v6_stage2_gcr_decision_tree.",
        components=("CAT", "GCR", "NPR", "UPR", "PCC"),
        source="wording_v6_stage2_gcr_decision_tree",
    ),
    "W12": WordingOverrideTarget(
        id="W12",
        name="stage2_gcr_guarded",
        hypothesis="W9 + GCR_guarded — does the default-NOVEL guard preserve accuracy + abstention in context?",
        description="Stage 2 arm C: cat_guarded + gcr_guarded; wording_v6_stage2_gcr_guarded.",
        components=("CAT", "GCR", "NPR", "UPR", "PCC"),
        source="wording_v6_stage2_gcr_guarded",
    ),
    # ---- Stage 3 — NPR framings on GCR=gcr_decision_tree (W13-W15) ---------
    "W13": WordingOverrideTarget(
        id="W13",
        name="stage3_npr_simple",
        hypothesis="W11 stack + NPR_simple — do bare NOVEL_ROLE sub-class definitions improve subclass_match on top of GCR?",
        description="Stage 3 arm A: cat_guarded + gcr_decision_tree + npr_simple; wording_v6_stage3_npr_simple.",
        components=("CAT", "GCR", "NPR", "UPR", "PCC"),
        source="wording_v6_stage3_npr_simple",
    ),
    "W14": WordingOverrideTarget(
        id="W14",
        name="stage3_npr_decision_tree",
        hypothesis="W11 stack + NPR_decision_tree — does the ordered procedure further improve subclass_match?",
        description="Stage 3 arm B: cat_guarded + gcr_decision_tree + npr_decision_tree; wording_v6_stage3_npr_decision_tree.",
        components=("CAT", "GCR", "NPR", "UPR", "PCC"),
        source="wording_v6_stage3_npr_decision_tree",
    ),
    "W15": WordingOverrideTarget(
        id="W15",
        name="stage3_npr_guarded",
        hypothesis="W11 stack + NPR_guarded — does 'prefer weaker level when uncertain' reduce over-promotion of NOVEL_ROLE sub-classes?",
        description="Stage 3 arm C: cat_guarded + gcr_decision_tree + npr_guarded; wording_v6_stage3_npr_guarded.",
        components=("CAT", "GCR", "NPR", "UPR", "PCC"),
        source="wording_v6_stage3_npr_guarded",
    ),
    # ---- Stage 4 — UPR framings on NPR=npr_simple (W16-W18) ----------------
    "W16": WordingOverrideTarget(
        id="W16",
        name="stage4_upr_simple",
        hypothesis="W13 stack + UPR_simple — do bare UNCHARACTERIZED sub-class definitions help on top of NPR?",
        description="Stage 4 arm A: cat_guarded + gcr_decision_tree + npr_simple + upr_simple; wording_v6_stage4_upr_simple.",
        components=("CAT", "GCR", "NPR", "UPR", "PCC"),
        source="wording_v6_stage4_upr_simple",
    ),
    "W17": WordingOverrideTarget(
        id="W17",
        name="stage4_upr_decision_tree",
        hypothesis="W13 stack + UPR_decision_tree — does the ordered characterization-stage procedure help in context?",
        description="Stage 4 arm B: cat_guarded + gcr_decision_tree + npr_simple + upr_decision_tree; wording_v6_stage4_upr_decision_tree.",
        components=("CAT", "GCR", "NPR", "UPR", "PCC"),
        source="wording_v6_stage4_upr_decision_tree",
    ),
    "W18": WordingOverrideTarget(
        id="W18",
        name="stage4_upr_guarded",
        hypothesis="W13 stack + UPR_guarded — does 'prefer lower-information sub-class' reduce over-characterization in context?",
        description="Stage 4 arm C: cat_guarded + gcr_decision_tree + npr_simple + upr_guarded; wording_v6_stage4_upr_guarded.",
        components=("CAT", "GCR", "NPR", "UPR", "PCC"),
        source="wording_v6_stage4_upr_guarded",
    ),
    # ---- Stage 5 — PCC framings on UPR=upr_simple (W19-W21) ----------------
    "W19": WordingOverrideTarget(
        id="W19",
        name="stage5_pcc_simple",
        hypothesis="W16 stack + PCC_simple — does brief confidence + off-ramp guidance restore abstention?",
        description="Stage 5 arm A: full stack + pcc_simple; wording_v6_stage5_pcc_simple.",
        components=("CAT", "GCR", "NPR", "UPR", "PCC"),
        source="wording_v6_stage5_pcc_simple",
    ),
    "W20": WordingOverrideTarget(
        id="W20",
        name="stage5_pcc_detailed",
        hypothesis="W16 stack + PCC_detailed — do detailed per-level criteria help confidence calibration?",
        description="Stage 5 arm B: full stack + pcc_detailed; wording_v6_stage5_pcc_detailed.",
        components=("CAT", "GCR", "NPR", "UPR", "PCC"),
        source="wording_v6_stage5_pcc_detailed",
    ),
    "W21": WordingOverrideTarget(
        id="W21",
        name="stage5_pcc_guarded",
        hypothesis="W16 stack + PCC_guarded — does 'prefer lower confidence + no coherent pathway is positive' restore abstention without breaking coverage?",
        description="Stage 5 arm C: full stack + pcc_guarded; wording_v6_stage5_pcc_guarded.",
        components=("CAT", "GCR", "NPR", "UPR", "PCC"),
        source="wording_v6_stage5_pcc_guarded",
    ),
    # ---- Stage 6 — VER framings on PCC=pcc_simple (W22-W24) ----------------
    # W23 = the final locked optimal prompt.
    "W22": WordingOverrideTarget(
        id="W22",
        name="stage6_ver_simple",
        hypothesis="W19 stack + VER_simple — does bare verification rescue coverage on coherent clusters?",
        description="Stage 6 arm A: full W19 stack + verification text appended to PCC; wording_v6_stage6_ver_simple.",
        components=("CAT", "GCR", "NPR", "UPR", "PCC"),
        source="wording_v6_stage6_ver_simple",
    ),
    "W23": WordingOverrideTarget(
        id="W23",
        name="stage6_ver_precondition",
        hypothesis="W19 stack + VER_precondition — does conditional verification preserve abstention while rescuing coverage?",
        description="Stage 6 arm B: full W19 stack + conditional verification text; wording_v6_stage6_ver_precondition. Final optimal prompt.",
        components=("CAT", "GCR", "NPR", "UPR", "PCC"),
        source="wording_v6_stage6_ver_precondition",
    ),
    "W24": WordingOverrideTarget(
        id="W24",
        name="stage6_ver_guarded",
        hypothesis="W19 stack + VER_guarded — does prefer-leave-unclassified guard balance coverage rescue against overcrediting?",
        description="Stage 6 arm C: full W19 stack + verification with guard against overcrediting; wording_v6_stage6_ver_guarded.",
        components=("CAT", "GCR", "NPR", "UPR", "PCC"),
        source="wording_v6_stage6_ver_guarded",
    ),
}
# NOTE: a base component (e.g. GCR) override WILL NOT propogate to a CoT component (e.g. cGCR) (embedded components are fixed at import time);
# Every component needs to be overridden explicitly.

# =============================================================================
# Concrete run spec
# =============================================================================


@dataclass(frozen=True)
class WordingOverrideRun:
    """A (base_route, target, resolved-source) triple — one wording condition.

    ``component_overrides`` is the concrete ``{component_key: text}`` dict ready to
    pass to ``make_cluster_analysis_system_prompt(component_overrides=...)``.
    """

    base_route: Route
    source: str | None
    target_id: str
    target_name: str
    hypothesis: str
    components: tuple[str, ...]
    component_overrides: dict[str, str] = field(default_factory=dict, hash=False, compare=False)

    @property
    def run_route_name(self) -> str:
        """Condition label: ``{base_route}_{source_or_canonical}_{id}_{name}``."""
        src = self.source if self.source is not None else "canonical"
        return f"{self.base_route.name}_{src}_{self.target_id}_{self.target_name}"


# =============================================================================
# Source resolution
# =============================================================================


def resolve_source(
    target: WordingOverrideTarget,
    default_source: str | None = None,
    force_source: str | None = None,
) -> str | None:
    """Resolve the alternate-text source for a target.

    Precedence: force_source > target.source > default_source.
    Canonical (no components) always resolves to None.
    """
    if not target.components:
        return None  # canonical / no override

    if force_source is not None:
        return force_source

    if target.source is not None:
        return target.source

    if default_source is not None:
        return default_source

    raise ValueError(
        f"Target {target.id} requires component overrides but no source was provided. "
        "Set target.source, wording_benchmark.default_source, or "
        "wording_benchmark.force_source."
    )


# =============================================================================
# Target selection (all / list / range)
# =============================================================================


def resolve_target_ids(target_selector: str | list[str]) -> list[str]:
    """Expand a target selector into an ordered, de-duplicated list of target ids.

    Supports:
        - "all"                  -> every id in the registry (registry order)
        - "W1-W5"                -> inclusive range
        - ["W1", "W3", "W5"]     -> explicit list
        - "W1"                   -> single id

    ``W0`` (canonical) is always included and placed first.

    Delegates to :func:`~.order_bench_orderings.resolve_variant_ids`.
    """
    return resolve_variant_ids(
        target_selector,
        prefix="W",
        registry=WORDING_OVERRIDE_TARGET_REGISTRY,
        canonical_key="W0",
        label="wording target",
    )


# =============================================================================
# Builder
# =============================================================================


def build_wording_override_runs(
    base_route_names: list[str],
    target_selector: str | list[str],
    default_source: str | None = None,
    force_source: str | None = None,
) -> list[WordingOverrideRun]:
    """Build concrete wording run specs (base_route x selected_target).

    Always includes the canonical W0 baseline per base route. Does NOT generate
    combinations of targets — only the explicitly selected W-numbered targets.

    Raises ValueError if a resolved source is unknown or is missing a component
    required by its target.
    """
    target_ids = resolve_target_ids(target_selector)
    runs: list[WordingOverrideRun] = []

    for route_name in base_route_names:
        if route_name not in MODE_REGISTRY:
            raise ValueError(f"Unknown base route {route_name!r}. Known: {sorted(MODE_REGISTRY)}")
        route = MODE_REGISTRY[route_name]

        for tid in target_ids:
            target = WORDING_OVERRIDE_TARGET_REGISTRY[tid]
            source_name = resolve_source(target, default_source, force_source)

            if source_name is None:
                # Canonical / no override.
                runs.append(
                    WordingOverrideRun(
                        base_route=route,
                        source=None,
                        target_id=target.id,
                        target_name=target.name,
                        hypothesis=target.hypothesis,
                        components=target.components,
                        component_overrides={},
                    )
                )
                continue

            if source_name not in WORDING_ALTERNATE_SET_REGISTRY:
                raise ValueError(
                    f"Target {target.id}: unknown alternate source {source_name!r}. "
                    f"Known: {sorted(WORDING_ALTERNATE_SET_REGISTRY)}"
                )
            alt_set = WORDING_ALTERNATE_SET_REGISTRY[source_name]
            missing = [c for c in target.components if c not in alt_set]
            if missing:
                raise ValueError(
                    f"Target {target.id}: alternate source {source_name!r} is missing "
                    f"component(s) {missing}. Available: {sorted(alt_set)}"
                )

            overrides = {c: alt_set[c] for c in target.components}
            runs.append(
                WordingOverrideRun(
                    base_route=route,
                    source=source_name,
                    target_id=target.id,
                    target_name=target.name,
                    hypothesis=target.hypothesis,
                    components=target.components,
                    component_overrides=overrides,
                )
            )

    return runs
