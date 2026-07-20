"""Prompt walkup driver: tune one prompt component at a time.

For each pre-registered stage, score every wording variant against the current
baseline, pick a winner by that stage's primary metric without regressing its
guard metrics, and carry the winner forward as the baseline for the next
stage. Generation and scoring are injected (generate_fn / score_fn) so the
driver is unit-testable with zero API calls.

VER's decoy-abstention criterion (did the model correctly decline to annotate
a decoy/synthetic cluster) is evaluated separately during final validation --
the MetricPanel here only covers real-cluster metrics (category, subclass,
coherence), so VER's stage in WALKUP_STAGES is scored on those alone.
"""

from __future__ import annotations

from dataclasses import dataclass

from architecture_benchmarking_workflow.scorer import MetricPanel


def metric_value(panel: MetricPanel, name: str) -> float:
    """Return a scalar for any MetricPanel field: correct/n for tuple metrics."""
    if name == "category":
        return panel.category
    correct, n = getattr(panel, name)
    return correct / n if n else 0.0


def select_winner(
    baseline: MetricPanel,
    variants: dict[str, MetricPanel],
    primary: str,
    guards: list[str],
    eps: float = 0.0,
) -> str:
    """Pick the winning variant id, or "baseline" if nothing qualifies."""
    baseline_primary = metric_value(baseline, primary)

    qualifying = {}
    for key, panel in variants.items():
        regressed = any(metric_value(panel, g) < metric_value(baseline, g) - eps for g in guards)
        if not regressed:
            qualifying[key] = panel

    if not qualifying:
        return "baseline"

    best_key = None
    best_value = None
    for key in sorted(qualifying):
        value = metric_value(qualifying[key], primary)
        if best_value is None or value > best_value:
            best_key, best_value = key, value

    if best_value is None or not (best_value > baseline_primary):
        return "baseline"
    return best_key


def select_holistic(
    cells: dict[str, MetricPanel], primary: str, metrics: list[str]
) -> tuple[str, list[str]]:
    """Holistic pick for the source/mode axes: best `primary` among survivors.

    Unlike select_winner (greedy primary + do-no-harm guards, for the prompt
    walkup), this makes no baseline assumption. A cell is dominated when another
    cell is >= on every metric in `metrics` and strictly > on at least one;
    dominated cells are dropped, then the highest-`primary` survivor wins
    (lexicographic tie-break). Returns (winner_key, sorted dominated keys).
    """

    def dominates(a: MetricPanel, b: MetricPanel) -> bool:
        ge_all = all(metric_value(a, m) >= metric_value(b, m) for m in metrics)
        gt_any = any(metric_value(a, m) > metric_value(b, m) for m in metrics)
        return ge_all and gt_any

    dominated = sorted(
        key for key, panel in cells.items() if any(dominates(o, panel) for o in cells.values())
    )
    survivors = [k for k in cells if k not in dominated]

    best_key = None
    best_value = None
    for key in sorted(survivors):
        value = metric_value(cells[key], primary)
        if best_value is None or value > best_value:
            best_key, best_value = key, value
    return best_key, dominated


@dataclass(frozen=True)
class StageSpec:
    name: str
    primary: str
    guards: tuple[str, ...]


WALKUP_STAGES = (
    StageSpec("CAT", "category", ("novel_subclass", "unchar_subclass", "coherence")),
    StageSpec("GCR", "category", ("novel_subclass", "unchar_subclass", "coherence")),
    StageSpec("NPR", "novel_subclass", ("category",)),
    StageSpec("UPR", "unchar_subclass", ("category",)),
    StageSpec("PCC", "coherence", ("category",)),
    StageSpec("VER", "category", ("coherence", "novel_subclass")),
)


@dataclass
class WalkupResult:
    stage_rows: list[dict]
    carried: list[str]
    final_panel: MetricPanel


def _print_stage_table(
    stage: StageSpec, baseline: MetricPanel, panels: dict, recommended: str
) -> None:
    print(f"\n=== Stage {stage.name} (primary={stage.primary}) ===")
    print(f"  baseline: {metric_value(baseline, stage.primary):.4f}")
    for variant in sorted(panels):
        marker = "  <== recommended" if variant == recommended else ""
        print(f"  {variant}: {metric_value(panels[variant], stage.primary):.4f}{marker}")


def run_walkup(
    stages,
    baseline_panel: MetricPanel,
    variants_for,
    generate_fn,
    score_fn,
    interactive: bool = False,
) -> WalkupResult:
    """Walk each stage in order, carrying the winning variant forward."""
    carried: list[str] = []
    current_baseline = baseline_panel
    stage_rows: list[dict] = []

    for stage in stages:
        panels = {}
        for variant in variants_for(stage):
            run_dir = generate_fn(stage, variant, carried)
            panels[variant] = score_fn(run_dir)

        baseline_before = current_baseline
        recommended = select_winner(current_baseline, panels, stage.primary, list(stage.guards))
        winner = recommended

        if interactive:
            _print_stage_table(stage, baseline_before, panels, recommended)
            response = input(
                f"[{stage.name}] accept recommended winner {recommended!r}? "
                "[Enter=accept, or type a variant id / 'baseline' to override]: "
            ).strip()
            if response:
                if response == "baseline" or response in panels:
                    winner = response
                else:
                    print(f"Unrecognized override {response!r}; keeping {recommended!r}.")

        if winner != "baseline":
            carried.append(winner)
            current_baseline = panels[winner]

        stage_rows.append(
            {
                "stage": stage.name,
                "primary": stage.primary,
                "variant_primary_values": {
                    v: metric_value(p, stage.primary) for v, p in panels.items()
                },
                "recommended": recommended,
                "winner": winner,
                "baseline_primary_before": metric_value(baseline_before, stage.primary),
                "baseline_primary_after": metric_value(current_baseline, stage.primary),
            }
        )

    return WalkupResult(stage_rows=stage_rows, carried=carried, final_panel=current_baseline)
