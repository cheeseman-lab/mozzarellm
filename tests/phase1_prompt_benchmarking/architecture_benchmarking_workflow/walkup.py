"""Winner-selection primitives over metric panels.

Two rules, one per benchmark axis:
- select_winner: greedy primary + do-no-harm guards, for the PROMPT walkup
  (best wording variant that doesn't regress the other metrics vs baseline).
- select_holistic: eliminate-dominated + argmax primary, for the SOURCE and
  MODE axes (richer sources/modes legitimately trade one metric for another).
"""

from __future__ import annotations

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

    if not best_value > baseline_primary:  # qualifying is non-empty here, so best_value is set
        return "baseline"
    return best_key


def select_holistic(
    cells: dict[str, MetricPanel], primary: str, metrics: list[str]
) -> tuple[str, list[str]]:
    """Holistic pick for the source/mode axes: best `primary` among survivors.

    A cell is dominated when another is >= on every metric and strictly > on at
    least one; dominated cells drop, then the highest-`primary` survivor wins
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
