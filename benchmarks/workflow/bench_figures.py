"""Figures from experiment state files -- no metric re-derivation.

Each figure reads one experiment's state JSON (the single metric output) and
renders it with the shared publication style. Figures regenerate as states
land: partial states (e.g. a walkup mid-build) render the stages that exist.

Usage:
    python -m benchmarks.workflow.bench_figures [--out benchmarks/outputs/figures]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .plot_style import (
    COLORS,
    FIGSIZE,
    PALETTE,
    alpha_ramp,
    label_panels,
    legend_panel,
    save_figure,
    setup_plot_style,
)

BENCH_DIR = Path(__file__).resolve().parents[1]
OUTPUTS = BENCH_DIR / "outputs"
N_REAL_GENES = 103


def _cw(panel: dict) -> float:
    return panel["category"] * panel["n"] / N_REAL_GENES


def _load(experiment: str) -> dict | None:
    path = OUTPUTS / experiment / f"{experiment}_state.json"
    return json.loads(path.read_text()) if path.exists() else None


def fig_source(state: dict, out: Path) -> Path:
    """Source comparison: coverage-weighted recall + per-class recall per arm."""
    arms = [c.split("__")[0] for c in state["cells"]]
    colors = [COLORS.get(a, PALETTE["neutral"]) for a in arms]

    fig, axes = plt.subplots(
        1, 3, figsize=FIGSIZE["wide"], gridspec_kw={"width_ratios": [1, 1.6, 0.5]}
    )
    cw = [_cw(p) for p in state["cells"].values()]
    axes[0].bar(range(len(arms)), cw, color=colors, label=arms)
    axes[0].set_xticks([])
    axes[0].set_ylabel("Coverage-weighted category recall")
    axes[0].set_ylim(0, 1.0)

    classes = ["ESTABLISHED", "NOVEL_ROLE", "UNCHARACTERIZED"]
    width = 0.35
    for i, arm in enumerate(arms):
        per_class = state["diagnostics"][arm]["per_class"]
        vals = [per_class[c]["recall_consensus"] for c in classes]
        axes[1].bar(
            [x + i * width for x in range(len(classes))], vals, width, color=colors[i]
        )
    axes[1].set_xticks([x + width / 2 for x in range(len(classes))])
    axes[1].set_xticklabels(["Established", "Novel role", "Uncharacterized"])
    axes[1].set_ylabel("Consensus recall")
    axes[1].set_ylim(0, 1.05)

    handles = [Patch(color=c) for c in colors]
    legend_panel(axes[2], handles, arms)
    label_panels(axes[:2])
    fig.tight_layout(pad=1.5)
    return save_figure(fig, out / "source_comparison")


def fig_walkup(state: dict, out: Path) -> Path:
    """Build-up trajectory: prior vs candidates per stage, selected build carried."""
    stages = state.get("stages", [])
    fig, ax = plt.subplots(figsize=FIGSIZE["single"])
    xs, selected_cw = [], []
    for i, rec in enumerate(stages):
        prior_cw = _cw(rec["prior"])
        ax.scatter([i], [prior_cw], color=COLORS["prior"], zorder=3, label=None)
        for cid, panel in rec["candidates"].items():
            is_sel = rec.get("selected") == cid
            ax.scatter(
                [i],
                [_cw(panel)],
                color=COLORS["selected"] if is_sel else COLORS["candidate"],
                zorder=4 if is_sel else 2,
                s=60 if is_sel else 25,
            )
        chosen = rec.get("selected")
        cw = prior_cw if chosen in (None, "prior") else _cw(rec["candidates"][chosen])
        xs.append(i)
        selected_cw.append(cw)
    if xs:
        ax.plot(xs, selected_cw, color=COLORS["selected"], lw=1.5, zorder=1)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([r["stage"] for r in stages])
    ax.set_xlabel("Walkup stage (component added)")
    ax.set_ylabel("Coverage-weighted category recall")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=COLORS["prior"]),
        plt.Line2D([], [], marker="o", ls="", color=COLORS["candidate"]),
        plt.Line2D([], [], marker="o", ls="", color=COLORS["selected"]),
    ]
    ax.legend(handles, ["prior (carried build)", "candidate", "selected"], loc="best")
    fig.tight_layout(pad=1.5)
    return save_figure(fig, out / "walkup_trajectory")


def fig_mode(state: dict, out: Path) -> Path:
    """Delivery x MCP matrix: cw-recall per condition, grouped by delivery."""
    deliveries = ["single_call", "cot", "stepwise"]
    settings = ["", "_lit", "_litb"]
    fig, axes = plt.subplots(
        1, 2, figsize=FIGSIZE["double"], gridspec_kw={"width_ratios": [3, 0.6]}
    )
    ax = axes[0]
    for d_i, d in enumerate(deliveries):
        ramp = alpha_ramp(COLORS[d], len(settings))
        for s_i, s in enumerate(settings):
            cell = next(
                (p for k, p in state["cells"].items() if k.startswith(f"{d}{s}__")), None
            )
            if cell is None:
                continue
            ax.bar([d_i * len(settings) + s_i], [_cw(cell)], color=ramp[s_i])
    ax.set_xticks([i * len(settings) + 1 for i in range(len(deliveries))])
    ax.set_xticklabels(deliveries)
    ax.set_ylabel("Coverage-weighted category recall")
    ax.set_ylim(0, 1.0)
    handles = [Patch(color=PALETTE["neutral"], alpha=a) for a in (0.3, 0.65, 1.0)]
    legend_panel(axes[1], handles, ["no literature step", "LIT (validation)", "LITB (gap-fill)"])
    fig.tight_layout(pad=1.5)
    return save_figure(fig, out / "mode_matrix")


def fig_order(state: dict, out: Path) -> Path:
    """Component-order sensitivity: cw-recall per variant + the spread."""
    fig, ax = plt.subplots(figsize=FIGSIZE["single"])
    names = [k.split("__")[0] for k in state["cells"]]
    vals = [_cw(p) for p in state["cells"].values()]
    winner = state.get("winner_condition")
    colors = [COLORS["selected"] if n == winner else PALETTE["neutral"] for n in names]
    ax.bar(range(len(names)), vals, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_xlabel("Component-order variant")
    ax.set_ylabel("Coverage-weighted category recall")
    ax.set_ylim(0, 1.0)
    spread = max(vals) - min(vals) if vals else 0.0
    ax.axhline(max(vals), color="black", alpha=0.3, ls="--")
    ax.set_title(f"order spread = {spread:.3f}")
    fig.tight_layout(pad=1.5)
    return save_figure(fig, out / "order_spread")


_FIGURES = {
    "source": fig_source,
    "walkup": fig_walkup,
    "mode": fig_mode,
    "order": fig_order,
}


def render_all(out: Path) -> list[Path]:
    setup_plot_style()
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for experiment, fn in _FIGURES.items():
        state = _load(experiment)
        if state is None:
            print(f"[figures] no state for {experiment}; skipped")
            continue
        written.append(fn(state, out))
        print(f"[figures] {written[-1]}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUTPUTS / "figures")
    args = ap.parse_args()
    render_all(args.out)


if __name__ == "__main__":
    main()
