"""Benchmark figures, one per pipeline stage, rendered from pipeline_state.json.

Figure 1 = source x MCP + decoy validation (Stage 1); Figure 2 = prompt walkup
(Stage 2); Figure 3 = mode (Stage 3).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

from architecture_benchmarking_workflow.scorer import MetricPanel
from architecture_benchmarking_workflow.walkup import metric_value

METRICS = ("category", "novel_subclass", "unchar_subclass", "coherence")
METRIC_LABELS = {
    "category": "Category",
    "novel_subclass": "Novel subclass",
    "unchar_subclass": "Unchar subclass",
    "coherence": "Coherence",
}

# Colorblind-safe, distinguishable per evidence source; matches the palette
# used for the same source axis in analysis/make_figures.py.
SOURCE_COLORS = {
    "uniprot": "#2E5C8A",
    "affinage": "#B84A3E",
    "both": "#6E8B3D",
}


def _panel_from_json(d: dict) -> MetricPanel:
    return MetricPanel(
        category=d["category"],
        novel_subclass=tuple(d["novel_subclass"]),
        unchar_subclass=tuple(d["unchar_subclass"]),
        coherence=tuple(d["coherence"]),
        n=d["n"],
        failures=d["failures"],
    )


_DECOY_ROWS = [
    ("aconcagua_interphase_shuffled", "17", "shuffled/17\n(abstain: no cluster)"),
    ("whitney", "49", "whitney/49\n(abstain: controls)"),
    ("jebel", "0", "jebel/0\n(functional: 156 genes)"),
]


def _draw_source_panel(ax, cells: dict[str, MetricPanel], ceiling: tuple[float, float]) -> None:
    """Left panel of Figure 1: source × MCP metric bars (real clusters)."""
    order = [
        "uniprot::single_call",
        "uniprot::single_call_mcp",
        "affinage::single_call",
        "affinage::single_call_mcp",
        "both::single_call",
        "both::single_call_mcp",
    ]
    order = [k for k in order if k in cells]
    x = np.arange(len(METRICS))
    width = 0.8 / len(order)

    lo, hi = ceiling
    ax.add_patch(
        plt.Rectangle(
            (x[0] - 0.4, lo * 100),
            0.8,
            (hi - lo) * 100,
            facecolor="#999999",
            alpha=0.25,
            edgecolor="none",
            zorder=0,
            label=f"human inter-reviewer range ({lo * 100:.0f}–{hi * 100:.0f}%)",
        )
    )
    for i, key in enumerate(order):
        source = key.split("::")[0]
        is_mcp = key.endswith("_mcp")
        offset = (i - (len(order) - 1) / 2) * width
        values = [metric_value(cells[key], m) * 100 for m in METRICS]
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=f"{source}{'+mcp' if is_mcp else ''}",
            color=SOURCE_COLORS.get(source, "#666666"),
            edgecolor="white",
            linewidth=1.0,
            zorder=2,
            hatch="///" if is_mcp else None,
            alpha=0.85 if is_mcp else 1.0,
        )
        for b in bars:
            ax.annotate(
                f"{b.get_height():.0f}",
                xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=10)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Agreement with consensus (%)", fontsize=11)
    ax.set_title("Evidence source × MCP (W0, real clusters)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=8)


def _draw_decoy_grid(ax, decoys_by_source: dict) -> None:
    """Right panel of Figure 1: decoy pass/fail grid (abstention + robustness)."""
    sources = [s for s in ("uniprot", "affinage", "both") if s in decoys_by_source]
    n = len(_DECOY_ROWS)
    for r, (screen, cluster, _) in enumerate(_DECOY_ROWS):
        for c, source in enumerate(sources):
            rec = next(
                (
                    d
                    for d in decoys_by_source[source]
                    if (d["screen"], d["cluster"]) == (screen, cluster)
                ),
                None,
            )
            passed = bool(rec and rec["passed"])
            ax.add_patch(
                plt.Rectangle(
                    (c, n - 1 - r),
                    1,
                    1,
                    facecolor="#6E8B3D" if passed else "#B84A3E",
                    edgecolor="white",
                    linewidth=2,
                    alpha=0.9,
                )
            )
            if rec:
                fails = f"  ({rec['failures']} err)" if rec["failures"] else ""
                txt = f"{'PASS' if passed else 'FAIL'}\n{rec['modal_confidence']}{fails}"
            else:
                txt = "—"
            ax.text(
                c + 0.5,
                n - 1 - r + 0.5,
                txt,
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )
    ax.set_xlim(0, len(sources))
    ax.set_ylim(0, n)
    ax.set_xticks([c + 0.5 for c in range(len(sources))])
    ax.set_xticklabels(sources, fontsize=10)
    ax.set_yticks([n - 1 - r + 0.5 for r in range(n)])
    ax.set_yticklabels([lbl for _, _, lbl in _DECOY_ROWS], fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Decoy validation", fontsize=12, fontweight="bold")


def figure_source_and_decoys(
    state_path: Path,
    out_png: Path,
    ceiling: tuple[float, float] = (0.80, 0.86),
) -> None:
    """Figure 1 (Stage 1): source × MCP bars (real clusters) + decoy grid, one figure.

    Both panels come from the same Stage-1 run: the left drives source selection,
    the right validates the 3 decoys (abstention + robustness).
    """
    _setup_style()
    stage1 = json.loads(Path(state_path).read_text())["stage1"]
    cells = {k: _panel_from_json(v) for k, v in stage1["cells"].items()}
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [3, 2]}
    )
    _draw_source_panel(ax_left, cells, ceiling)
    _draw_decoy_grid(ax_right, stage1["decoys"])
    fig.suptitle(
        "Figure 1 — Stage 1: evidence source + decoy validation (Sonnet 5)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _setup_style() -> None:
    plt.rcdefaults()
    arial = Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf")
    if arial.exists():
        fm.fontManager.addfont(str(arial))
        for variant in arial.parent.glob("Arial*.ttf"):
            fm.fontManager.addfont(str(variant))
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["savefig.dpi"] = 300


N_REAL_GENES = 133


def figure_prompt_walkup(state_path: Path, out_png: Path, winner: str = "W0") -> None:
    """Figure 2: prompt walkup (W0..W24) with coverage made explicit.

    Category is confounded by coverage (verbose variants omit genes, inflating it
    on a smaller denominator). Bars = category (green full / orange partial); dots
    = coverage-weighted recall (category x n/133); winner outlined.
    """
    _setup_style()
    cells = {
        k: _panel_from_json(v)
        for k, v in json.loads(Path(state_path).read_text())["stage2"]["cells"].items()
    }
    wids = sorted(cells, key=lambda w: int(w[1:]))
    x = np.arange(len(wids))

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, wid in enumerate(wids):
        p = cells[wid]
        coverage = p.n / N_REAL_GENES
        full = p.n >= N_REAL_GENES
        cat = p.category * 100
        eff = p.category * coverage * 100
        ax.bar(
            i,
            cat,
            0.7,
            color="#6E8B3D" if full else "#E69F00",
            edgecolor=("black" if wid == winner else "white"),
            linewidth=(2.2 if wid == winner else 0.8),
            hatch=None if full else "///",
            zorder=2,
        )
        ax.plot(i, eff, "o", color="black", markersize=5, zorder=3)
        if not full:
            ax.annotate(f"n={p.n}", xy=(i, 3), ha="center", va="bottom", fontsize=6, rotation=90)

    ax.axhline(cells[winner].category * 100, color="#333333", ls="--", lw=1, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(wids, fontsize=8, rotation=90)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Agreement with consensus (%)", fontsize=12)
    ax.set_title(
        "Figure 2 — Prompt walkup on affinage (Sonnet 5)\n"
        "bars = category (green full-coverage / orange partial); dots = coverage-weighted recall",
        fontsize=13,
        fontweight="bold",
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#6E8B3D", label="category, full coverage (n=133)"),
        plt.Rectangle(
            (0, 0), 1, 1, color="#E69F00", hatch="///", label="category, partial coverage"
        ),
        plt.Line2D([0], [0], marker="o", color="black", ls="", label="coverage-weighted recall"),
        plt.Line2D([0], [0], color="#333333", ls="--", label=f"{winner} category (winner)"),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_mode_panel(state_path: Path, out_png: Path) -> None:
    """Figure 3: mode axis (single_call / cot / stepwise) x 4 metrics."""
    _setup_style()
    cells = {
        k: _panel_from_json(v)
        for k, v in json.loads(Path(state_path).read_text())["stage3"]["cells"].items()
    }
    order = [m for m in ("single_call", "cot", "stepwise") if m in cells]
    x = np.arange(len(METRICS))
    width = 0.8 / len(order)
    colors = {"single_call": "#2E5C8A", "cot": "#B84A3E", "stepwise": "#6E8B3D"}

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, mode in enumerate(order):
        offset = (i - (len(order) - 1) / 2) * width
        values = [metric_value(cells[mode], m) * 100 for m in METRICS]
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=mode,
            color=colors.get(mode, "#666666"),
            edgecolor="white",
            linewidth=1.2,
            zorder=2,
        )
        for b in bars:
            ax.annotate(
                f"{b.get_height():.0f}",
                xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Agreement with consensus (%)", fontsize=12)
    ax.set_title("Figure 3 — Mode axis on affinage + W0 (Sonnet 5)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(order), frameon=False)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
