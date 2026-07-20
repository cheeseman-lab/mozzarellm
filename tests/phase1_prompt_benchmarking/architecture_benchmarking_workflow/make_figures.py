"""Render the evidence-source comparison figure for the benchmarking gate.

Scores the uniprot / affinage / both cached runs against reviewer-consensus
ground truth and draws the grouped bar chart (Category, Novel subclass,
Unchar subclass, Coherence) that decides which evidence source feeds the
downstream prompt.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

from architecture_benchmarking_workflow.scorer import MetricPanel, score_run
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
FALLBACK_COLORS = ("#E69F00", "#0072B2", "#009E73", "#CC79A7", "#D55E00")


def evidence_panel(
    gt: dict, run_dirs: dict[str, Path], cluster_coherence: dict
) -> dict[str, MetricPanel]:
    """Score each evidence source's cached run dir against consensus ground truth."""
    return {
        source: score_run(run_dir, gt, cluster_coherence=cluster_coherence)
        for source, run_dir in run_dirs.items()
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


def figure_source_mcp_panel(
    state_path: Path,
    out_png: Path,
    ceiling: tuple[float, float] = (0.80, 0.86),
) -> None:
    """Figure 1: source x MCP panel from a pipeline_state.json stage-1 block.

    Six bars per metric (uniprot/affinage/both x single_call/+mcp); MCP cells are
    hatched. Shows the headline story: uniprot-alone is the category floor, MCP
    lifts it, affinage is already at the ceiling without MCP.
    """
    _setup_style()
    cells = {
        k: _panel_from_json(v)
        for k, v in json.loads(Path(state_path).read_text())["stage1"]["cells"].items()
    }
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
    fig, ax = plt.subplots(figsize=(11, 6))

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
        color = SOURCE_COLORS.get(source, "#666666")
        offset = (i - (len(order) - 1) / 2) * width
        values = [metric_value(cells[key], m) * 100 for m in METRICS]
        label = f"{source}{'+mcp' if is_mcp else ''}"
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=color,
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
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Agreement with consensus (%)", fontsize=12)
    ax.set_title(
        "Figure 1 — Evidence source × MCP (W0, Sonnet 5)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout()
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


def figure_evidence_panel(
    panels: dict[str, MetricPanel],
    out_png: Path,
    ceiling: tuple[float, float] = (0.80, 0.86),
) -> None:
    """Render a grouped bar chart of the four metrics across evidence sources."""
    _setup_style()

    sources = list(panels)
    n_sources = len(sources)
    width = 0.8 / n_sources
    x = np.arange(len(METRICS))

    fig, ax = plt.subplots(figsize=(9, 6))

    lo, hi = ceiling
    group_half = 0.8 / 2
    ax.add_patch(
        plt.Rectangle(
            (x[0] - group_half, lo * 100),
            group_half * 2,
            (hi - lo) * 100,
            facecolor="#999999",
            alpha=0.25,
            edgecolor="none",
            zorder=0,
            label=f"human inter-reviewer range ({lo * 100:.0f}–{hi * 100:.0f}%)",
        )
    )

    for i, source in enumerate(sources):
        color = SOURCE_COLORS.get(source, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])
        offset = (i - (n_sources - 1) / 2) * width
        values = [metric_value(panels[source], m) * 100 for m in METRICS]
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=source,
            color=color,
            edgecolor="white",
            linewidth=1.2,
            zorder=2,
        )
        for b in bars:
            ax.annotate(
                f"{b.get_height():.1f}",
                xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Agreement with consensus (%)", fontsize=12)
    ax.set_title(
        "Evidence source comparison: uniprot vs affinage vs both",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=n_sources + 1, frameon=False)

    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


N_REAL_GENES = 133


def figure_prompt_walkup(state_path: Path, out_png: Path, winner: str = "W0") -> None:
    """Figure 2: prompt walkup (W0..W24) with coverage made explicit.

    Raw category is confounded by coverage — verbose variants make the model omit
    genes, inflating category on a smaller denominator. Bars show category (full
    vs partial coverage colored differently); black dots show coverage-weighted
    recall (category x n/133), the honest recall-first number. Winner is outlined.
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
    """Figure 4: mode axis (single_call / cot / stepwise) x 4 metrics."""
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
    ax.set_title("Figure 4 — Mode axis on affinage + W0 (Sonnet 5)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(order), frameon=False)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
