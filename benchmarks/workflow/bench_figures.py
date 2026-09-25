"""Figures from experiment state files -- no metric re-derivation.

Each data figure reads one experiment's state JSON (the single metric output),
flattens it to a tidy table, and renders that table with the shared publication
style. The tidy table is written next to the figure as CSV so a later re-render
can start from the plotted values without parsing the state schema. The two
schematics (pipeline, benchmark design) are drawn from constants plus the
benchmark facts the states already carry.

Usage:
    python -m benchmarks.workflow.bench_figures [--out benchmarks/outputs/figures]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from .plot_style import (
    CATEGORICAL,
    COLORS,
    PALETTE,
    alpha_ramp,
    label_panels,
    legend_panel,
    save_figure,
    setup_plot_style,
)

BENCH_DIR = Path(__file__).resolve().parents[1]
OUTPUTS = BENCH_DIR / "outputs"

# The five selection criteria in panel order, with the label the figures print.
CRITERIA = [
    ("category", "Category\nrecall"),
    ("novel_subclass", "Novel-role\nsub-class"),
    ("unchar_subclass", "Uncharacterized\nsub-class"),
    ("coherence", "Pathway\ncoherence"),
    ("controls", "Controls\npassed"),
]
CLASSES = [
    ("ESTABLISHED", "Established"),
    ("NOVEL_ROLE", "Novel role"),
    ("UNCHARACTERIZED", "Uncharacterized"),
]
DELIVERIES = ["single_call", "cot", "stepwise"]
LIT_SETTINGS = [
    ("", "no literature step"),
    ("_lit", "LIT (validation)"),
    ("_litb", "LITB (gap-fill)"),
]
COMPONENTS = [
    ("CAT", "Cluster analysis task"),
    ("GCR", "Gene classification rules"),
    ("NPR", "Novel-role sub-classes"),
    ("UPR", "Uncharacterized sub-classes"),
    ("PCC", "Pathway confidence"),
]


# --- tidy table -------------------------------------------------------------


def criterion_value(
    panel: dict, key: str, controls: str | list | None = None
) -> tuple[float, int, int]:
    """Return (fraction, numerator, denominator) for one criterion of one metrics panel.

    ``category`` is a fraction over ``n`` genes; the sub-class and coherence
    criteria are ``[correct, total]`` pairs; ``controls`` comes from the decoy
    record (``"4/4"`` in walkup states, a list of per-decoy dicts elsewhere).
    """
    if key == "category":
        n = int(panel["n"])
        return float(panel["category"]), round(panel["category"] * n), n
    if key == "controls":
        if isinstance(controls, str):
            k, n = (int(x) for x in controls.split("/"))
        elif controls:
            k, n = sum(1 for d in controls if d["passed"]), len(controls)
        else:
            return float("nan"), 0, 0
        return (k / n if n else float("nan")), k, n
    k, n = panel[key]
    return (k / n if n else float("nan")), int(k), int(n)


def _row(figure: str, panel: str, condition: str, role: str, key: str, triple, **extra) -> dict:
    value, num, den = triple
    return {
        "figure": figure,
        "panel": panel,
        "condition": condition,
        "role": role,
        "metric": key,
        "value": value,
        "numerator": num,
        "denominator": den,
        **extra,
    }


def rows_from_cells(
    figure: str, state: dict, condition_of=lambda c: c.split("__")[0]
) -> list[dict]:
    """Tidy rows for a cells-style state (source / mode / order / features)."""
    rows = []
    winner = state.get("winner_condition")
    dominated = {c.split("__")[0] for c in state.get("dominated", [])}
    for cell, panel in state["cells"].items():
        cond = condition_of(cell)
        role = "winner" if cond == winner else ("dominated" if cond in dominated else "candidate")
        for key, _ in CRITERIA:
            rows.append(
                _row(
                    figure,
                    "criteria",
                    cond,
                    role,
                    key,
                    criterion_value(panel, key, state.get("decoys", {}).get(cond)),
                )
            )
        diag = state.get("diagnostics", {}).get(cond)
        if diag:
            for cls, _ in CLASSES:
                pc = diag["per_class"][cls]
                rows.append(
                    _row(
                        figure,
                        "per_class",
                        cond,
                        role,
                        f"recall_{cls.lower()}",
                        (pc["recall_consensus"], pc["tp"], pc["n"]),
                    )
                )
    return rows


def rows_from_walkup(state: dict) -> list[dict]:
    """Tidy rows for the staged walkup: prior, candidates, selected, per stage and criterion."""
    rows = []
    for rec in state.get("stages", []):
        stage, goal = rec["stage"], rec.get("goal")
        entries = [("prior", "prior", rec["prior"], rec.get("prior_abstain"))]
        for cid, panel in rec.get("candidates", {}).items():
            role = "selected" if rec.get("selected") == cid else "candidate"
            entries.append((cid, role, panel, rec.get("candidate_abstain", {}).get(cid)))
        for cond, role, panel, abstain in entries:
            for key, _ in CRITERIA:
                rows.append(
                    _row(
                        "walkup",
                        stage,
                        cond,
                        role,
                        key,
                        criterion_value(panel, key, abstain),
                        goal=(key == goal),
                    )
                )
    return rows


def write_rows(rows: list[dict], path: Path) -> Path:
    fields = [
        "figure",
        "panel",
        "condition",
        "role",
        "metric",
        "value",
        "numerator",
        "denominator",
        "goal",
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**{"goal": ""}, **r})
    return path


def _get(rows: list[dict], **match) -> list[dict]:
    return [r for r in rows if all(r.get(k) == v for k, v in match.items())]


# --- shared panels ----------------------------------------------------------


def _bars_by_criterion(axes, rows: list[dict], conditions: list[str], colors: dict, ylim=None):
    """One panel per criterion; a bar per condition.

    Without ``ylim`` the y-axis is truncated to the decade below the smallest bar
    (the differences are the point; the CSV carries the full values).
    """
    for ax, (key, label) in zip(axes, CRITERIA, strict=True):
        vals = []
        for i, cond in enumerate(conditions):
            r = _get(rows, condition=cond, metric=key)
            v = r[0]["value"] if r else float("nan")
            vals.append(v)
            ax.bar(i, v, color=colors[cond], width=0.7)
        ax.set_xticks([])
        finite = [v for v in vals if v == v]
        lo = (
            0.0
            if ylim is None and not finite
            else (ylim[0] if ylim else min(0.5, (min(finite) // 0.1) * 0.1))
        )
        ax.set_ylim(lo, 1.02 if ylim is None else ylim[1])
        ax.set_title(label, fontsize=plt.rcParams["axes.labelsize"], fontweight="normal")
        if ax is axes[0]:
            ax.set_ylabel("Fraction")


# --- figures ----------------------------------------------------------------


def fig_source(state: dict, out: Path) -> Path:
    """Source comparison on the blank floor: five criteria + per-class recall per arm."""
    rows = rows_from_cells("source", state)
    write_rows(rows, out / "fig3_source.csv")
    arms = list(dict.fromkeys(r["condition"] for r in rows))
    colors = {a: COLORS.get(a, PALETTE["neutral"]) for a in arms}

    fig = plt.figure(figsize=(14, 4.2))
    gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 1, 1, 1.8, 0.7], wspace=0.35)
    crit_axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    _bars_by_criterion(crit_axes, rows, arms, colors, ylim=(0, 1.02))

    ax = fig.add_subplot(gs[0, 5])
    width = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        vals = [
            _get(rows, condition=arm, metric=f"recall_{c.lower()}")[0]["value"] for c, _ in CLASSES
        ]
        ax.bar([x + i * width for x in range(len(CLASSES))], vals, width, color=colors[arm])
    ax.set_xticks([x + width * (len(arms) - 1) / 2 for x in range(len(CLASSES))])
    ax.set_xticklabels([lbl for _, lbl in CLASSES], rotation=20, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_title("Per-class\nrecall", fontsize=plt.rcParams["axes.labelsize"], fontweight="normal")

    legend_panel(fig.add_subplot(gs[0, 6]), [Patch(color=colors[a]) for a in arms], arms)
    label_panels([crit_axes[0], ax], labels="AB")
    return save_figure(fig, out / "fig3_source")


def fig_walkup(state: dict, out: Path) -> Path:
    """Criteria x stage small multiples: what each stage moved, and what it held.

    Rows are the five criteria, columns the walkup stages. Each cell shows the
    prior (carried build), the stage's candidates, and the selected candidate;
    the stage's goal criterion is outlined. The band around the prior is one
    count (one gene, one cluster, one control) -- the resolution of the metric.
    """
    rows = rows_from_walkup(state)
    write_rows(rows, out / "fig4_walkup.csv")
    stages = [rec["stage"] for rec in state.get("stages", [])]
    if not stages:
        raise ValueError("walkup state has no stages")

    fig, axes = plt.subplots(
        len(CRITERIA),
        len(stages),
        figsize=(2.2 * len(stages) + 1.5, 1.9 * len(CRITERIA)),
        sharex=True,
        squeeze=False,
    )
    for r_i, (key, label) in enumerate(CRITERIA):
        for c_i, stage in enumerate(stages):
            ax = axes[r_i][c_i]
            cell = _get(rows, panel=stage, metric=key)
            prior = next(r for r in cell if r["role"] == "prior")
            den = prior["denominator"] or 1
            ax.axhspan(
                prior["value"] - 1 / den,
                prior["value"] + 1 / den,
                color=PALETTE["neutral"],
                alpha=0.35,
                lw=0,
            )
            ax.axhline(prior["value"], color=COLORS["prior"], lw=1.2)
            cands = [r for r in cell if r["role"] in ("candidate", "selected")]
            xs = [i - (len(cands) - 1) / 2 for i in range(len(cands))]
            for x, r in zip(xs, cands, strict=True):
                sel = r["role"] == "selected"
                ax.scatter(
                    [x * 0.25],
                    [r["value"]],
                    s=70 if sel else 28,
                    color=COLORS["selected"] if sel else COLORS["candidate"],
                    zorder=4 if sel else 3,
                    edgecolor="white",
                    lw=0.8,
                )
            ax.set_xlim(-0.6, 0.6)
            ax.set_xticks([])
            ax.set_ylim(*_ylim_for(key, _get(rows, metric=key)))
            if prior["goal"]:
                for s in ax.spines.values():
                    s.set_visible(True)
                    s.set_edgecolor(PALETTE["key"])
                    s.set_linewidth(2)
            if c_i == 0:
                ax.set_ylabel(label, fontsize=plt.rcParams["xtick.labelsize"])
            else:
                ax.set_yticklabels([])
            if r_i == 0:
                ax.set_title(
                    f"+{stage}", fontsize=plt.rcParams["axes.labelsize"], fontweight="bold"
                )
    handles = [
        plt.Line2D([], [], color=COLORS["prior"], lw=1.2),
        Patch(color=PALETTE["neutral"], alpha=0.35),
        plt.Line2D([], [], marker="o", ls="", color=COLORS["candidate"]),
        plt.Line2D([], [], marker="o", ls="", color=COLORS["selected"], markersize=9),
        Patch(facecolor="none", edgecolor=PALETTE["key"], lw=2),
    ]
    fig.legend(
        handles,
        ["prior (carried build)", "one-count band", "candidate", "selected", "stage goal"],
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(pad=1.2, rect=(0, 0.04, 1, 1))
    return save_figure(fig, out / "fig4_walkup")


def _ylim_for(key: str, cell: list[dict]) -> tuple[float, float]:
    vals = [r["value"] for r in cell if r["value"] == r["value"]]
    if key in ("controls", "coherence", "unchar_subclass"):
        return (-0.05, 1.05)
    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * 0.3, 0.03)
    return (max(lo - pad, 0), min(hi + pad, 1.02))


def fig_mode_order(mode: dict | None, order: dict | None, out: Path) -> Path:
    """Delivery x literature matrix (top) and component-order permutations (bottom), all criteria."""
    n_rows = int(mode is not None) + int(order is not None)
    fig = plt.figure(figsize=(14, 3.6 * n_rows))
    gs = fig.add_gridspec(n_rows, 6, width_ratios=[1, 1, 1, 1, 1, 0.9], wspace=0.3, hspace=0.6)
    first_axes = []
    r = 0
    if mode is not None:
        rows = rows_from_cells("mode", mode)
        write_rows(rows, out / "fig5_mode.csv")
        conds = [f"{d}{s}" for d in DELIVERIES for s, _ in LIT_SETTINGS]
        conds = [c for c in conds if _get(rows, condition=c)]
        colors = {}
        for d in DELIVERIES:
            ramp = alpha_ramp(COLORS[d], len(LIT_SETTINGS))
            for (s, _), col in zip(LIT_SETTINGS, ramp, strict=True):
                colors[f"{d}{s}"] = col
        axes = [fig.add_subplot(gs[r, i]) for i in range(5)]
        _bars_by_criterion(axes, rows, conds, colors)
        _mark_winner(axes, conds, mode.get("winner_condition"))
        for ax in axes:
            ax.set_xticks([1, 4, 7][: len(DELIVERIES)])
            ax.set_xticklabels(
                ["single", "cot", "stepwise"], fontsize=plt.rcParams["xtick.labelsize"] - 2
            )
        handles = [Patch(color=PALETTE["neutral_dark"], alpha=a) for a in (0.3, 0.65, 1.0)]
        legend_panel(
            fig.add_subplot(gs[r, 5]), handles, [lbl for _, lbl in LIT_SETTINGS], loc="center left"
        )
        first_axes.append(axes[0])
        r += 1
    if order is not None:
        rows = rows_from_cells("order", order)
        write_rows(rows, out / "fig5_order.csv")
        conds = list(dict.fromkeys(rr["condition"] for rr in rows))
        winner = order.get("winner_condition")
        colors = {c: (COLORS["selected"] if c == winner else PALETTE["neutral"]) for c in conds}
        axes = [fig.add_subplot(gs[r, i]) for i in range(5)]
        _bars_by_criterion(axes, rows, conds, colors)
        for ax in axes:
            ax.set_xticks(range(len(conds)))
            ax.set_xticklabels(conds)
        gate = order.get("human_gate")
        handles = [Patch(color=COLORS["selected"]), Patch(color=PALETTE["neutral"])]
        labels = ["selected order", "permutation"]
        if gate:
            handles.append(Patch(facecolor="none", edgecolor=PALETTE["red_3"], lw=1.5))
            labels.append(f"auto-primary pick ({gate['overrode']}), overridden")
            _mark_winner(axes, conds, gate["overrode"], color=PALETTE["red_3"])
        legend_panel(fig.add_subplot(gs[r, 5]), handles, labels, loc="center left")
        first_axes.append(axes[0])
    label_panels(first_axes, x=-0.35)
    return save_figure(fig, out / "fig5_mode_order")


def _mark_winner(axes, conds: list[str], winner: str | None, color: str | None = None) -> None:
    if winner not in conds:
        return
    i = conds.index(winner)
    for ax in axes:
        bar = ax.patches[i]
        bar.set_edgecolor(color or "black")
        bar.set_linewidth(1.5)


def fig_features(state: dict, out: Path) -> Path:
    """Supplement: adding phenotype features does not degrade the audited build."""
    rows = rows_from_cells("features", state)
    write_rows(rows, out / "figS1_features.csv")
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    colors = dict(
        zip(conds, [PALETTE["neutral_dark"]] + CATEGORICAL[: len(conds) - 1], strict=True)
    )
    fig = plt.figure(figsize=(14, 3.6))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.9], wspace=0.3)
    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    _bars_by_criterion(axes, rows, conds, colors)
    legend_panel(
        fig.add_subplot(gs[0, 5]), [Patch(color=colors[c]) for c in conds], conds, loc="center left"
    )
    return save_figure(fig, out / "figS1_features")


# --- schematics -------------------------------------------------------------


def _box(ax, xy, w, h, text, fc="white", ec="#333333", fontsize=None, weight="normal"):
    ax.add_patch(
        FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.3)
    )
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize or plt.rcParams["xtick.labelsize"],
        fontweight=weight,
        wrap=True,
    )


def _arrow(ax, p, q):
    ax.add_patch(
        FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, color="#333333", lw=1.3)
    )


def fig_pipeline(out: Path) -> Path:
    """Schematic: cluster -> evidence bundle -> component prompt -> recall JSON -> MCP + expert."""
    fig, ax = plt.subplots(figsize=(14, 4.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.6)
    ax.set_axis_off()

    _box(
        ax,
        (0.3, 2.6),
        2.2,
        1.3,
        "Phenotypic cluster\n(genes co-clustered by\nmorphology)",
        fc="#F2F2F2",
    )
    _box(ax, (0.3, 0.6), 2.2, 1.3, "Screen context\n(channels, cell line)", fc="#F2F2F2")
    _box(
        ax,
        (3.1, 1.6),
        2.3,
        2.1,
        "Evidence bundle\nper gene\n\nUniProt · Affinage",
        fc=PALETTE["green_1"],
    )
    small = plt.rcParams["xtick.labelsize"] - 2
    _arrow(ax, (2.5, 3.2), (3.1, 3.0))
    _arrow(ax, (2.5, 1.3), (3.1, 2.2))

    x0, y0, w, h = 6.0, 0.5, 2.9, 3.6
    ax.set_xlim(0, 14)
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0), w, h, boxstyle="round,pad=0.02", fc="white", ec=PALETTE["key"], lw=1.8
        )
    )
    ax.text(x0 + w / 2, y0 + h - 0.22, "Component prompt", ha="center", va="top", fontweight="bold")
    for i, (code, name) in enumerate(COMPONENTS):
        yy = y0 + h - 0.85 - i * 0.55
        ax.add_patch(
            FancyBboxPatch(
                (x0 + 0.2, yy - 0.2),
                w - 0.4,
                0.42,
                boxstyle="round,pad=0.01",
                fc=PALETTE["key_light"],
                ec="none",
                alpha=0.25,
            )
        )
        ax.text(
            x0 + 0.35,
            yy,
            f"{code}",
            va="center",
            fontweight="bold",
            fontsize=plt.rcParams["xtick.labelsize"],
        )
        ax.text(x0 + 1.0, yy, name, va="center", fontsize=small)
    _arrow(ax, (5.4, 2.65), (6.0, 2.65))
    ax.text(
        x0 + w / 2,
        y0 + 0.28,
        "chain-of-thought · literature gap-fill",
        ha="center",
        va="center",
        fontsize=small,
        style="italic",
    )

    _box(
        ax,
        (9.5, 1.3),
        2.4,
        2.1,
        "Recall JSON\n\npathway + confidence\nper-gene category\n(established / novel role /\nuncharacterized)\nsub-class + rationale",
        fc=PALETTE["green_1"],
        fontsize=small,
    )
    _arrow(ax, (8.9, 2.35), (9.5, 2.35))
    _box(
        ax,
        (12.4, 1.3),
        1.4,
        2.1,
        "MCP chat\n+ domain\nexpert\n\n(synthesis)",
        fc="#F2F2F2",
        fontsize=small,
    )
    _arrow(ax, (11.9, 2.35), (12.4, 2.35))
    ax.text(
        0.3,
        4.35,
        "MozzareLLM: recall upstream, interpretation downstream",
        fontweight="bold",
        va="center",
    )
    return save_figure(fig, out / "fig1_pipeline")


def fig_benchmark_design(state: dict, out: Path) -> Path:
    """Schematic of the benchmark (clusters, reviewers, consensus, chain) + reviewer concordance."""
    conc = state["reviewer_concordance"]
    diag = next(iter(state["diagnostics"].values()))
    decoys = next(iter(state["decoys"].values()))
    scored = diag["per_cluster"]
    n_genes = diag["n_scored"]
    small = plt.rcParams["xtick.labelsize"] - 2

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[3.0, 1, 1], wspace=0.4)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 13.5)
    ax.set_ylim(-0.6, 5)
    ax.set_axis_off()

    ax.text(
        0.2,
        4.75,
        f"Scored clusters ({n_genes} genes)",
        fontweight="bold",
        va="center",
        fontsize=small + 1,
    )
    y = 4.2
    for name, rec in scored.items():
        _box(
            ax,
            (0.2, y - 0.22),
            4.0,
            0.44,
            f"{name}  ·  n = {rec['n']}",
            fc=PALETTE["green_1"],
            fontsize=small - 1,
        )
        y -= 0.55
    y -= 0.15
    ax.text(0.2, y, "Controls", fontweight="bold", va="center", fontsize=small + 1)
    y -= 0.5
    for d in decoys:
        tag = "expect abstain" if d["expectation"] == "abstain" else "expect functional"
        label = f"{d['screen']}/{d['cluster']}  ·  {tag}".replace(
            "aconcagua_interphase_shuffled", "shuffled"
        )
        _box(ax, (0.2, y - 0.22), 4.0, 0.44, label, fc="#F2F2F2", fontsize=small - 1)
        y -= 0.55

    _box(
        ax,
        (4.6, 2.9),
        3.4,
        1.5,
        f"{len(conc['reviewers'])} expert reviewers\nblinded to source\n\n≥2-of-3 consensus",
        fc="white",
        fontsize=small,
    )
    _arrow(ax, (4.2, 3.7), (4.6, 3.7))
    _box(
        ax,
        (8.8, 2.6),
        4.5,
        1.8,
        "Five selection criteria\ncategory recall\nnovel-role sub-class\nuncharacterized sub-class\npathway coherence\ncontrols",
        fc="white",
        fontsize=small,
    )
    _arrow(ax, (8.0, 3.7), (8.8, 3.7))

    ax.text(
        4.4,
        1.9,
        "Experiment chain: each step carries its winner forward",
        va="center",
        fontsize=small,
        style="italic",
    )
    steps = ["source", "walkup", "mode", "order"]
    for i, s in enumerate(steps):
        x = 4.4 + i * 1.9
        _box(
            ax,
            (x, 0.9),
            1.5,
            0.6,
            s,
            fc=PALETTE["key_light"] if s == "walkup" else "#F2F2F2",
            fontsize=small,
        )
        if i:
            _arrow(ax, (x - 0.4, 1.2), (x, 1.2))

    ax2 = fig.add_subplot(gs[0, 1])
    fracs = [conc["by_class"][c]["frac"] for c, _ in CLASSES]
    ax2.bar(range(len(CLASSES)), fracs, color=PALETTE["neutral_dark"], width=0.7)
    ax2.axhline(conc["unanimous_frac"], color="black", alpha=0.3, ls="--")
    ax2.set_xticks(range(len(CLASSES)))
    ax2.set_xticklabels([lbl for _, lbl in CLASSES], rotation=20, ha="right")
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("Reviewer unanimity")
    ax2.set_title(
        f"Fleiss κ = {conc['fleiss_kappa']:.2f}",
        fontsize=plt.rcParams["axes.labelsize"],
        fontweight="normal",
    )

    ax3 = fig.add_subplot(gs[0, 2])
    names = list(conc["pairwise"])
    ax3.bar(
        range(len(names)),
        [conc["pairwise"][k] for k in names],
        color=PALETTE["neutral_dark"],
        width=0.7,
    )
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels([k.replace("-", " vs ") for k in names], rotation=20, ha="right")
    ax3.set_ylim(0, 1.02)
    ax3.set_ylabel("Pairwise category agreement")
    label_panels([ax, ax2, ax3])
    return save_figure(fig, out / "fig2_benchmark_design")


# --- driver ---------------------------------------------------------------------


def _load(experiment: str) -> dict | None:
    path = OUTPUTS / experiment / f"{experiment}_state.json"
    return json.loads(path.read_text()) if path.exists() else None


def render_all(out: Path) -> list[Path]:
    setup_plot_style()
    out.mkdir(parents=True, exist_ok=True)
    states = {e: _load(e) for e in ("source", "walkup", "mode", "order", "features")}
    written = [fig_pipeline(out)]
    design_state = states["mode"] or states["order"] or states["source"]
    if design_state:
        written.append(fig_benchmark_design(design_state, out))
    if states["source"]:
        written.append(fig_source(states["source"], out))
    if states["walkup"] and states["walkup"].get("stages"):
        written.append(fig_walkup(states["walkup"], out))
    if states["mode"] or states["order"]:
        written.append(fig_mode_order(states["mode"], states["order"], out))
    if states["features"]:
        written.append(fig_features(states["features"], out))
    for p in written:
        print(f"[figures] {p}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUTPUTS / "figures")
    args = ap.parse_args()
    render_all(args.out)


if __name__ == "__main__":
    main()
