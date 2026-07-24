"""Benchmark figures, rendered from the step state files.

Figure 0 = inter-reviewer concordance (the incoming ground-truth quality);
Figure 1 = source decision, one plot showing the winner (source_state.json);
Figure 2 = build-up prompt walkup (walkup_state.json);
Figure 3 = mode (mode_state.json).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

from architecture_benchmarking_workflow.bench_evaluator import MetricPanel
from architecture_benchmarking_workflow.bench_pipeline_common import metric_value

# Colorblind-safe, distinguishable per evidence source; matches the palette
# used for the same source axis in analysis/make_figures.py.
SOURCE_COLORS = {
    "uniprot": "#2E5C8A",
    "affinage": "#B84A3E",
    "both": "#6E8B3D",
}
_CLASSES = ("ESTABLISHED", "NOVEL_ROLE", "UNCHARACTERIZED")
_SOURCE_ORDER = ("uniprot", "affinage", "both")

# The per-source metric profile (Fig 1).
_PROFILE = [("coverage_weighted_category", "cw-recall"), ("category", "category"),
            ("coverage", "coverage"), ("novel_subclass", "novel"), ("unchar_subclass", "unchar")]


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


def _panel_from_json(d: dict) -> MetricPanel:
    return MetricPanel(
        category=d["category"],
        novel_subclass=tuple(d["novel_subclass"]),
        unchar_subclass=tuple(d["unchar_subclass"]),
        coherence=tuple(d["coherence"]),
        n=d["n"],
        failures=d["failures"],
    )


def _save(fig, out_png: Path) -> None:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_reviewer_concordance(state_path: Path, out_png: Path) -> None:
    """Figure 0: inter-reviewer concordance & evidence sufficiency (incoming ground truth).

    A: full-agreement by annotation level -- pathway (cluster), category (gene), and
    sub-class split by family. B: the NOVEL_ROLE evidence-ladder is an ordinal scale
    on which reviewers differ in calibration (eric conservative -> iain lenient), so
    exact agreement is low but the ordering is monotone and the consensus is the
    median reviewer. C: reviewer web usage by class (evidence-bundle sufficiency).
    """
    _setup_style()
    conc = json.loads(Path(state_path).read_text())["reviewer_concordance"]
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(18, 6))

    # A: concordance by level, with sub-class split into UNCHAR (categorical) and NOVEL (ordinal)
    lv, sc = conc["levels"], conc["subclass"]
    un, nv = sc["unchar"], sc["novel"]
    a_vals = [lv["pathway"]["frac"] * 100, lv["category"]["frac"] * 100,
              (un["agree"] / un["n"] * 100) if un["n"] else 0.0,
              (nv["exact"] / nv["n"] * 100) if nv["n"] else 0.0]
    a_tags = [f'{lv["pathway"]["agree"]}/{lv["pathway"]["n"]}', f'{conc["unanimous"]}/{lv["category"]["n"]}',
              f'{un["agree"]}/{un["n"]}', f'{nv["exact"]}/{nv["n"]}']
    a_labs = ["Pathway\n(cluster)", "Category\n(gene)", "Sub-class\nUNCHAR", "Sub-class\nNOVEL"]
    bars = ax_a.bar(range(4), a_vals, color=["#6E8B3D", "#2E5C8A", "#4C9A8A", "#B84A3E"],
                    edgecolor="white", zorder=2, width=0.66)
    for b, tag in zip(bars, a_tags, strict=True):
        ax_a.annotate(f"{b.get_height():.0f}%\n{tag}", (b.get_x() + b.get_width() / 2, b.get_height()),
                      xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9,
                      fontweight="bold")
    ax_a.annotate(f"but {nv['monotone']}/{nv['n']}\nmonotone (ordinal)", (3, a_vals[3]),
                  xytext=(0, 40), textcoords="offset points", ha="center", fontsize=8,
                  color="#B84A3E", style="italic")
    ax_a.set_xticks(range(4))
    ax_a.set_xticklabels(a_labs, fontsize=9)
    ax_a.set_ylim(0, 118)
    ax_a.set_ylabel("Reviewers in full agreement (%)", fontsize=11)
    ax_a.set_title(f"Concordance by level  (Fleiss κ = {conc['fleiss_kappa']:.2f})",
                   fontsize=12, fontweight="bold")

    # B: NOVEL evidence-ladder calibration -- reviewer marginals ordered by mean level
    cal = sc["calibration"]
    order, marg, labs = cal["order"], cal["marginals"], cal["labels"]
    shades = {order[0]: "#A8C0D0", order[1]: "#5A7D92", order[2]: "#243B47"}
    xb = np.arange(len(labs))
    w = 0.8 / len(order)
    for i, r in enumerate(order):
        offset = (i - (len(order) - 1) / 2) * w
        ax_b.bar(xb + offset, [marg[r][lab] for lab in labs], w,
                 label=f"{r} (μ={cal['means'][r]:.1f})", color=shades[r], edgecolor="white", zorder=2)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels([lab.replace("_EVIDENCE", "").title() for lab in labs], fontsize=8)
    ax_b.set_ylabel("Novel genes (count)", fontsize=11)
    med = sc["consensus_is_median"]
    ax_b.set_title(f"Novel evidence-ladder: reviewer calibration\n"
                   f"consensus = median ({med['reviewer']}) in {med['match']}/{med['n']}",
                   fontsize=11, fontweight="bold")
    ax_b.legend(frameon=False, fontsize=8, title="reviewer (mean level)")

    # C: reviewer web usage by class -- evidence-bundle sufficiency
    web = conc["web"]
    wv = [web["by_class"][c]["frac"] * 100 for c in _CLASSES]
    b3 = ax_c.bar(range(len(_CLASSES)), wv,
                  color=[SOURCE_COLORS["uniprot"], SOURCE_COLORS["affinage"], SOURCE_COLORS["both"]],
                  edgecolor="white", zorder=2, width=0.6)
    for b, c in zip(b3, _CLASSES, strict=True):
        ax_c.annotate(f"{b.get_height():.0f}%\nn={web['by_class'][c]['n']}",
                      (b.get_x() + b.get_width() / 2, b.get_height()),
                      xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax_c.axhline(web["overall"]["frac"] * 100, ls="--", color="#333333", lw=1,
                 label=f"overall {web['overall']['frac'] * 100:.0f}%")
    ax_c.set_xticks(range(len(_CLASSES)))
    ax_c.set_xticklabels([c.replace("_", "\n") for c in _CLASSES], fontsize=9)
    ax_c.set_ylim(0, 100)
    ax_c.set_ylabel("Annotations needing web lookup (%)", fontsize=11)
    ax_c.set_title("Reviewer web use → bundle sufficiency", fontsize=12, fontweight="bold")
    ax_c.legend(frameon=False, fontsize=9)

    fig.suptitle("Figure 0 — Inter-reviewer concordance & evidence sufficiency "
                 "(ground-truth characterization)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out_png)


def figure_source_winner(state_path: Path, out_png: Path) -> None:
    """Figure 1: the source decision -- metric profile + human vote.

    Left: a parallel-coordinates profile of each source across the evaluation
    metrics (cw-recall, category, coverage, novel/unchar sub-class), with the human
    inter-reviewer ceiling band. The sources trade metrics off, but affinage wins
    the selection primary (cw-recall). Right: the reviewers' blinded source
    preference -- affinage wins the human vote too.
    """
    _setup_style()
    state = json.loads(Path(state_path).read_text())
    cells = {k: _panel_from_json(v) for k, v in state["cells"].items()}
    pref = state["source_preference"]["overall"]
    lo, hi = state["reviewer_concordance"]["ceiling"]
    winner_src = state["source"]
    srcs = [s for s in _SOURCE_ORDER if f"{s}::single_call" in cells]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 6.5),
                                     gridspec_kw={"width_ratios": [3, 1.3]})

    xs = list(range(len(_PROFILE)))
    ax_l.axhspan(lo * 100, hi * 100, color="#999999", alpha=0.14, zorder=0)
    ax_l.annotate("human ceiling", (xs[-1], hi * 100), xytext=(-4, 4), textcoords="offset points",
                  ha="right", fontsize=8, color="#666666")
    for s in srcs:
        p = cells[f"{s}::single_call"]
        ys = [metric_value(p, m) * 100 for m, _ in _PROFILE]
        win = s == winner_src
        ax_l.plot(xs, ys, "-o", color=SOURCE_COLORS.get(s, "#666"), lw=3.0 if win else 1.8,
                  markersize=9 if win else 6, alpha=1.0 if win else 0.5, zorder=3 if win else 2,
                  label=f"{s} (winner)" if win else s)
        if win:
            for x, y in zip(xs, ys, strict=True):
                ax_l.annotate(f"{y:.0f}", (x, y), xytext=(0, 9), textcoords="offset points",
                              ha="center", fontsize=9, fontweight="bold", color=SOURCE_COLORS.get(s))
    ax_l.set_xticks(xs)
    ax_l.set_xticklabels([lab for _, lab in _PROFILE], fontsize=11)
    ax_l.set_xlim(-0.3, len(_PROFILE) - 0.7)
    ax_l.set_ylim(0, 108)
    ax_l.set_ylabel("Score (%)", fontsize=11)
    ax_l.grid(axis="x", ls=":", alpha=0.4)
    ax_l.set_title("Source metric profile (single_call, completeness floor)",
                   fontsize=12, fontweight="bold")
    ax_l.legend(frameon=False, fontsize=10, loc="lower left")

    order = ["affinage", "uniprot", "both", "neither"]
    pv = [pref.get(p, 0) for p in order]
    b = ax_r.barh(range(len(order)), pv,
                  color=[SOURCE_COLORS.get(p, "#999999") for p in order], edgecolor="white", zorder=2)
    for bar, val in zip(b, pv, strict=True):
        ax_r.annotate(str(val), (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                      xytext=(4, 0), textcoords="offset points", va="center", fontsize=10,
                      fontweight="bold")
    ax_r.set_yticks(range(len(order)))
    ax_r.set_yticklabels(order, fontsize=10)
    ax_r.invert_yaxis()
    ax_r.set_xlim(0, max(pv) * 1.2)
    ax_r.set_xlabel("reviewer preference votes", fontsize=10)
    ax_r.set_title("Human source preference", fontsize=12, fontweight="bold")

    fig.suptitle("Figure 1 — Affinage wins the primary metric (cw-recall) and the human vote",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out_png)
