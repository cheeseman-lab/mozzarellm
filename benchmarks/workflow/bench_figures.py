"""Paper figures for MozzareLLM, drawn from benchmark state files and one screen run.

Two figures, each written as one file per panel plus a lettered composite for review:

- ``mozzarellm/``: the MozzareLLM third of the tool figure -- how a cluster is labeled,
  the labeled screen, and one cluster zoomed in (one set per ``EXAMPLES`` cluster).
- ``benchmark/``: the benchmark supplement -- its design, expert agreement, what each
  step of building the prompt changed, the delivery matrix, and the confidence calls on
  a full screen.

Every panel is drawn into a millimetre ``Frame``, so the panel files and the composite
come from the same code at print size (183 mm wide). Panels read only the tidy CSVs under
``figures/data/``: the benchmark tables are flattened from the experiment state JSONs, and
the screen tables are collected once from a screen run with ``--collect-aconcagua``.

Usage:
    python -m benchmarks.workflow.bench_figures [--out benchmarks/outputs/figures]
    python -m benchmarks.workflow.bench_figures --collect-aconcagua RUNS_DIR LEGACY_DIR PHATE_TSV
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import MultipleLocator

from .plot_style import (
    CATEGORY_COLORS,
    CONFIDENCE_COLORS,
    INK,
    STEP_COLORS,
    TOOL,
    setup_plot_style,
)

BENCH_DIR = Path(__file__).resolve().parents[1]
OUTPUTS = BENCH_DIR / "outputs"

CRITERIA = [
    ("category", "category\nrecall"),
    ("novel_subclass", "novel-role\nsub-class"),
    ("unchar_subclass", "uncharacterized\nsub-class"),
    ("coherence", "pathway\ncoherence"),
    ("controls", "controls\npassed"),
]
COMPONENTS = [
    ("CAT", "task"),
    ("GCR", "category rules"),
    ("NPR", "novel-role rules"),
    ("UPR", "uncharacterized rules"),
    ("PCC", "pathway confidence"),
]
CLASSES = [
    ("ESTABLISHED", "established"),
    ("NOVEL_ROLE", "novel role"),
    ("UNCHARACTERIZED", "uncharacterized"),
]
DELIVERIES = [("single_call", "single call"), ("cot", "chain of thought"), ("stepwise", "stepwise")]
LIT_SETTINGS = [("", "none"), ("_lit", "check"), ("_litb", "gap-fill")]
KEPT_LABELS = {
    "affinage": "Affinage",
    "cot_litb": "chain of thought + PubMed",
    "O": "canonical",
    "features_strength": "features + strength",
}
SPREAD_ARMS = [
    ("legacy", "legacy pipeline"),
    ("shipped", "previous rubric"),
    ("counted_tiers", "counted tiers"),
    ("coverage_tiers", "coverage tiers"),
]
SHIPPED_ARM = "coverage_tiers"
LEVELS = ["High", "Medium", "Low", "none"]
LEVEL_LABEL = {"High": "High", "Medium": "Medium", "Low": "Low", "none": "no pathway"}

# The main-figure clusters, with display titles abridged by hand from the model's own label.
EXAMPLES = {
    163: "CCT/TRiC chaperonin complex-mediated protein folding",
    13: "Pre-mRNA splicing / spliceosome assembly and catalysis",
}
CALLOUTS = {
    27: ("Ribosome biogenesis", (-0.083, -0.009, "left", "top")),
    36: ("Ribosome biogenesis and translation initiation", (-0.050, -0.036, "center", "top")),
    94: ("Pre-mRNA splicing", (-0.002, -0.0545, "right", "center")),
}

MM = 1 / 25.4
T6, T5 = 6, 5  # label and smallest size, in points (Nature's floor is 5 pt)
LW, LEAD = 0.5, 0.35
RUST, AFF, BLUE = TOOL["mozzarellm"], TOOL["affinage"], TOOL["mascarpone"]
GOLD = STEP_COLORS["truth"]
INK1, INK2, INK3, RULE = INK["ink"], INK["ink2"], INK["ink3"], INK["rule"]
CAT, CONF = CATEGORY_COLORS, CONFIDENCE_COLORS


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
    """Tidy rows for a cells-style state (source / mode / order / features).

    A recorded human gate relabels the build it kept as ``kept`` and the automatic
    pick it overrode as ``overridden``.
    """
    rows = []
    winner = state.get("winner_condition")
    dominated = {c.split("__")[0] for c in state.get("dominated", [])}
    gate = state.get("human_gate") or {}
    for cell, panel in state["cells"].items():
        cond = condition_of(cell)
        if cond == gate.get("to"):
            role = "kept"
        elif cond == gate.get("overrode"):
            role = "overridden"
        else:
            role = (
                "winner" if cond == winner else ("dominated" if cond in dominated else "candidate")
            )
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


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["value"] = float(r["value"])
        r["numerator"] = int(float(r["numerator"]))
        r["denominator"] = int(float(r["denominator"]))
    return rows


# --- the step table ---------------------------------------------------------


def _panel(rows: list[dict], condition: str, role: str | None = None) -> dict:
    """{metric: (value, numerator, denominator)} for one condition of one tidy table."""
    out = {
        r["metric"]: (r["value"], r["numerator"], r["denominator"])
        for r in rows
        if r["condition"] == condition
        and (role is None or r["role"] == role)
        and r["metric"] in dict(CRITERIA)
    }
    missing = [k for k, _ in CRITERIA if k not in out]
    if missing:
        raise ValueError(f"condition {condition!r} (role={role}) lacks {missing}")
    return out


def _winner(rows: list[dict]) -> str:
    """The build a step kept: the human-gated pick when one is recorded, else the winner."""
    for role in ("kept", "winner"):
        w = sorted({r["condition"] for r in rows if r["role"] == role})
        if len(w) > 1:
            raise ValueError(f"expected one {role}, found {w}")
        if w:
            return w[0]
    raise ValueError("no winner or kept build in the table")


def step_table(data: Path) -> list[dict]:
    """The chain of steps in order, each with its before / after panels and its goal criterion.

    Every step is measured against its own same-batch baseline, so a row's "before" need not
    equal the row above's "after".
    """

    def criteria(name):
        path = data / f"{name}.csv"
        return [r for r in read_rows(path) if r["panel"] == "criteria"] if path.exists() else []

    steps = []
    src = criteria("source")
    steps.append(_step("source", "source", "category", src, "uniprot", _winner(src)))
    wk = read_rows(data / "walkup.csv")
    for stage, name in COMPONENTS:
        d = [r for r in wk if r["panel"] == stage]
        if not d:
            continue
        sel = sorted({r["condition"] for r in d if r["role"] == "selected"})
        goal = sorted({r["metric"] for r in d if r["role"] == "prior" and r["goal"] == "True"})
        if len(sel) != 1 or len(goal) != 1:
            raise ValueError(f"walkup stage {stage}: selected={sel} goal={goal}")
        steps.append(
            {
                "group": "walkup",
                "label": f"+ {name}",
                "goal": goal[0],
                "kept": sel[0],
                "before": _panel(d, "prior", "prior"),
                "after": _panel(d, sel[0], "selected"),
            }
        )
    md = criteria("mode")
    steps.append(_step("mode", "delivery", "category", md, "single_call", _winner(md)))
    od = criteria("order")
    steps.append(_step("order", "order", "category", od, "O", _winner(od)))
    fd = criteria("features")
    if fd:
        steps.append(_step("features", "+ features", "category", fd, "baseline", _winner(fd)))
    return steps


def _step(group, label, goal, rows, before, after) -> dict:
    return {
        "group": group,
        "label": label,
        "goal": goal,
        "kept": after,
        "before": _panel(rows, before),
        "after": _panel(rows, after),
    }


# --- screen data (collected once from a run) ---------------------------------


def _level(dominant: str, confidence: str) -> str:
    return "none" if str(dominant).startswith("No coherent") else confidence


def _levels(clusters: pd.DataFrame) -> list[str]:
    return [_level(d, v) for d, v in clusters[["dominant_process", "pathway_confidence"]].values]


def collect_aconcagua(runs_dir: Path, legacy_dir: Path, phate: Path, data: Path) -> list[Path]:
    """Flatten a screen's runs into the tidy tables the screen panels read.

    ``runs_dir`` holds one ``run_<arm>/`` per rubric (``analyze_screen`` output);
    ``legacy_dir`` holds the legacy pipeline's ``cluster_<id>.json`` files; ``phate`` is the
    screen's ``phate_leiden_clustering.tsv`` (per-gene map coordinates and cluster).
    """
    data.mkdir(parents=True, exist_ok=True)
    conf = []
    for f in sorted(glob.glob(str(legacy_dir / "cluster_*.json"))):
        x = json.loads(Path(f).read_text())
        conf.append(("legacy", _level(x["dominant_process"], x["pathway_confidence"])))
    for arm, _ in SPREAD_ARMS[1:]:
        c = pd.read_csv(runs_dir / f"run_{arm}" / "aconcagua_r12_clusters.csv")
        conf += [(arm, lv) for lv in _levels(c)]
    run = runs_dir / f"run_{SHIPPED_ARM}"
    clusters = pd.read_csv(run / "aconcagua_r12_clusters.csv")
    clusters["level"] = _levels(clusters)
    genes = pd.read_csv(run / "aconcagua_r12_genes.csv")
    cols = {"gene_symbol_0": "gene", "PHATE_0": "x", "PHATE_1": "y", "perturbation_auc": "auc"}
    p = pd.read_csv(phate, sep="\t", usecols=[*cols, "cluster"]).rename(columns=cols)
    out = [data / f"screen_{n}.csv" for n in ("confidence", "clusters", "genes", "map")]
    with out[0].open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "level", "n"])
        for (arm, key), n in sorted(Counter(conf).items()):
            w.writerow([arm, key, n])
    clusters[["cluster_id", "n_genes", "pathway_confidence", "level"]].to_csv(out[1], index=False)
    genes[["cluster_id", "gene", "category"]].to_csv(out[2], index=False)
    p[["gene", "cluster", "x", "y", "auc"]].to_csv(out[3], index=False)
    return out


def load_screen(data: Path):
    p = pd.read_csv(data / "screen_map.csv")
    c = pd.read_csv(data / "screen_clusters.csv").set_index("cluster_id")
    p["level"] = p.cluster.map(c.level)
    return p, c, pd.read_csv(data / "screen_genes.csv")


def mann_whitney_greater(a, b) -> float:
    """One-sided Mann-Whitney P (a > b): normal approximation, tie- and continuity-corrected."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    ranks = pd.Series(np.concatenate([a, b])).rank().to_numpy()
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    n = n1 + n2
    ties = pd.Series(np.concatenate([a, b])).value_counts().to_numpy()
    sigma = np.sqrt(n1 * n2 / 12 * ((n + 1) - (ties**3 - ties).sum() / (n * (n - 1))))
    z = (u - n1 * n2 / 2 - 0.5) / sigma
    return 0.5 * math.erfc(z / math.sqrt(2))


# --- the millimetre frame ----------------------------------------------------


class Frame:
    """A panel's box on a page, in millimetres with y running down from its top-left."""

    def __init__(self, fig, x, y, w, h):
        self.fig, self.x, self.y, self.w, self.h = fig, x, y, w, h
        self.pw, self.ph = (v / MM for v in fig.get_size_inches())

    def axes(self, x, y, w, h):
        return self.fig.add_axes(
            ((self.x + x) / self.pw, 1 - (self.y + y + h) / self.ph, w / self.pw, h / self.ph)
        )

    def canvas(self, x=0, y=0, w=None, h=None):
        """An axes whose data coordinates are this panel's millimetres (y down)."""
        w, h = w or self.w - x, h or self.h - y
        ax = self.axes(x, y, w, h)
        ax.set_xlim(x, x + w)
        ax.set_ylim(y + h, y)
        ax.set_axis_off()
        return ax

    def letter(self, s):
        self.fig.text(
            self.x / self.pw, 1 - self.y / self.ph, s, fontsize=8, fontweight="bold", va="top"
        )


def style() -> None:
    """The house style at print size: text is set in points, so nothing is scaled at placement."""
    setup_plot_style(font_size=T6)
    plt.rcParams.update(
        {
            "axes.linewidth": LW,
            "axes.edgecolor": INK1,
            "axes.labelcolor": INK1,
            "axes.labelpad": 1.5,
            "axes.titleweight": "normal",
            "xtick.labelsize": T5,
            "ytick.labelsize": T5,
            "xtick.major.width": LW,
            "ytick.major.width": LW,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
            "xtick.major.pad": 1.2,
            "ytick.major.pad": 1.2,
            "xtick.color": INK1,
            "ytick.color": INK1,
            "savefig.bbox": "standard",
        }
    )


def page(w, h):
    return plt.figure(figsize=(w * MM, h * MM))


def save(fig, path: Path) -> list[Path]:
    """SVG and PDF with live text, PNG preview; never cropped, so the page stays at print size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = [path.with_suffix(f".{ext}") for ext in ("svg", "pdf", "png")]
    for p in written:
        fig.savefig(p, dpi=300)
    plt.close(fig)
    return written


def text_w(ax, s, size, **kw) -> float:
    """Width of a string in an axes' x data units."""
    t = ax.text(0, 0, s, fontsize=size, **kw)
    ax.figure.canvas.draw()
    bb = t.get_window_extent()
    t.remove()
    inv = ax.transData.inverted()
    return abs(inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0])


def _splits(words, k):
    if k == 1:
        yield [" ".join(words)]
        return
    for i in range(1, len(words) - k + 2):
        for rest in _splits(words[i:], k - 1):
            yield [" ".join(words[:i]), *rest]


def wrap_mm(ax, s, size, width, **kw) -> str:
    """Wrap to `width` mm in the fewest lines, then balance them; a "/" never starts a line."""
    words = s.replace(" / ", " / ").split(" ")
    n, cur = 1, ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if cur and text_w(ax, trial, size, **kw) > width:
            n, cur = n + 1, word
        else:
            cur = trial
    if n == 1 or len(words) > 12:
        return s if n == 1 else "\n".join(_greedy(ax, words, size, width, **kw))
    best, best_cost = None, float("inf")
    for cand in _splits(words, n):
        ws = [text_w(ax, line, size, **kw) for line in cand]
        if max(ws) > width:
            continue
        cost = sum((max(ws) - w) ** 2 for w in ws[:-1])
        cost += 50 * sum(line.split()[0] in ("/", "and", "of", "&") for line in cand[1:])
        cost += 50 * sum(line.split()[-1] in ("/", "and", "of") for line in cand[:-1])
        if cost < best_cost:
            best, best_cost = cand, cost
    return "\n".join(best or _greedy(ax, words, size, width, **kw))


def _greedy(ax, words, size, width, **kw) -> list[str]:
    lines, cur = [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if cur and text_w(ax, trial, size, **kw) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return [*lines, cur]


def arrow(ax, p, q, color=INK3, lw=0.6):
    ax.add_patch(
        FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=5, lw=lw, color=color, shrinkA=0)
    )


def swatches(ax, x, y, items, size=1.6, gap=3.2):
    """A one-line legend of (color, text) pairs on a mm canvas."""
    for color, t in items:
        ax.add_patch(Rectangle((x, y - size / 2), size, size, fc=color, ec="none"))
        ax.text(x + size + 0.7, y, t, fontsize=T5, va="center", color=INK2)
        x += size + 0.7 + text_w(ax, t, T5) + gap


# --- the MozzareLLM panels ---------------------------------------------------

MAIN_W, MAIN_H = 183, 57
A_W, B_W, C_W, GUT = 61, 57, 55, 5
MAP_X, MAP_Y = (-0.0935, 0.0325), (-0.0585, 0.0225)  # the aconcagua map's extent
B_MAP = (0, 5, B_W, 41)  # the map inside panel b, mm
C_AX = (0, 11, C_W, 39)  # the zoom inside panel c, mm


def panel_labeling(fr: Frame, p, c, genes, k: int):
    """How one cluster is labeled: its genes and their evidence go in, a pathway comes out."""
    ax = fr.canvas()
    ex = genes[genes.cluster_id == k]
    xy = p[p.cluster == k][["x", "y"]].to_numpy()
    core = np.median(xy, 0)
    span = np.percentile(np.abs(xy - core), 75, 0).max() * 2.6 or 1
    xy = (xy - core) / span
    xy = xy[(np.abs(xy) <= 0.55).all(1)]
    ax.scatter(8 + xy[:, 0] * 11, 10.5 - xy[:, 1] * 11, s=1.6, color=INK2, lw=0)
    ax.text(0, 17.2, f"cluster {k} · {len(ex)} genes", fontsize=T5, va="top", color=INK2)

    est = ex[ex.category == "ESTABLISHED"].sort_values("gene")
    first = (est if len(est) else ex).gene.iloc[0]
    for j in range(3):
        ax.add_patch(Rectangle((j * 1.1, 21 + j * 2.1), 13.2, 8, fc="white", ec=RULE, lw=LW))
    x0, y0 = 2.2, 25.2
    ax.text(x0 + 1, y0 + 1.5, first, fontsize=T5, style="italic", va="center")
    ax.add_patch(Rectangle((x0 + 10.2, y0 + 0.8), 1.4, 1.4, fc=AFF, ec="none"))
    for i, w in enumerate([10.8, 9.6, 7.4]):
        y = y0 + 3.6 + i * 1.35
        ax.plot([x0 + 1, x0 + 1 + w], [y, y], color="#CFCFCF", lw=0.9, solid_capstyle="round")
    ax.text(0, 35.2, "per-gene evidence", fontsize=T5, va="top", color=INK2)
    ax.text(0, 37.8, "Affinage, then UniProt", fontsize=T5, va="top", color=AFF)
    ax.add_patch(
        FancyBboxPatch(
            (0, 43), 15.4, 7.2, boxstyle="round,pad=0,rounding_size=1.2", fc="#F0F0F0", ec="none"
        )
    )
    ax.text(7.7, 46.6, "screen context\nHeLa · 4 markers", fontsize=T5, ha="center",
            va="center", color=INK2, linespacing=1.15)  # fmt: skip

    nx, ny, nh = 20.0, 27.5, 7.4
    nw = text_w(ax, "MozzareLLM", T6, fontweight="bold") + 3.4
    ax.add_patch(
        FancyBboxPatch(
            (nx, ny - nh / 2), nw, nh, boxstyle="round,pad=0,rounding_size=1.8", fc=RUST, ec="none"
        )
    )
    ax.text(nx + nw / 2, ny, "MozzareLLM", fontsize=T6, ha="center", va="center", color="white",
            fontweight="bold")  # fmt: skip
    ax.text(nx + nw / 2, ny + nh / 2 + 2, "one reasoning call\nPubMed only for\nunannotated genes",
            fontsize=T5, ha="center", va="top", color=INK2, linespacing=1.15)  # fmt: skip
    bus = 17.6
    for y in (10.5, 27.5, 46.6):
        ax.plot([16.0, bus], [y, y], color=INK3, lw=0.6, solid_capstyle="butt")
    ax.plot([bus, bus], [10.5, 46.6], color=INK3, lw=0.6)
    arrow(ax, (bus, ny), (nx - 0.4, ny))
    arrow(ax, (nx + nw + 0.5, ny), (nx + nw + 3.0, ny))

    ox = nx + nw + 3.8
    ax.text(ox, 5, "pathway", fontsize=T5, va="top", color=INK3)
    title = wrap_mm(ax, EXAMPLES[k], T6, A_W - ox - 0.3)
    ax.text(ox, 7.6, title, fontsize=T6, va="top", linespacing=1.15, color=INK1)
    y = 7.6 + (title.count("\n") + 1) * 2.65 + 1.8
    ax.text(ox, y, "confidence", fontsize=T5, color=INK3, va="center")
    ax.text(ox + text_w(ax, "confidence ", T5), y, c.loc[k].pathway_confidence, fontsize=T5,
            color=RUST, va="center")  # fmt: skip
    cells = [kk for kk, _ in CLASSES for _ in range(int((ex.category == kk).sum()))]
    ncol = 10
    s = (A_W - ox - 0.6) / ncol - 0.35
    y0 = y + 2.4
    for i, kk in enumerate(cells):
        r, q = divmod(i, ncol)
        ax.add_patch(
            Rectangle((ox + q * (s + 0.35), y0 + r * (s + 0.35)), s, s, fc=CAT[kk], ec="none")
        )
    y = y0 + -(-len(cells) // ncol) * (s + 0.35) + 2.2
    for kk, name in CLASSES:
        ax.add_patch(Rectangle((ox, y - 0.75), 1.5, 1.5, fc=CAT[kk], ec="none"))
        ax.text(ox + 2.3, y, f"{name} {int((ex.category == kk).sum())}", fontsize=T5,
                va="center", color=INK2)  # fmt: skip
        y += 2.6


def zoom_window(p, k):
    """Every gene of the cluster in view at the map's aspect, the right 28% left for labels."""
    xmm = (MAP_X[1] - MAP_X[0]) / B_MAP[2]
    ymm = (MAP_Y[1] - MAP_Y[0]) / B_MAP[3]
    d = p[p.cluster == k][["x", "y"]]
    (x0, y0), (x1, y1) = d.min().to_numpy(), d.max().to_numpy()
    padx, pady = 0.07 * (x1 - x0) + 0.0004, 0.07 * (y1 - y0) + 0.0004
    s = max((x1 - x0 + 2 * padx) / 0.72 / (C_AX[2] * xmm), (y1 - y0 + 2 * pady) / (C_AX[3] * ymm))
    w, h = C_AX[2] * xmm * s, C_AX[3] * ymm * s
    left = x0 - padx - (0.72 * w - (x1 - x0 + 2 * padx)) / 2
    cy = (y0 + y1) / 2
    return left, left + w, cy - h / 2, cy + h / 2


def densest(p, k, n=6):
    """The gene of cluster k with the tightest n nearest same-cluster neighbours."""
    xy = p[p.cluster == k][["x", "y"]].to_numpy()
    dist = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    return xy[np.argmin(np.sort(dist, 1)[:, 1 : n + 1].sum(1))]


def panel_screen(fr: Frame, p, c, k: int, tag: str | None = None):
    """The whole screen on its map, every gene colored by its cluster's confidence."""
    ax = fr.axes(*B_MAP)
    for lv in ["none", "Low", "Medium", "High"]:
        d = p[p.level == lv]
        ax.scatter(d.x, d.y, s=0.6, color=CONF[lv], lw=0, rasterized=True,
                   zorder=2 if lv == "High" else 1)  # fmt: skip
    ax.set_axis_off()
    ax.set_xlim(*MAP_X)
    ax.set_ylim(*MAP_Y)
    zx0, zx1, zy0, zy1 = zoom_window(p, k)
    ax.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fc="none", ec=INK1, lw=LW, zorder=4))
    if tag:
        ax.text(zx0, zy1 + 0.0008, tag, fontsize=T6, fontweight="bold", va="bottom", zorder=5,
                bbox={"fc": "white", "ec": "none", "pad": 0.3, "alpha": 0.85})  # fmt: skip
    halo = {"fc": "white", "ec": "none", "pad": 0.4, "alpha": 0.85}
    for kk, (label, (tx, ty, ha, va)) in CALLOUTS.items():
        if not (p.cluster == kk).any():
            continue
        mx, my = densest(p, kk)
        ax.plot([mx, tx], [my, ty], color=INK3, lw=LEAD, zorder=4)
        dx = -0.0008 if ha == "right" else (-0.001 if ha == "left" else 0)
        ax.text(tx + dx, ty - (0.0008 if va == "top" else 0), label, fontsize=T5, va=va, ha=ha,
                color=INK1, zorder=5, bbox=halo)  # fmt: skip
    top = B_MAP[1] + B_MAP[3] + 1.5
    cv = fr.canvas(0, top, B_W, fr.h - top)
    counts = c.level.value_counts()
    cv.text(0, top + 1.5, f"{len(c)} clusters labeled", fontsize=T6, va="center", color=INK1)
    items = [(CONF[lv], f"{LEVEL_LABEL[lv]} {counts.get(lv, 0)}") for lv in LEVELS]
    swatches(cv, 0, top + 5.1, items)


def panel_zoom(fr: Frame, p, c, genes, k: int):
    """One cluster zoomed in on the map, its novel-role and uncharacterized genes named."""
    ex_g = genes[genes.cluster_id == k]
    cv = fr.canvas()
    head = wrap_mm(cv, EXAMPLES[k], T6, fr.w - 5)
    cv.text(5, 0.3, head, fontsize=T6, va="top", linespacing=1.15, color=INK1)
    y = 0.3 + (head.count("\n") + 1) * 2.65 + 1.4
    lead = f"cluster {k} · {len(ex_g)} genes · confidence "
    cv.text(5, y, lead, fontsize=T5, va="center", color=INK2)
    cv.text(5 + text_w(cv, lead, T5), y, c.loc[k].pathway_confidence, fontsize=T5, va="center",
            color=RUST)  # fmt: skip

    ax = fr.axes(*C_AX)
    zx0, zx1, zy0, zy1 = zoom_window(p, k)
    ax.set_xlim(zx0, zx1)
    ax.set_ylim(zy0, zy1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(LW)
    bg = p[p.cluster != k]
    for lv in ["none", "Low", "Medium", "High"]:
        d = bg[bg.level == lv]
        ax.scatter(d.x, d.y, s=1.5, color=CONF[lv], lw=0, alpha=0.4, rasterized=True)
    ex = p[p.cluster == k].copy()
    ex["kind"] = ex.gene.map(ex_g.set_index("gene").category).fillna("ESTABLISHED")
    for kk, _ in CLASSES:
        d = ex[ex.kind == kk]
        ax.scatter(d.x, d.y, s=8 if kk == "ESTABLISHED" else 12, color=CAT[kk], ec="white",
                   lw=0.3, zorder=3)  # fmt: skip
    off = ex[ex.kind != "ESTABLISHED"].sort_values("y", ascending=False)
    col_x = zx1 - 0.26 * (zx1 - zx0)
    gap = (zy1 - zy0) * 0.062
    ys, last = [], zy1 - gap * 0.8
    for yv in off.y:
        ys.append(min(max(yv, zy0 + gap * 0.8), last))
        last = ys[-1] - gap
    for (_, r), ty in zip(off.iterrows(), ys, strict=True):
        if zx0 <= r.x <= zx1 and zy0 <= r.y <= zy1:
            ax.plot([r.x, col_x - 0.012 * (zx1 - zx0)], [r.y, ty], color=INK3, lw=LEAD, zorder=2)
            name = r.gene
        else:
            name = f"{r.gene} ›"
        ax.text(col_x, ty, name, fontsize=T5, va="center", color=CAT[r.kind], zorder=4,
                style="italic")  # fmt: skip
    items = [(CAT[kk], f"{name} {int((ex.kind == kk).sum())}") for kk, name in CLASSES]
    swatches(cv, 0, C_AX[1] + C_AX[3] + 3.4, items)


def render_mozzarellm(out: Path, p, c, genes, k: int) -> list[Path]:
    panels = [
        (
            "mozzarellm_labeling_schematic",
            A_W,
            lambda fr, t=None: panel_labeling(fr, p, c, genes, k),
        ),
        ("mozzarellm_labeled_screen", B_W, lambda fr, t=None: panel_screen(fr, p, c, k, t)),
        ("mozzarellm_cluster_zoom", C_W, lambda fr, t=None: panel_zoom(fr, p, c, genes, k)),
    ]
    written = []
    for name, w, draw in panels:
        fig = page(w, MAIN_H)
        draw(Frame(fig, 0, 0, w, MAIN_H))
        written += save(fig, out / name)
    fig = page(MAIN_W, MAIN_H)
    x = 0
    for (_, w, draw), letter in zip(panels, "abc", strict=True):
        fr = Frame(fig, x, 0, w, MAIN_H)
        draw(fr, "c" if letter == "b" else None)
        fr.letter(letter)
        x += w + GUT
    return written + save(fig, out / "mozzarellm_composite")


# --- the benchmark panels ----------------------------------------------------

SUP_W, SUP_H = 183, 185
SUP_LAYOUT = {  # file name: (x, y, w, h) in mm, in panel-letter order
    "benchmark_design": (0, 0, 112, 46),
    "benchmark_expert_agreement": (117, 0, 66, 46),
    "benchmark_prompt_scorecard": (0, 51, 104, 88),
    "benchmark_delivery_matrix": (109, 51, 74, 76),
    "benchmark_confidence_by_rubric": (0, 141, 112, 44),
    "benchmark_confidence_vs_phenotype": (117, 141, 66, 44),
}


def _waffle(ax, x, y, n, cols, s=0.9, gap=0.28):
    for i in range(n):
        r, q = divmod(i, cols)
        ax.add_patch(
            Rectangle((x + q * (s + gap), y + r * (s + gap)), s, s, fc="#CFCFCF", ec="none")
        )
    return -(-n // cols) * (s + gap)


def panel_design(fr: Frame, state: dict):
    """What is scored, by whom, on what."""
    ax = fr.canvas()
    diag = next(iter(state["diagnostics"].values()))
    conc = state["reviewer_concordance"]
    decoys = next(iter(state["decoys"].values()))
    cells = next(iter(state["cells"].values()))
    ax.text(0, 5, f"scored clusters · {diag['n_scored']} genes", fontsize=T6, va="top")
    y = 9.5
    for name, rec in diag["per_cluster"].items():
        screen, cid = name.split("/")[0].split("_")[0], name.split("/")[-1]
        ax.text(0, y + 0.5, f"{screen} #{cid}", fontsize=T5, va="center", color=INK2)
        ax.text(0, y + 2.6, f"{rec['n']} genes", fontsize=T5, va="center", color=INK3)
        y += max(_waffle(ax, 17, y, rec["n"], cols=16), 3.2) + 1.8
    ax.text(0, y + 1.5, "control clusters", fontsize=T6, va="top")
    y += 5.8
    for d in decoys:
        name = d["screen"].replace("aconcagua_interphase_shuffled", "shuffled").split("_")[0]
        exp = "expect Low or none" if d["expectation"] == "abstain" else "expect a pathway"
        ax.text(0, y, f"{name} #{d['cluster']}", fontsize=T5, va="center", color=INK2)
        ax.text(17, y, exp, fontsize=T5, va="center", color=INK3)
        y += 2.8

    ex = 50
    ax.text(ex, 5, f"{len(conc['reviewers'])} lab experts, blind to source", fontsize=T6,
            va="top", ha="center")  # fmt: skip
    for i, lab in enumerate("ABC"[: len(conc["reviewers"])]):
        cy = 15 + i * 7
        ax.add_patch(Circle((ex - 6, cy), 2.3, fc="white", ec=GOLD, lw=0.7))
        ax.text(ex - 6, cy, lab, fontsize=T6, ha="center", va="center", color=GOLD)
        ax.plot([ex - 3.5, ex + 3], [cy, 22], color=INK3, lw=LEAD)
    ax.add_patch(Rectangle((ex + 3, 18.2), 12.5, 7.6, fc=GOLD, ec="none", alpha=0.18))
    ax.text(ex + 9.25, 22, "consensus\n2 of 3", fontsize=T5, ha="center", va="center",
            linespacing=1.1)  # fmt: skip
    arrow(ax, (ex + 16.5, 22), (ex + 21.5, 22))

    sx = 74
    ax.text(sx, 5, "each build scored on", fontsize=T6, va="top")
    items = [
        ("category recall", f"every gene ({diag['n_scored']})"),
        ("novel-role sub-class", f"novel-role genes ({cells['novel_subclass'][1]})"),
        ("uncharacterized sub-class", f"uncharacterized genes ({cells['unchar_subclass'][1]})"),
        ("pathway coherence", f"one call per cluster ({cells['coherence'][1]})"),
        ("controls passed", f"control clusters ({len(decoys)})"),
    ]
    for i, (k, v) in enumerate(items):
        ax.text(sx, 10 + i * 6.2, k, fontsize=T6, va="top")
        ax.text(sx, 12.7 + i * 6.2, v, fontsize=T5, va="top", color=INK3)


def panel_agreement(fr: Frame, conc: dict):
    """Pairwise and three-way expert agreement."""
    cv = fr.canvas()
    cv.text(0, 5, "how often the experts agree", fontsize=T6, va="top")
    cv.text(fr.w, 5, f"Fleiss κ = {conc['fleiss_kappa']:.2f}", fontsize=T5, va="top", ha="right",
            color=INK2)  # fmt: skip
    ax = fr.axes(22, 10, fr.w - 23, fr.h - 16)
    blind = {r: chr(65 + i) for i, r in enumerate(conc["reviewers"])}
    pairs = [("–".join(blind[r] for r in kk.split("-")), v) for kk, v in conc["pairwise"].items()]
    groups = [
        ("expert pairs", pairs, GOLD),
        ("all three", [(n, conc["by_class"][kk]["frac"]) for kk, n in CLASSES], INK2),
    ]
    y = 0
    for head, items, col in groups:
        ax.text(0.495, y, head, fontsize=T5, ha="right", va="center", color=INK1)
        y += 1
        for name, v in items:
            ax.plot([0.5, v], [y, y], color=RULE, lw=0.6, zorder=1)
            ax.scatter([v], [y], s=9, color=col, zorder=2, lw=0)
            ax.text(0.495, y, name, fontsize=T5, ha="right", va="center", color=INK2)
            y += 1
        y += 0.5
    ax.set_ylim(y - 0.9, -0.7)
    ax.set_xlim(0.5, 1.0)
    ax.set_xticks([0.5, 0.75, 1.0], [".5", ".75", "1"])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("fraction of genes")


def panel_scorecard(fr: Frame, steps: list[dict]):
    """Each step's kept build and its change against the step's own same-batch baseline."""
    ax = fr.canvas()
    heads = ["category\nrecall", "novel-role\nsub-class", "unchar.\nsub-class",
             "pathway\ncoherence", "controls\npassed"]  # fmt: skip
    lab_w, cw, rh, top = 38, 13.2, 7.0, 13
    ax.text(0, 5, "what each step of building the prompt changed", fontsize=T6, va="top")
    for j, hd in enumerate(heads):
        ax.text(lab_w + j * cw + cw / 2, top - 1, hd, fontsize=T5, ha="center", va="bottom",
                linespacing=1.05)  # fmt: skip
    for i, st in enumerate(steps):
        y = top + i * rh
        ax.text(0, y + 2.6, st["label"], fontsize=T6, va="center", color=STEP_COLORS[st["group"]])
        kept = KEPT_LABELS.get(st["kept"], st["kept"].replace("_", " "))
        ax.text(0, y + 5.0, f"kept: {kept}", fontsize=T5, va="center", color=INK3)
        for j, (m, _) in enumerate(CRITERIA):
            (bv, bn, bd), (av, an, ad) = st["before"][m], st["after"][m]
            delta = an - bn if bd == ad else round((av - bv) * ad)
            shift = min(abs(av - bv) / 0.2, 1.0)
            fc = "#F4F4F4"
            if delta > 0:
                fc = (0.71, 0.27, 0.11, 0.12 + 0.73 * shift)
            elif delta < 0:
                fc = (0.24, 0.42, 0.56, 0.12 + 0.55 * shift)
            x = lab_w + j * cw
            ax.add_patch(Rectangle((x + 0.3, y + 0.3), cw - 0.6, rh - 0.6, fc=fc, ec="none"))
            dark = delta > 0 and shift > 0.6
            ax.text(x + cw / 2, y + 2.7, f"{an}/{ad}", fontsize=T6, ha="center", va="center",
                    color="white" if dark else INK1)  # fmt: skip
            ax.text(x + cw / 2, y + 5.1, "±0" if delta == 0 else f"{delta:+d}", fontsize=T5,
                    ha="center", va="center", color="white" if dark else INK2)  # fmt: skip
            if m == st["goal"]:
                ax.add_patch(Rectangle((x + 0.3, y + 0.3), cw - 0.6, rh - 0.6, fc="none", ec=INK1,
                                       lw=0.8))  # fmt: skip
    y = top + len(steps) * rh + 2
    ax.text(0, y, "each cell: the kept build's score and its change against the step's own "
            "baseline, run in the same batch", fontsize=T5, va="top", color=INK3)  # fmt: skip
    ax.text(0, y + 2.6, "shade: size of the change as a fraction (full at 20 points); outlined: "
            "the criterion the step targeted", fontsize=T5, va="top", color=INK3)  # fmt: skip


def panel_delivery(fr: Frame, rows: list[dict], gate: dict):
    """Every delivery x literature-step build on four criteria, the kept build outlined."""
    ax = fr.canvas()
    heads = ["category\nrecall", "novel-role\nsub-class", "unchar.\nsub-class",
             "pathway\ncoherence"]  # fmt: skip
    cm = LinearSegmentedColormap.from_list("delivery", ["#F4F6F8", BLUE])
    lab_w, cw, rh, top = 34, 8.4, 5.2, 13
    ax.text(0, 5, "delivery and literature step", fontsize=T6, va="top")
    ax.text(lab_w - 11, top - 1, "PubMed", fontsize=T5, va="bottom", color=INK3)
    for j, hd in enumerate(heads):
        ax.text(lab_w + j * cw + cw / 2, top - 1, hd, fontsize=T5, ha="center", va="bottom",
                linespacing=1.05)  # fmt: skip
    i = 0
    for d, dn in DELIVERIES:
        ax.text(0, top + (i + 1.5) * rh, dn, fontsize=T6, va="center", color=BLUE)
        for s, sn in LIT_SETTINGS:
            arm, y = d + s, top + i * rh
            ax.text(lab_w - 11, y + rh / 2, sn, fontsize=T5, va="center", color=INK2)
            for j, (m, _) in enumerate(CRITERIA[:4]):
                r = next(r for r in rows if r["condition"] == arm and r["metric"] == m)
                lo = 0.85 if m == "category" else 0.3
                v = float(np.clip((r["value"] - lo) / (1 - lo), 0.0, 1.0))
                x = lab_w + j * cw
                ax.add_patch(
                    Rectangle((x + 0.25, y + 0.25), cw - 0.5, rh - 0.5, fc=cm(v), ec="none")
                )
                if m in ("category", "novel_subclass"):
                    t = f"{r['value']:.2f}".lstrip("0")
                else:
                    t = f"{r['numerator']}/{r['denominator']}"
                ax.text(x + cw / 2, y + rh / 2, t, fontsize=T5, ha="center", va="center",
                        color="white" if v > 0.6 else INK1)  # fmt: skip
            if arm in (gate.get("to"), gate.get("overrode")):
                kept = arm == gate["to"]
                ax.add_patch(
                    Rectangle((lab_w + 0.2, y + 0.2), 4 * cw - 0.4, rh - 0.4, fc="none",
                              ec=RUST if kept else INK1, lw=0.9, ls="-" if kept else (0, (2, 1.2)))
                )  # fmt: skip
            i += 1
    y = top + i * rh + 2
    ax.text(0, y, "rust: kept · dashed: automatic pick, overridden", fontsize=T5, va="top",
            color=INK3)  # fmt: skip
    ax.text(0, y + 2.6, "shading is relative within each column", fontsize=T5, va="top",
            color=INK3)  # fmt: skip


def panel_rubrics(fr: Frame, conf: dict[str, dict[str, int]], n_clusters: int):
    """Confidence calls on one full screen under each rubric."""
    cv = fr.canvas()
    cv.text(0, 5, f"confidence calls on one full screen (aconcagua, {n_clusters} clusters)",
            fontsize=T6, va="top")  # fmt: skip
    swatches(cv, 24, 10, [(CONF[lv], LEVEL_LABEL[lv]) for lv in LEVELS], gap=3.5)
    ax = fr.axes(24, 13, fr.w - 25, fr.h - 22)
    for y, (arm, name) in enumerate(SPREAD_ARMS):
        left = 0
        for lv in LEVELS:
            n = conf.get(arm, {}).get(lv, 0)
            ax.barh(y, n, left=left, height=0.68, color=CONF[lv], ec="white", lw=0.4)
            if n >= 12:
                ax.text(left + n / 2, y, str(n), fontsize=T5, ha="center", va="center",
                        color="white" if lv in ("High", "Medium") else INK1)  # fmt: skip
            left += n
        ax.text(-5, y, name, fontsize=T5, ha="right", va="center",
                color=RUST if arm == SHIPPED_ARM else INK2)  # fmt: skip
    ax.set_ylim(len(SPREAD_ARMS) - 0.5, -0.5)
    ax.set_xlim(0, n_clusters)
    ax.set_xticks([t for t in (0, 100, 200) if t <= n_clusters])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("clusters")


def panel_strength(fr: Frame, p, c):
    """Cluster median phenotype strength by confidence call (features were not shown)."""
    cv = fr.canvas()
    cv.text(0, 5, "confidence tracks phenotype strength", fontsize=T6, va="top")
    ax = fr.axes(11, 14, fr.w - 12, fr.h - 22)
    per = p.groupby("cluster").auc.median().to_frame("auc").join(c.level)
    rng = np.random.default_rng(0)
    for i, lv in enumerate(LEVELS):
        v = per[per.level == lv].auc.to_numpy()
        ax.scatter(i + rng.uniform(-0.22, 0.22, len(v)), v, s=3, color=CONF[lv], lw=0, zorder=2)
        if len(v):
            ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color=INK1, lw=0.7, zorder=3)
    ns = per.level.value_counts()
    ax.set_xticks(range(len(LEVELS)), [f"{lv}\n{ns.get(lv, 0)}" for lv in LEVELS])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(LEVELS) - 0.4)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_ylabel("phenotype strength\n(cluster median AUC)", linespacing=1.1)
    pval = mann_whitney_greater(per[per.level == "High"].auc, per[per.level != "High"].auc)
    exp = int(np.floor(np.log10(pval)))
    base = f"High vs rest, one-sided Mann–Whitney: P = {pval / 10**exp:.1f} × 10"
    cv.text(0, 8.6, base, fontsize=T5, va="top", color=INK2)
    cv.text(text_w(cv, base, T5) + 0.1, 7.9, f"−{-exp}", fontsize=T5, va="top", color=INK2)


def render_benchmark(out: Path, data: Path, mode: dict, screen=None) -> list[Path]:
    draws = {
        "benchmark_design": lambda fr: panel_design(fr, mode),
        "benchmark_expert_agreement": lambda fr: panel_agreement(fr, mode["reviewer_concordance"]),
        "benchmark_prompt_scorecard": lambda fr: panel_scorecard(fr, step_table(data)),
        "benchmark_delivery_matrix": lambda fr: panel_delivery(
            fr,
            [r for r in read_rows(data / "mode.csv") if r["panel"] == "criteria"],
            mode.get("human_gate") or {"to": mode["winner_condition"].split("__")[0]},
        ),  # fmt: skip
    }
    if screen is not None:
        p, c, _ = screen
        conf: dict[str, dict[str, int]] = {}
        with (data / "screen_confidence.csv").open() as fh:
            for r in csv.DictReader(fh):
                conf.setdefault(r["arm"], {})[r["level"]] = int(r["n"])
        draws["benchmark_confidence_by_rubric"] = lambda fr: panel_rubrics(fr, conf, len(c))
        draws["benchmark_confidence_vs_phenotype"] = lambda fr: panel_strength(fr, p, c)
    written = []
    for name, draw in draws.items():
        _, _, w, h = SUP_LAYOUT[name]
        fig = page(w, h)
        draw(Frame(fig, 0, 0, w, h))
        written += save(fig, out / name)
    fig = page(SUP_W, SUP_H)
    for (name, draw), letter in zip(draws.items(), "abcdef", strict=False):
        fr = Frame(fig, *SUP_LAYOUT[name])
        draw(fr)
        fr.letter(letter)
    return written + save(fig, out / "benchmark_composite")


# --- driver ------------------------------------------------------------------


def _load(experiment: str, outputs: Path = OUTPUTS) -> dict | None:
    path = outputs / experiment / f"{experiment}_state.json"
    return json.loads(path.read_text()) if path.exists() else None


def write_tables(states: dict, data: Path) -> list[Path]:
    """The tidy per-experiment tables every benchmark panel reads."""
    data.mkdir(parents=True, exist_ok=True)
    written = []
    for name in ("source", "mode", "order", "features"):
        if states.get(name):
            written.append(write_rows(rows_from_cells(name, states[name]), data / f"{name}.csv"))
    if states.get("walkup") and states["walkup"].get("stages"):
        written.append(write_rows(rows_from_walkup(states["walkup"]), data / "walkup.csv"))
    return written


def render_all(out: Path, outputs: Path = OUTPUTS) -> list[Path]:
    style()
    data = out / "data"
    states = {e: _load(e, outputs) for e in ("source", "walkup", "mode", "order", "features")}
    written = write_tables(states, data)
    screen = load_screen(data) if (data / "screen_map.csv").exists() else None
    if all(states[e] for e in ("source", "walkup", "mode", "order")):
        written += render_benchmark(out / "benchmark", data, states["mode"], screen)
    if screen is not None:
        for k in EXAMPLES:
            written += render_mozzarellm(out / f"mozzarellm_cluster_{k}", *screen, k)
    for p in written:
        print(f"[figures] {p}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=OUTPUTS / "figures")
    ap.add_argument(
        "--collect-aconcagua",
        nargs=3,
        type=Path,
        metavar=("RUNS_DIR", "LEGACY_DIR", "PHATE_TSV"),
        help="flatten a screen run into figures/data/screen_*.csv, then render",
    )
    args = ap.parse_args()
    if args.collect_aconcagua:
        for p in collect_aconcagua(*args.collect_aconcagua, args.out / "data"):
            print(f"[figures] {p}")
    render_all(args.out)


if __name__ == "__main__":
    main()
