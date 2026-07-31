"""Shared helpers for the three-step benchmark pipeline.

The pipeline is source -> walkup -> mode, one script per step
(run_source.py / run_walkup.py / run_mode.py). Each step runs conditions
through the same RunSpec + _run_benchmark_loop machinery, scores real clusters
against reviewer-consensus GT, validates the 3 decoys, and writes a state file
with a shared envelope (see write_state) so the next step can pick it up.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from architecture_benchmarking_workflow.bench_evaluator import (
    N_REAL_GENES,
    MetricPanel,
    build_consensus_gt,
    consensus_coherence,
    load_consensus_gt,
    score_decoys,
    score_run,
)

PHASE1_DIR = Path(__file__).resolve().parents[1]
INPUTS_DIR = PHASE1_DIR / "benchmark_inputs"
GT_DIR = INPUTS_DIR / "ground_truth"
CONFIGS = PHASE1_DIR / "configs"
OUTPUTS = PHASE1_DIR / "benchmarking_outputs"
CLUSTERS_ALL = INPUTS_DIR / "benchmark_input.csv"
SURVEY_KEY = GT_DIR / "survey_key.csv"
GT_PATH = OUTPUTS / "consensus_gt.csv"

SOURCES = ("uniprot", "affinage", "both")
MODES = ("single_call", "cot", "stepwise")
REVIEWERS = ("eric", "liz", "iain")

# Source/mode selection primary is coverage-weighted (correct categories / 133),
# so a cell can't win by dropping hard genes to inflate raw category. Coverage
# joins the four quality metrics as a domination axis (a full-coverage cell can
# dominate a coverage-collapsed one). The walkup uses its own guarded rule.
PRIMARY = "coverage_weighted_category"
HOLISTIC_METRICS = ["category", "novel_subclass", "unchar_subclass", "coherence", "coverage"]

# Negative-control decoys: nonsense/control clusters must abstain (Low), the
# large coherent cluster must stay functional (valid output, no truncation).
DECOY_SPECS = {
    ("aconcagua_interphase_shuffled", "17"): "abstain",
    ("whitney", "49"): "abstain",
    ("jebel", "0"): "functional",
}


def metric_value(panel: MetricPanel, name: str) -> float:
    """Return a scalar for any MetricPanel field: correct/n for tuple metrics.

    "coverage" is the fraction of real genes that received a category vote
    (panel.n / N_REAL_GENES) -- used as a guard so a stage can't ratchet a goal
    metric up by dropping genes. "coverage_weighted_category" is the honest recall
    over all real genes (correct categories / N_REAL_GENES = category x coverage)
    -- the source/mode selection primary, so a coverage-collapsing cell can't win
    by dropping hard genes and inflating raw category on the survivors.
    """
    if name == "category":
        return panel.category
    if name == "coverage":
        return panel.n / N_REAL_GENES
    if name == "coverage_weighted_category":
        return panel.category * panel.n / N_REAL_GENES
    correct, n = getattr(panel, name)
    return correct / n if n else 0.0


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


def reviewer_csvs() -> dict[str, Path]:
    return {r: GT_DIR / f"annotation_{r}.csv" for r in REVIEWERS}


# A real cluster whose consensus coherence is Low has no coherent pathway; the
# correct behaviour is to abstain (Low pathway_confidence, no gene calls), so its
# genes leave the per-gene scored set and it is validated for abstention only.
ABSTAIN_COHERENCE = "Low"


def _apply_coherence_abstain(gt: dict, coh: dict) -> dict:
    """Reassign real clusters with Low consensus coherence to the 'abstain' role."""
    for (screen, cluster, _gene), row in gt.items():
        if row.get("cluster_role") == "real" and coh.get((screen, cluster)) == ABSTAIN_COHERENCE:
            row["cluster_role"] = "abstain"
    return gt


def abstain_clusters(gt: dict) -> list[tuple[str, str]]:
    """(screen, cluster) of the Low-coherence real clusters now scored abstain-only."""
    seen = {(s, c) for (s, c, _g), r in gt.items() if r.get("cluster_role") == "abstain"}
    return sorted(seen)


def validation_specs(gt: dict) -> dict[tuple[str, str], str]:
    """All control clusters to validate: fixed decoys + coherence-derived abstains.

    The 3 fixed decoys (2 abstain, 1 functional) plus every Low-coherence real
    cluster (abstain). This is the full set the model must handle as controls.
    """
    specs = dict(DECOY_SPECS)
    for key in abstain_clusters(gt):
        specs[key] = "abstain"
    return specs


def load_gt_and_coherence():
    """Build the consensus GT + per-cluster coherence; demote Low-coherence clusters.

    Low-coherence (no-coherent-pathway) real clusters are reassigned to the
    'abstain' role so every downstream scorer (which keeps cluster_role=='real')
    excludes their genes from per-gene metrics; they are validated for abstention.
    """
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    build_consensus_gt(reviewer_csvs(), SURVEY_KEY, [], GT_PATH)
    gt, coh = load_consensus_gt(GT_PATH), consensus_coherence(reviewer_csvs())
    return _apply_coherence_abstain(gt, coh), coh


def run_stamp() -> str:
    """Timestamp baked into each run's dir name so runs archive in place and
    previous runs are never overwritten. Computed once per run by the caller --
    unlike ``experiment_output_dir``, which recomputes on every access."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def latest_run_dir(step: str, label: str) -> Path | None:
    """Newest archived run dir under OUTPUTS/<step>/<label>_<stamp>/, or None.
    Used by --score-only to locate the most recent run to re-score."""
    base = OUTPUTS / step
    runs = sorted(base.glob(f"{label}_*")) if base.exists() else []
    return runs[-1] if runs else None


def prepare(config, experiment_id: str, dry_run: bool, out_root: Path = OUTPUTS):
    """Point a config at out_root/<experiment_id>, set dry-run.

    experiment_id carries a run stamp (e.g. ``affinage_20260731_101500``): with
    overwrite_outputs the resolved dir is stable across accesses AND unique per
    run, so previous runs archive in place and nothing is wiped."""
    config.experiment_id = experiment_id
    config.paths.output_dir = out_root
    config.run.overwrite_outputs = True
    config.run.dry_run = dry_run
    return config


def score_cell(run_dir: Path, gt: dict, coh: dict, condition: str) -> MetricPanel:
    """Score one condition in isolation (exact route/condition match)."""
    return score_run(run_dir, gt, cluster_coherence=coh, route_equals=condition)


def expected_gene_counts() -> dict[tuple, int]:
    """{(screen, cluster): gene count} from the benchmark input, for decoy completion."""
    counts: dict[tuple, int] = {}
    with open(CLUSTERS_ALL, newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["screen_name"].strip(), str(row["cluster_id"]).strip())
            counts[key] = counts.get(key, 0) + 1
    return counts


def decoy_results(
    run_dir: Path, condition: str | None = None, specs: dict | None = None
) -> list[dict]:
    """Validate control clusters in a run dir (optionally isolating one condition).

    specs defaults to the 3 fixed decoys; pass validation_specs(gt) to also validate
    the coherence-derived abstain clusters (e.g. denali/24). Reports output
    completeness (genes classified vs expected) alongside the pass/fail verdict.
    """
    return [
        {
            "screen": r.screen,
            "cluster": r.cluster,
            "expectation": r.expectation,
            "reps": r.reps,
            "failures": r.failures,
            "modal_confidence": r.modal_confidence,
            "passed": r.passed,
            "genes_per_rep": r.genes_per_rep,
            "median_genes": r.median_genes,
            "expected_genes": r.expected_genes,
            "completion": r.completion,
        }
        for r in score_decoys(
            run_dir, specs or DECOY_SPECS, route_equals=condition,
            expected_counts=expected_gene_counts(),
        )
    ]


def panel_json(p: MetricPanel) -> dict:
    return {
        "category": p.category,
        "novel_subclass": list(p.novel_subclass),
        "unchar_subclass": list(p.unchar_subclass),
        "coherence": list(p.coherence),
        "coverage": round(metric_value(p, "coverage"), 4),
        "n": p.n,
        "failures": p.failures,
    }


def print_panel(label: str, p: MetricPanel) -> None:
    cw = metric_value(p, PRIMARY)
    quality = " ".join(
        f"{m}={metric_value(p, m):.3f}"
        for m in ("category", "novel_subclass", "unchar_subclass", "coherence")
    )
    print(f"  {label:<40} cw-recall={cw:.3f}  {quality}  cov={p.n}/{N_REAL_GENES} fail={p.failures}")


def write_state(path: Path, step: str, source: str, winner, decoys, carry: dict, **extra) -> None:
    """Write a step state file with the shared envelope.

    step   -- "source" | "walkup" | "mode"
    source -- the bundle source the step ran on
    winner -- the selected condition/build for this step
    carry  -- what the next step consumes (e.g. {"source": ...})
    extra  -- step-specific detail (cells, stages, dominated, ...)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "source": source,
        "winner": winner,
        "decoys": decoys,
        "carry": carry,
        **extra,
    }
    path.write_text(json.dumps(state, indent=2))
    print(f"\nstate -> {path}")


def read_carry(step: str) -> dict:
    """Return the `carry` block from a prior step's state file (or {} if absent)."""
    path = OUTPUTS / step / f"{step}_state.json"
    if path.exists():
        return json.loads(path.read_text()).get("carry", {})
    return {}
