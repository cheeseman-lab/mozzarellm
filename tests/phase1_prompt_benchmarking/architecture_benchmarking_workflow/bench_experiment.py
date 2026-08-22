"""Experiment orchestrator -- one function, one yaml per experiment.

``run_experiment`` reads an experiment yaml (shared model + run regime, plus the
conditions that vary), drives every condition through the engine's RunSpec +
``_run_benchmark_loop`` machinery, scores real clusters against the
reviewer-consensus ground truth, validates the control clusters, applies the
yaml's selection rule, and writes the experiment's state file --
``benchmarking_outputs/<experiment>/<experiment>_state.json``. State files are
the only metric output; downstream steps read state, never re-derive.

Runs archive under ``benchmarking_outputs/<experiment>/<condition>_<stamp>/``
and are never overwritten. ``score_only`` re-scores the newest archived run dir
per condition without API calls; ``dry_run`` exercises the full plumbing on
mock outputs.

``stages:`` and ``uses:`` are reserved schema keys for staged experiments (the
walkup); the loader rejects them until that PR lands.

Usage:
    python -m tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_experiment \
        tests/phase1_prompt_benchmarking/experiments/source.yaml [--dry-run | --score-only]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import yaml

from .bench_configparse import BenchmarkConfig, ModelConfig, PathsConfig, RunConfig
from .bench_evaluator import (
    N_REAL_GENES,
    MetricPanel,
    build_consensus_gt,
    consensus_coherence,
    inter_reviewer_concordance,
    load_consensus_gt,
    pathway_diagnostics,
    reviewer_label_sets,
    score_decoys,
    score_run,
    source_diagnostics,
    source_preference_tally,
)
from .bench_evaluator import audit_flag_diagnostics as _audit_flag_diagnostics
from .bench_orchestrator import RunSpec, _build_config_snapshot, _run_benchmark_loop
from .bench_routes import ROUTE_REGISTRY

PHASE1_DIR = Path(__file__).resolve().parents[1]
INPUTS_DIR = PHASE1_DIR / "benchmark_inputs"
GT_DIR = INPUTS_DIR / "ground_truth"
BUNDLES_DIR = PHASE1_DIR / "benchmark_bundles"
OUTPUTS = PHASE1_DIR / "benchmarking_outputs"
CLUSTERS_ALL = INPUTS_DIR / "benchmark_input.csv"
SURVEY_KEY = GT_DIR / "survey_key.csv"
GT_PATH = OUTPUTS / "consensus_gt.csv"

REVIEWERS = ("eric", "liz", "iain")

# Selection primaries/guards resolvable from a MetricPanel (see metric_value).
SELECTION_METRICS = (
    "category",
    "novel_subclass",
    "unchar_subclass",
    "coherence",
    "coverage",
    "coverage_weighted_category",
)

# Negative-control decoys: nonsense/control clusters must abstain (Low), the
# large coherent cluster must stay functional (valid output, no truncation).
DECOY_SPECS = {
    ("aconcagua_interphase_shuffled", "17"): "abstain",
    ("whitney", "49"): "abstain",
    ("jebel", "0"): "functional",
}

# A real cluster whose consensus coherence is Low has no coherent pathway; the
# correct behaviour is to abstain (Low pathway_confidence, no gene calls), so its
# genes leave the per-gene scored set and it is validated for abstention only.
ABSTAIN_COHERENCE = "Low"


# =============================================================================
# EXPERIMENT YAML
# =============================================================================

# Keys a condition may set; anything a condition sets overrides the shared
# ``run:`` block for that condition only.
_CONDITION_KEYS = {"name", "bundle_source", "route", "component_overrides"}
_RESERVED_KEYS = ("stages", "uses")


def load_experiment(yaml_path: Path) -> dict:
    """Load and validate an experiment yaml; raise ValueError on schema errors."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Experiment yaml not found: {yaml_path}")
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    for key in _RESERVED_KEYS:
        if key in raw:
            raise ValueError(
                f"{yaml_path.name}: '{key}:' is reserved for staged experiments, "
                "which arrive with the walkup PR"
            )
    for key in ("experiment", "model", "run", "conditions", "selection"):
        if key not in raw:
            raise ValueError(f"{yaml_path.name}: missing required key '{key}:'")

    route = raw["run"].get("route")
    if route not in ROUTE_REGISTRY:
        raise ValueError(
            f"{yaml_path.name}: run.route {route!r} not in registry "
            f"{sorted(ROUTE_REGISTRY)}"
        )

    names = []
    for cond in raw["conditions"]:
        unknown = set(cond) - _CONDITION_KEYS
        if unknown:
            raise ValueError(
                f"{yaml_path.name}: condition {cond.get('name')!r} has unknown "
                f"key(s) {sorted(unknown)}; allowed: {sorted(_CONDITION_KEYS)}"
            )
        cond_route = cond.get("route", route)
        if cond_route not in ROUTE_REGISTRY:
            raise ValueError(
                f"{yaml_path.name}: condition {cond.get('name')!r} route "
                f"{cond_route!r} not in registry"
            )
        names.append(cond["name"])
    if len(names) != len(set(names)):
        raise ValueError(f"{yaml_path.name}: duplicate condition names in {names}")

    selection = raw["selection"]
    for metric in [selection["primary"], *selection.get("metrics", [])]:
        if metric not in SELECTION_METRICS:
            raise ValueError(
                f"{yaml_path.name}: unknown selection metric {metric!r}; "
                f"allowed: {list(SELECTION_METRICS)}"
            )
    return raw


# =============================================================================
# SELECTION
# =============================================================================


def metric_value(panel: MetricPanel, name: str) -> float:
    """Return a scalar for any MetricPanel field: correct/n for tuple metrics.

    "coverage" is the fraction of real genes that received a category vote
    (panel.n / N_REAL_GENES) -- used as a guard so a condition can't ratchet a
    goal metric up by dropping genes. "coverage_weighted_category" is the honest
    recall over all real genes (correct categories / N_REAL_GENES = category x
    coverage) -- the selection primary, so a coverage-collapsing condition can't
    win by dropping hard genes and inflating raw category on the survivors.
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
    """Holistic pick: best ``primary`` among non-dominated cells.

    A cell is dominated when another is >= on every metric and strictly > on at
    least one; dominated cells drop, then the highest-``primary`` survivor wins
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


# =============================================================================
# GROUND TRUTH + CONTROLS
# =============================================================================


def reviewer_csvs() -> dict[str, Path]:
    return {r: GT_DIR / f"annotation_{r}.csv" for r in REVIEWERS}


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


def expected_gene_counts() -> dict[tuple, int]:
    """{(screen, cluster): gene count} from the benchmark input, for decoy completion."""
    counts: dict[tuple, int] = {}
    with open(CLUSTERS_ALL, newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["screen_name"].strip(), str(row["cluster_id"]).strip())
            counts[key] = counts.get(key, 0) + 1
    return counts


def decoy_results(run_dir: Path, condition: str, specs: dict) -> list[dict]:
    """Validate control clusters in a run dir, isolating one condition.

    Reports output completeness (genes classified vs expected) alongside the
    pass/fail verdict.
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
            run_dir, specs, route_equals=condition, expected_counts=expected_gene_counts()
        )
    ]


# =============================================================================
# RUN DIRS + STATE
# =============================================================================


def latest_run_dir(experiment: str, condition: str) -> Path | None:
    """Newest archived run dir OUTPUTS/<experiment>/<condition>_<stamp>/, or None.

    Matches the stamp exactly so one condition name that prefixes another
    (uniprot vs uniprot_backfill) never picks up the other's runs.
    """
    base = OUTPUTS / experiment
    pattern = re.compile(re.escape(condition) + r"_\d{8}_\d{6}")
    if not base.exists():
        return None
    runs = sorted(p for p in base.glob(f"{condition}_*") if pattern.fullmatch(p.name))
    return runs[-1] if runs else None


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


def _condition_config(exp: dict, cond: dict, stamp: str, dry_run: bool) -> BenchmarkConfig:
    """BenchmarkConfig for one condition: shared blocks + the condition's overrides.

    experiment_id carries the run stamp: with overwrite_outputs the resolved dir
    (OUTPUTS/<experiment>/<condition>_<stamp>) is stable across accesses AND
    unique per run, so previous runs archive in place and nothing is wiped.
    """
    model, run = exp["model"], exp["run"]
    cfg = BenchmarkConfig()
    cfg.experiment_id = f"{cond['name']}_{stamp}"
    cfg.model = ModelConfig(
        provider=model.get("provider", cfg.model.provider),
        model_name=model["model_name"],
        temperature=model.get("temperature", cfg.model.temperature),
        max_tokens=model.get("max_tokens", cfg.model.max_tokens),
        top_p=model.get("top_p"),
        top_k=model.get("top_k"),
        thinking=model.get("thinking"),
    )
    cfg.run = RunConfig(
        num_replicates=run.get("replicates", cfg.run.num_replicates),
        max_workers=run.get("max_workers", cfg.run.max_workers),
        dry_run=dry_run,
        overwrite_outputs=True,
    )
    cfg.paths = PathsConfig(
        benchmark_inputs_dir=INPUTS_DIR,
        benchmark_clusters_csv=CLUSTERS_ALL,
        evidence_bundles_dir=BUNDLES_DIR,
        output_dir=OUTPUTS / exp["experiment"],
        bundle_source=cond["bundle_source"],
    )
    return cfg


# =============================================================================
# ENTRY POINT
# =============================================================================


def run_experiment(
    yaml_path: Path,
    *,
    stage: str | None = None,
    select: tuple[str, str] | None = None,
    dry_run: bool = False,
    score_only: bool = False,
) -> dict:
    """Run every condition an experiment yaml declares, score, select, write state.

    stage/select serve staged experiments (walkup PR); on a stage-less
    experiment they raise cleanly. score_only re-scores the newest archived run
    dir per condition without API calls and rewrites state.
    """
    exp = load_experiment(yaml_path)
    name = exp["experiment"]
    if stage is not None or select is not None:
        raise ValueError(
            f"experiment {name!r} declares no stages; stage/select apply to "
            "staged experiments only (walkup PR)"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gt, coh = load_gt_and_coherence()
    reviewer_labels = reviewer_label_sets(reviewer_csvs())
    controls = validation_specs(gt)
    run_block = exp["run"]
    shared_overrides = run_block.get("component_overrides") or {}

    cells: dict[str, MetricPanel] = {}
    decoys, diagnostics, pathway, audit_flags, run_dirs = {}, {}, {}, {}, {}
    for cond in exp["conditions"]:
        cond_name = cond["name"]
        route_name = cond.get("route", run_block["route"])
        condition = f"{cond_name}__{route_name}"

        if score_only:
            out = latest_run_dir(name, cond_name)
            if out is None:
                raise FileNotFoundError(
                    f"score_only: no archived run dir {OUTPUTS / name}/{cond_name}_<stamp>/"
                )
        else:
            cfg = _condition_config(exp, cond, stamp, dry_run)
            overrides = {**shared_overrides, **(cond.get("component_overrides") or {})}
            specs = [
                RunSpec(
                    route=ROUTE_REGISTRY[route_name],
                    condition_name=condition,
                    component_overrides=overrides,
                )
            ]
            snapshot = _build_config_snapshot(
                cfg,
                experiment={
                    "experiment": name,
                    "condition": cond_name,
                    "bundle_source": cond["bundle_source"],
                    "route": route_name,
                    "component_overrides": overrides,
                },
            )
            print(f"\n[{name}] {condition} -> {cfg.experiment_output_dir}")
            _run_benchmark_loop(cfg, specs, snapshot, phase_label=f"{name}:{cond_name}")
            out = cfg.experiment_output_dir

        run_dirs[cond_name] = out.name
        cells[condition] = score_run(out, gt, cluster_coherence=coh, route_equals=condition)
        decoys[cond_name] = decoy_results(out, condition, controls)
        diagnostics[cond_name] = source_diagnostics(out, gt, reviewer_labels, condition)
        audit_flags[cond_name] = {
            f"{screen}/{cluster}/{gene}": report
            for (screen, cluster, gene), report in _audit_flag_diagnostics(
                BUNDLES_DIR, out, route_equals=condition
            ).items()
        }
        # Semantic pathway agreement is skipped in dry-run (mock text, no model load).
        pathway[cond_name] = pathway_diagnostics(
            out, gt, reviewer_csvs(), condition, use_semantic=not dry_run
        )

    selection = exp["selection"]
    winner_cell, dominated = select_holistic(
        cells, selection["primary"], list(selection.get("metrics", []))
    )
    winner = winner_cell.split("__")[0]

    state = {
        "experiment": name,
        "stamp": stamp if not score_only else None,
        "runs": run_dirs,
        "winner": winner_cell,
        "winner_condition": winner,
        "dominated": dominated,
        "selection": selection,
        "carry": dict.fromkeys(exp.get("carry", []), winner),
        "cells": {k: panel_json(v) for k, v in cells.items()},
        "decoys": decoys,
        "diagnostics": diagnostics,
        "audit_flags": audit_flags,
        "pathway": pathway,
        "source_preference": source_preference_tally(gt),
        "reviewer_concordance": inter_reviewer_concordance(reviewer_csvs(), gt),
    }
    state_path = OUTPUTS / name / f"{name}_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2))
    print(f"\nstate -> {state_path}")
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("yaml_path", type=Path, help="experiment yaml (e.g. experiments/source.yaml)")
    ap.add_argument("--dry-run", action="store_true", help="mock outputs, zero API cost")
    ap.add_argument(
        "--score-only", action="store_true", help="re-score newest archived run dirs, no API calls"
    )
    args = ap.parse_args()
    run_experiment(args.yaml_path, dry_run=args.dry_run, score_only=args.score_only)


if __name__ == "__main__":
    main()
