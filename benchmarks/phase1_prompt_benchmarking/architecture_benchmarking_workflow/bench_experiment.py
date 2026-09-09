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

A stage-less experiment may declare ``uses:`` as a mapping of input slots to
carry paths (``source: walkup.carry.source``, ``component_overrides:
walkup.carry.final_component_texts``) -- resolved from the named experiment's
state at invocation, logged, and recorded in the state file, so a downstream
experiment (mode, order) runs on exactly what the upstream one carried.

STAGED experiments (the walkup) replace the static ``conditions:`` with a
``stages:`` block; each stage generates its conditions at runtime -- ``prior``
(the carried build read from the experiment's own state) plus the stage's
candidates. A stage is an INVOCATION, not an experiment: ``stage="CAT"`` runs
exactly one stage, upserts its record into the experiment state, and STOPS --
the human gate. ``select=("CAT", "process_guarded")`` records the choice and
carries the winning text forward; recording the last stage's choice assembles
the final build into ``carry``. ``uses: <experiment>.carry.<key>`` resolves a
cross-experiment input (the walkup's evidence source) from that experiment's
state file; every staged invocation logs and snapshots its resolved inputs.

Usage:
    python -m benchmarks.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_experiment \
        benchmarks/phase1_prompt_benchmarking/experiments/source.yaml [--dry-run | --score-only]
    ... bench_experiment experiments/walkup.yaml --stage CAT [--source affinage]
    ... bench_experiment experiments/walkup.yaml --select CAT process_guarded
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
from .bench_routes import ROUTE_REGISTRY, Route
from .order_bench_orderings import ORDER_VARIANTS, apply_order_variant

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
_CONDITION_KEYS = {"name", "bundle_source", "route", "component_overrides", "order_variant"}
# ``uses: <experiment>.carry.<key>`` -- a cross-experiment input, resolved from
# that experiment's state file at invocation.
_USES_RE = re.compile(r"^(?P<experiment>\w+)\.carry\.(?P<key>\w+)$")


def load_experiment(yaml_path: Path) -> dict:
    """Load and validate an experiment yaml; raise ValueError on schema errors."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Experiment yaml not found: {yaml_path}")
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    staged = "stages" in raw
    if staged and "conditions" in raw:
        raise ValueError(
            f"{yaml_path.name}: an experiment declares 'stages:' or 'conditions:', never both"
        )
    if not staged and "uses" in raw:
        _validate_uses_mapping(yaml_path.name, raw["uses"])
    required = ("experiment", "model", "run") + (
        ("stages",) if staged else ("conditions", "selection")
    )
    for key in required:
        if key not in raw:
            raise ValueError(f"{yaml_path.name}: missing required key '{key}:'")

    route = raw["run"].get("route")
    if route not in ROUTE_REGISTRY:
        raise ValueError(
            f"{yaml_path.name}: run.route {route!r} not in registry "
            f"{sorted(ROUTE_REGISTRY)}"
        )

    if staged:
        _validate_stages(yaml_path.name, raw, route)
        return raw

    uses_source = "source" in (raw.get("uses") or {})
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
        variant = cond.get("order_variant")
        if variant is not None and variant not in ORDER_VARIANTS:
            raise ValueError(
                f"{yaml_path.name}: condition {cond.get('name')!r} order_variant "
                f"{variant!r} not in {sorted(ORDER_VARIANTS)}"
            )
        if uses_source == ("bundle_source" in cond):
            raise ValueError(
                f"{yaml_path.name}: condition {cond.get('name')!r} needs its evidence "
                "source from exactly one place -- its own bundle_source, or the "
                "experiment's uses.source"
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


_USES_SLOTS = ("source", "component_overrides")


def _validate_uses_mapping(yaml_name: str, uses) -> None:
    """A stage-less experiment's uses: maps input slots to carry paths."""
    if not isinstance(uses, dict) or not uses:
        raise ValueError(
            f"{yaml_name}: a stage-less experiment's 'uses:' is a mapping of input "
            f"slot ({'/'.join(_USES_SLOTS)}) -> '<experiment>.carry.<key>'"
        )
    unknown = set(uses) - set(_USES_SLOTS)
    if unknown:
        raise ValueError(
            f"{yaml_name}: unknown uses slot(s) {sorted(unknown)}; allowed: {list(_USES_SLOTS)}"
        )
    for slot, path in uses.items():
        if not isinstance(path, str) or not _USES_RE.match(path):
            raise ValueError(
                f"{yaml_name}: uses.{slot} {path!r} must be '<experiment>.carry.<key>'"
            )


def _validate_stages(yaml_name: str, raw: dict, route: str) -> None:
    """Schema checks for a staged experiment's uses/stages blocks."""
    uses = raw.get("uses")
    if uses is not None and (not isinstance(uses, str) or not _USES_RE.match(uses)):
        raise ValueError(
            f"{yaml_name}: a staged experiment's uses {uses!r} must be "
            "'<experiment>.carry.<key>' (the evidence source)"
        )
    components = []
    for stage in raw["stages"]:
        for key in ("component", "goal", "candidates"):
            if key not in stage:
                raise ValueError(
                    f"{yaml_name}: stage {stage.get('component')!r} missing '{key}'"
                )
        if stage["component"] not in ROUTE_REGISTRY[route].component_order:
            raise ValueError(
                f"{yaml_name}: stage component {stage['component']!r} is not a "
                f"prompt slot of route {route!r}"
            )
        if stage["goal"] not in SELECTION_METRICS:
            raise ValueError(
                f"{yaml_name}: stage {stage['component']!r} goal {stage['goal']!r} "
                f"not in {list(SELECTION_METRICS)}"
            )
        ids = []
        for cand in stage["candidates"]:
            for key in ("id", "rationale", "text"):
                if not cand.get(key):
                    raise ValueError(
                        f"{yaml_name}: stage {stage['component']!r} candidate "
                        f"{cand.get('id')!r} missing '{key}' (every candidate "
                        "carries its rationale)"
                    )
            ids.append(cand["id"])
        if "prior" in ids or len(ids) != len(set(ids)):
            raise ValueError(
                f"{yaml_name}: stage {stage['component']!r} candidate ids must be "
                f"unique and not 'prior'; got {ids}"
            )
        components.append(stage["component"])
    if len(components) != len(set(components)):
        raise ValueError(f"{yaml_name}: duplicate stage components in {components}")


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


def _condition_route(cond: dict, route_name: str) -> Route:
    """The condition's Route: the registry route, reordered when it asks for it."""
    route = ROUTE_REGISTRY[route_name]
    if cond.get("order_variant"):
        route = apply_order_variant(route, cond["order_variant"])
    return route


def _condition_config(
    exp: dict, label: str, bundle_source: str, stamp: str, dry_run: bool
) -> BenchmarkConfig:
    """BenchmarkConfig for one invocation: the yaml's shared blocks + its inputs.

    label is the condition name (stage-less) or the stage component (staged).
    experiment_id carries the run stamp: with overwrite_outputs the resolved dir
    (OUTPUTS/<experiment>/<label>_<stamp>) is stable across accesses AND unique
    per run, so previous runs archive in place and nothing is wiped.
    """
    model, run = exp["model"], exp["run"]
    cfg = BenchmarkConfig()
    cfg.experiment_id = f"{label}_{stamp}"
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
        bundle_source=bundle_source,
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
    source: str | None = None,
) -> dict:
    """Run every condition an experiment yaml declares, score, select, write state.

    Stage-less experiments run all conditions and apply the yaml's selection
    rule. On a staged experiment, ``stage`` runs exactly one stage and stops at
    the human gate; ``select`` records the human's choice for a stage (candidate
    id or "prior"). ``source`` overrides a ``uses:``-resolved evidence source
    (the stale-carry guard) on any experiment that declares one; stage/select
    raise cleanly on a stage-less experiment and vice versa. score_only
    re-scores the newest archived run dir(s) without API calls and rewrites
    state.
    """
    exp = load_experiment(yaml_path)
    name = exp["experiment"]
    if "stages" in exp:
        if select is not None:
            if stage is not None:
                raise ValueError("pass stage= or select=, not both")
            return _record_selection(exp, tuple(select))
        if stage is None:
            raise ValueError(
                f"experiment {name!r} is staged: pass stage=<component> to run "
                "one stage, or select=(stage, choice) to record a gate decision"
            )
        return _run_stage(exp, stage, dry_run=dry_run, score_only=score_only, source=source)
    if stage is not None or select is not None:
        raise ValueError(
            f"experiment {name!r} declares no stages; stage/select apply to "
            "staged experiments only"
        )
    uses = exp.get("uses") or {}
    if source is not None and "source" not in uses:
        raise ValueError(
            f"experiment {name!r} declares no uses.source to override; its "
            "conditions declare bundle_source directly"
        )
    resolved_source: str | None = None
    resolved_overrides: dict = {}
    if "source" in uses:
        resolved_source = source or _resolve_uses(uses["source"])
        if not isinstance(resolved_source, str):
            raise ValueError(f"uses.source {uses['source']!r} resolved to a non-string")
    if "component_overrides" in uses:
        resolved_overrides = _resolve_uses(uses["component_overrides"])
        if not isinstance(resolved_overrides, dict):
            raise ValueError(
                f"uses.component_overrides {uses['component_overrides']!r} resolved "
                "to a non-mapping"
            )
    if uses:
        print(
            f"[{name}] resolved inputs: source={resolved_source!r}, "
            f"carried components: {sorted(resolved_overrides) or 'none'}"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gt, coh = load_gt_and_coherence()
    reviewer_labels = reviewer_label_sets(reviewer_csvs())
    controls = validation_specs(gt)
    run_block = exp["run"]
    shared_overrides = {**resolved_overrides, **(run_block.get("component_overrides") or {})}

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
            bundle_source = cond.get("bundle_source", resolved_source)
            cfg = _condition_config(exp, cond_name, bundle_source, stamp, dry_run)
            overrides = {**shared_overrides, **(cond.get("component_overrides") or {})}
            specs = [
                RunSpec(
                    route=_condition_route(cond, route_name),
                    condition_name=condition,
                    component_overrides=overrides,
                )
            ]
            snapshot = _build_config_snapshot(
                cfg,
                experiment={
                    "experiment": name,
                    "condition": cond_name,
                    "bundle_source": bundle_source,
                    "route": route_name,
                    "order_variant": cond.get("order_variant"),
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

    # carry.source names the experiment's evidence source: the uses:-resolved
    # input when there is one (the winner is then a condition like a mode, not
    # a source), else the winning condition (the source experiment itself).
    carry = {
        key: (resolved_source if key == "source" and resolved_source else winner)
        for key in exp.get("carry", [])
    }
    state = {
        "experiment": name,
        "stamp": stamp if not score_only else None,
        "uses": uses or None,
        "resolved": (
            {"source": resolved_source, "component_overrides": resolved_overrides}
            if uses
            else None
        ),
        "runs": run_dirs,
        "winner": winner_cell,
        "winner_condition": winner,
        "dominated": dominated,
        "selection": selection,
        "carry": carry,
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


# =============================================================================
# STAGED EXPERIMENTS
# =============================================================================


def _resolve_uses(uses: str) -> str:
    """Resolve '<experiment>.carry.<key>' from that experiment's state file."""
    m = _USES_RE.match(uses)
    src_name, key = m["experiment"], m["key"]
    state_path = OUTPUTS / src_name / f"{src_name}_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"uses {uses!r}: no state file at {state_path}")
    value = json.loads(state_path.read_text()).get("carry", {}).get(key)
    if not value:
        raise ValueError(f"uses {uses!r}: {state_path.name} carries no {key!r}")
    return value


def _staged_state(exp: dict) -> tuple[dict, Path]:
    """Load (or initialise) a staged experiment's state; carried starts all-blank.

    The all-blank carried dict IS the blank W0 floor: every stage component is
    overridden to "" until a stage's selection fills it.
    """
    name = exp["experiment"]
    order = [s["component"] for s in exp["stages"]]
    path = OUTPUTS / name / f"{name}_state.json"
    if path.exists():
        return json.loads(path.read_text()), path
    return {
        "experiment": name,
        "uses": exp.get("uses"),
        "source": None,
        "order": order,
        "carried": dict.fromkeys(order, ""),
        "stages": [],
    }, path


def _save_staged_state(state: dict, path: Path) -> None:
    state["components_filled"] = [c for c in state["order"] if state["carried"].get(c)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))
    print(f"\nstate -> {path}")


def _run_stage(
    exp: dict, stage: str, *, dry_run: bool, score_only: bool, source: str | None
) -> dict:
    """Run one stage's conditions (prior + candidates), score, upsert, STOP.

    Selection is deliberately absent: several goal metrics are too low-powered
    on this benchmark to trust an automatic rule, so each stage ends at a human
    gate -- the panels land in the state file and a person records the choice
    via select=.
    """
    name = exp["experiment"]
    spec = next((s for s in exp["stages"] if s["component"] == stage), None)
    if spec is None:
        raise ValueError(
            f"unknown stage {stage!r}; stages: {[s['component'] for s in exp['stages']]}"
        )
    state, state_path = _staged_state(exp)

    # Resolve the evidence source once per experiment: explicit override, else
    # the source earlier stages ran on, else the uses: input. A mid-experiment
    # switch would mix evidence regimes across stages, so it is refused.
    resolved = source or state.get("source")
    if resolved is None and exp.get("uses"):
        resolved = _resolve_uses(exp["uses"])
    if resolved is None:
        raise ValueError(f"{name}: no evidence source (declare uses: or pass source=)")
    if state["stages"] and state.get("source") and resolved != state["source"]:
        raise ValueError(
            f"{name}: stages already ran on source={state['source']!r}; refusing "
            f"to run {stage} on {resolved!r}"
        )
    state["source"] = resolved

    carried = state["carried"]
    filled = [c for c in state["order"] if carried.get(c)]
    print(
        f"[{name}] stage {stage} on source={resolved}; "
        f"carried build: {'+'.join(filled) if filled else 'blank W0'}"
    )

    gt, coh = load_gt_and_coherence()
    controls = validation_specs(gt)
    route_name = exp["run"]["route"]
    conditions = {f"{stage}_prior": dict(carried)}
    for cand in spec["candidates"]:
        conditions[f"{stage}_{cand['id']}"] = {**carried, stage: cand["text"]}

    if score_only:
        out = latest_run_dir(name, stage)
        if out is None:
            raise FileNotFoundError(
                f"score_only: no archived run dir {OUTPUTS / name}/{stage}_<stamp>/"
            )
        stamp = None
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg = _condition_config(exp, stage, resolved, stamp, dry_run)
        specs = [
            RunSpec(
                route=ROUTE_REGISTRY[route_name],
                condition_name=label,
                component_overrides=overrides,
            )
            for label, overrides in conditions.items()
        ]
        snapshot = _build_config_snapshot(
            cfg,
            experiment={
                "experiment": name,
                "stage": stage,
                "goal": spec["goal"],
                "source": resolved,
                "carried_components": filled,
            },
        )
        print(f"[{name}] {len(conditions)} conditions -> {cfg.experiment_output_dir}")
        _run_benchmark_loop(cfg, specs, snapshot, phase_label=f"{name}:{stage}")
        out = cfg.experiment_output_dir

    def _abstain(label: str) -> str:
        rows = decoy_results(out, label, controls)
        return f"{sum(1 for d in rows if d['passed'])}/{len(rows)}"

    panels = {
        label: score_run(out, gt, cluster_coherence=coh, route_equals=label)
        for label in conditions
    }
    record = {
        "stage": stage,
        "goal": spec["goal"],
        "run_dir": out.name,
        "resolved": {"source": resolved, "carried_components": filled},
        "prior": panel_json(panels[f"{stage}_prior"]),
        "prior_abstain": _abstain(f"{stage}_prior"),
        "candidates": {
            cand["id"]: panel_json(panels[f"{stage}_{cand['id']}"])
            for cand in spec["candidates"]
        },
        "candidate_abstain": {
            cand["id"]: _abstain(f"{stage}_{cand['id']}") for cand in spec["candidates"]
        },
        "selected": None,
    }
    state["stages"] = [s for s in state["stages"] if s["stage"] != stage]
    state["stages"].append(record)
    state["stages"].sort(key=lambda s: state["order"].index(s["stage"]))
    _save_staged_state(state, state_path)
    print(f"[{name}] gate: record the choice with select=({stage!r}, <candidate|'prior'>)")
    return state


def _record_selection(exp: dict, select: tuple[str, str]) -> dict:
    """Record the human gate decision for a stage; finalize after the last one.

    The chosen candidate's text (or "" for prior) becomes the carried build for
    every later stage. When every stage has a recorded choice, the final build
    is assembled into ``carry`` for the next step to consume.
    """
    stage, choice = select
    name = exp["experiment"]
    spec = next((s for s in exp["stages"] if s["component"] == stage), None)
    if spec is None:
        raise ValueError(
            f"unknown stage {stage!r}; stages: {[s['component'] for s in exp['stages']]}"
        )
    state, state_path = _staged_state(exp)
    record = next((s for s in state["stages"] if s["stage"] == stage), None)
    if record is None:
        raise ValueError(f"stage {stage!r} has not been run yet (pass stage={stage!r} first)")
    texts = {cand["id"]: cand["text"] for cand in spec["candidates"]}
    if choice != "prior" and choice not in texts:
        raise ValueError(f"unknown candidate {choice!r}; options: {['prior', *texts]}")
    record["selected"] = choice
    state["carried"][stage] = "" if choice == "prior" else texts[choice]

    chosen = {s["stage"] for s in state["stages"] if s["selected"]}
    if chosen == set(state["order"]):
        filled = [c for c in state["order"] if state["carried"].get(c)]
        state["winner"] = "+".join(filled) if filled else "blank_W0"
        state["carry"] = {
            "source": state["source"],
            "components_filled": filled,
            "final_component_texts": {k: v for k, v in state["carried"].items() if v},
        }
        print(f"[{name}] all stages chosen; final build: {state['winner']}")
    else:
        pending = [c for c in state["order"] if c not in chosen]
        print(f"[{name}] recorded {stage} <- {choice}; next stage: {pending[0]}")
    _save_staged_state(state, state_path)
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("yaml_path", type=Path, help="experiment yaml (e.g. experiments/source.yaml)")
    ap.add_argument("--dry-run", action="store_true", help="mock outputs, zero API cost")
    ap.add_argument(
        "--score-only", action="store_true", help="re-score newest archived run dirs, no API calls"
    )
    ap.add_argument("--stage", help="staged experiments: run one stage's candidates and stop")
    ap.add_argument(
        "--select",
        nargs=2,
        metavar=("STAGE", "CHOICE"),
        help="staged experiments: record the human gate decision (candidate id or 'prior')",
    )
    ap.add_argument("--source", help="staged experiments: override the uses:-resolved source")
    args = ap.parse_args()
    run_experiment(
        args.yaml_path,
        stage=args.stage,
        select=tuple(args.select) if args.select else None,
        dry_run=args.dry_run,
        score_only=args.score_only,
        source=args.source,
    )


if __name__ == "__main__":
    main()
