#!/usr/bin/env python
"""Step 1 of the pipeline: SOURCE x MCP on the blank W0 floor.

Runs the blank prompt (every component rule empty -- the same floor the walkup
builds up from) across {uniprot, affinage, both} x {single_call, single_call_mcp}
on the 8-cluster slate (5 real clusters drive selection, 3 decoys are validated),
n=3, Sonnet 5. Scores each cell against reviewer-consensus GT and picks the source
holistically on coverage-weighted recall (correct categories / 133), so a cell
can't win by dropping hard genes to inflate raw category.

Alongside selection it records source diagnostics (source_metrics): per-class
recall, lenient any-reviewer recall, and the reviewers' direct source-preference
tally -- to explain WHY a source wins. Writes
benchmarking_outputs/source/source_state.json; run_walkup.py reads carry.source.

--dry-run exercises the plumbing with mock outputs (zero API); --score-only
re-scores the existing run dirs and rewrites state without any API calls.
"""

from __future__ import annotations

import argparse

from architecture_benchmarking_workflow.arch_bench_routes import MODE_REGISTRY
from architecture_benchmarking_workflow.bench_configparse import load_config
from architecture_benchmarking_workflow.bench_evaluator import (
    pathway_diagnostics,
    reviewer_concordance,
    reviewer_label_sets,
    source_diagnostics,
    source_preference_tally,
)
from architecture_benchmarking_workflow.bench_orchestrator import (
    RunSpec,
    _build_config_snapshot,
    _run_benchmark_loop,
)
from architecture_benchmarking_workflow.bench_pipeline_common import (
    CONFIGS,
    HOLISTIC_METRICS,
    OUTPUTS,
    PRIMARY,
    SOURCES,
    decoy_results,
    load_gt_and_coherence,
    panel_json,
    prepare,
    print_panel,
    reviewer_csvs,
    score_cell,
    select_holistic,
    validation_specs,
    write_state,
)

# Plain single_call only -- the regime the walkup and mode steps run. MCP is a
# delivery mode (it recovers coverage at 3-4x cost, no category gain) and is
# evaluated in the mode step on the final prompt, not swept across the sources.
ROUTES = ("single_call",)
# The blank W0 floor: every component rule empty. SC (screen context) and O
# (output schema) keep their registry defaults, matching run_walkup's start.
BLANK_W0 = dict.fromkeys(("CAT", "GCR", "NPR", "UPR", "PCC"), "")
STATE_PATH = OUTPUTS / "source" / "source_state.json"


def _config_for(source: str, dry_run: bool, score_only: bool):
    """Fresh (run) or non-destructive (score-only) config for one source."""
    cfg = load_config(CONFIGS / f"source_{source}.yaml")
    if score_only:
        cfg.experiment_id = f"source__{source}"
        cfg.paths.output_dir = OUTPUTS
        cfg.run.overwrite_outputs = True  # deterministic dir name; we only read it, no wipe
        return cfg
    return prepare(cfg, f"source__{source}", dry_run)


def run_source(dry_run: bool, score_only: bool = False) -> None:
    gt, coh = load_gt_and_coherence()
    reviewer_labels = reviewer_label_sets(reviewer_csvs())
    controls = validation_specs(gt)  # 3 decoys + coherence-derived abstains (denali/24)
    cells, decoys, diagnostics, pathway = {}, {}, {}, {}

    for source in SOURCES:
        cfg = _config_for(source, dry_run, score_only)
        if not score_only:
            specs = [
                RunSpec(
                    route=MODE_REGISTRY[route],
                    condition_name=f"{source}__{route}",
                    component_overrides={**BLANK_W0},
                )
                for route in ROUTES
            ]
            snapshot = _build_config_snapshot(cfg, source={"source": source, "prompt": "W0_blank"})
            print(f"\n[source] {source}: {[s.condition_name for s in specs]} -> {cfg.experiment_id}")
            _run_benchmark_loop(cfg, specs, snapshot, phase_label=f"source:{source}")

        out = cfg.experiment_output_dir
        for route in ROUTES:
            cells[f"{source}::{route}"] = score_cell(out, gt, coh, f"{source}__{route}")
        decoys[source] = decoy_results(out, condition=f"{source}__single_call", specs=controls)
        diagnostics[source] = source_diagnostics(out, gt, reviewer_labels, f"{source}__single_call")
        # Semantic pathway agreement is skipped in dry-run (mock text, no model load).
        pathway[source] = pathway_diagnostics(
            out, gt, reviewer_csvs(), f"{source}__single_call", use_semantic=not dry_run
        )

    preference = source_preference_tally(gt)
    concordance = reviewer_concordance(reviewer_csvs(), gt)

    print("\n[source] cell panels (blank W0):")
    for label in sorted(cells):
        print_panel(label, cells[label])

    print("\n[source] single_call diagnostics (consensus vs any-reviewer):")
    for source in SOURCES:
        d = diagnostics[source]
        print(
            f"  {source:9} cons-cw={d['consensus_cw']:.3f} any-cw={d['any_cw']:.3f} "
            f"(gain {d['leniency_gain']:+.3f}, forgiven {d['forgiven']}) | "
            f"novel cons={d['per_class']['NOVEL_ROLE']['recall_consensus']:.3f} "
            f"any={d['per_class']['NOVEL_ROLE']['recall_any']:.3f}"
        )
    print(f"[source] reviewer source-preference (real genes): {preference['overall']}")

    print("\n[source] pathway agreement (model dominant_process vs reviewer pathways):")
    for source in SOURCES:
        pw = pathway[source]
        sem = f"{pw['semantic_mean']:.3f}" if pw["semantic_mean"] is not None else "n/a"
        print(
            f"  {source:9} substring={pw['substring_rate']:.3f} loose={pw['loose_rate']:.3f} "
            f"semantic_mean={sem} match={pw['semantic_match_rate']:.3f}"
        )

    # Select the source on the PLAIN single_call cells only -- the regime the
    # walkup and mode steps actually run. MCP is an orthogonal augmentation tested
    # here (it recovers coverage but not category, at 3-4x cost) and reported in
    # the table, but it is never carried forward, so it must not pick the source.
    plain = {k: v for k, v in cells.items() if not k.endswith("_mcp")}
    winner_cell, dominated = select_holistic(plain, PRIMARY, HOLISTIC_METRICS)
    winner_source = winner_cell.split("::")[0]
    print(f"\n[source] holistic winner (coverage-weighted, single_call): {winner_cell} -> {winner_source}")
    if dominated:
        print(f"[source] dominated cells: {dominated}")
    for source in SOURCES:
        passes = sum(1 for d in decoys.get(source, []) if d["passed"])
        print(f"[source] decoys {source}: {passes}/{len(decoys.get(source, []))} passed")
    print("[source] NOTE: confirm/override carry.source in source_state.json before the walkup.")

    write_state(
        STATE_PATH,
        step="source",
        source=winner_source,
        winner=winner_cell,
        decoys=decoys,
        carry={"source": winner_source},
        cells={k: panel_json(v) for k, v in cells.items()},
        dominated=dominated,
        diagnostics=diagnostics,
        pathway=pathway,
        source_preference=preference,
        reviewer_concordance=concordance,
        selection_rule="coverage_weighted_category (holistic, coverage-aware)",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="mock outputs, zero API cost")
    ap.add_argument(
        "--score-only", action="store_true", help="re-score existing run dirs, no API calls"
    )
    args = ap.parse_args()
    run_source(args.dry_run, args.score_only)


if __name__ == "__main__":
    main()
