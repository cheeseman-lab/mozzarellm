#!/usr/bin/env python
"""Integrated source -> prompt -> mode optimization driver.

Runs the three benchmark axes as chained stages, scoring every cell against
reviewer-consensus ground truth and carrying the per-stage winner forward:

    Stage 1 SOURCE  {uniprot,affinage,both} x {single_call, single_call_mcp}
    Stage 2 PROMPT  W0..W24 wording variants on the winning source
    Stage 3 MODE    single_call / cot / stepwise on the winning source+prompt

Each stage reuses the existing orchestrator (Phase 1 architecture for the
source/mode axes, Phase 3 wording for the prompt axis), scores each cell with
scorer.score_run (isolating cells by exact route), and selects with
walkup.select_winner (category primary, do-no-harm guards). Selections are
written to pipeline_state.json so stages are independently launchable
(--stage {1,2,3}) and a human can confirm or edit the carried winner between
stages -- the holistic call the doc reserves for a person.

Validate the whole plumbing with --dry-run (mock outputs, zero API cost).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from architecture_benchmarking_workflow.bench_configparse import load_config
from architecture_benchmarking_workflow.bench_orchestrator import run_benchmark
from architecture_benchmarking_workflow.ground_truth import (
    build_consensus_gt,
    consensus_coherence,
    load_consensus_gt,
)
from architecture_benchmarking_workflow.scorer import MetricPanel, score_decoys, score_run
from architecture_benchmarking_workflow.walkup import metric_value, select_holistic, select_winner

HOLISTIC_METRICS = ["category", "novel_subclass", "unchar_subclass", "coherence"]

# Negative-control decoys and what a correct model does with each (validation,
# not consensus-scored): nonsense/control clusters must abstain (Low confidence,
# "no discernible cluster"); the large coherent cluster must stay functional
# (valid output, no truncation/error).
DECOY_SPECS = {
    ("aconcagua_interphase_shuffled", "17"): "abstain",
    ("whitney", "49"): "abstain",
    ("jebel", "0"): "functional",
}

SCRIPT_DIR = Path(__file__).resolve().parent
GT_DIR = SCRIPT_DIR / "ground_truth"
CONFIGS = SCRIPT_DIR / "configs"
# Stage 1 runs the full slate (5 real + 3 decoys, from the source configs) so
# decoy abstention/robustness is measured per source. Stages 2-3 tune prompt and
# mode on the real clusters only -- decoys don't inform those and cost 60% more.
CLUSTERS_REAL = SCRIPT_DIR / "benchmark_inputs_v3" / "benchmark_clusters_5real.csv"
PIPE_DIR = SCRIPT_DIR / "benchmarking_outputs" / "pipeline"
STATE_PATH = PIPE_DIR / "pipeline_state.json"
GT_PATH = PIPE_DIR / "consensus_gt.csv"

SOURCES = ("uniprot", "affinage", "both")
MODES = ("single_call", "cot", "stepwise")
PRIMARY = "category"
GUARDS = ["novel_subclass", "unchar_subclass", "coherence"]
_WID_RE = re.compile(r"_(W\d+)_")


def _reviewer_csvs() -> dict[str, Path]:
    return {r: GT_DIR / f"annotation_{r}.csv" for r in ("eric", "liz", "iain")}


def load_gt_and_coherence():
    """Build (or rebuild) the consensus GT CSV and derive per-cluster coherence."""
    PIPE_DIR.mkdir(parents=True, exist_ok=True)
    build_consensus_gt(_reviewer_csvs(), GT_DIR / "survey_v3_key.csv", [], GT_PATH)
    return load_consensus_gt(GT_PATH), consensus_coherence(_reviewer_csvs())


def _distinct_routes(run_dir: Path) -> list[str]:
    routes: list[str] = []
    seen: set[str] = set()
    with open(run_dir / "parsed_outputs.jsonl") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            route = json.loads(line).get("route", "")
            if route and route not in seen:
                seen.add(route)
                routes.append(route)
    return routes


def score_cells(run_dir: Path, gt: dict, coh: dict) -> dict[str, MetricPanel]:
    """Score each distinct route in a run dir in isolation (exact route match)."""
    return {
        route: score_run(run_dir, gt, cluster_coherence=coh, route_equals=route)
        for route in _distinct_routes(run_dir)
    }


def _panel_json(panel: MetricPanel) -> dict:
    return {
        "category": panel.category,
        "novel_subclass": list(panel.novel_subclass),
        "unchar_subclass": list(panel.unchar_subclass),
        "coherence": list(panel.coherence),
        "n": panel.n,
        "failures": panel.failures,
    }


def _print_panel(label: str, panel: MetricPanel) -> None:
    vals = " ".join(f"{m}={metric_value(panel, m):.3f}" for m in (PRIMARY, *GUARDS))
    print(f"  {label:<44} {vals}  n={panel.n} fail={panel.failures}")


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def _save_state(state: dict) -> None:
    PIPE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))
    print(f"\nstate -> {STATE_PATH}")


def _prepare(config, experiment_id: str, dry_run: bool):
    config.experiment_id = experiment_id
    config.paths.output_dir = PIPE_DIR
    config.run.overwrite_outputs = True  # deterministic dir: PIPE_DIR/experiment_id
    config.run.dry_run = dry_run
    # We own PIPE_DIR: clear this experiment's stale dir so the orchestrator
    # writes fresh (avoids its interactive truncate-confirm prompt).
    out = config.experiment_output_dir
    if out.is_relative_to(PIPE_DIR):
        shutil.rmtree(out, ignore_errors=True)
    return config


# =============================================================================
# STAGE 1 — SOURCE (3 x 2)
# =============================================================================


def stage1(dry_run: bool) -> dict:
    gt, coh = load_gt_and_coherence()
    cells: dict[str, MetricPanel] = {}
    decoys: dict[str, list] = {}
    for source in SOURCES:
        config = _prepare(
            load_config(CONFIGS / f"source_{source}.yaml"), f"s1_source__{source}", dry_run
        )
        print(
            f"\n[stage1] source={source} routes=single_call+single_call_mcp -> {config.experiment_id}"
        )
        run_benchmark(config)
        out = config.experiment_output_dir
        for route, panel in score_cells(out, gt, coh).items():
            cells[f"{source}::{route}"] = panel
        # Decoy validation on the plain single_call route (route_excludes drops mcp).
        decoys[source] = [
            {
                "screen": r.screen,
                "cluster": r.cluster,
                "expectation": r.expectation,
                "reps": r.reps,
                "failures": r.failures,
                "modal_confidence": r.modal_confidence,
                "passed": r.passed,
            }
            for r in score_decoys(out, DECOY_SPECS)
        ]

    print("\n[stage1] cell panels:")
    for label in sorted(cells):
        _print_panel(label, cells[label])

    # Source is a holistic axis (eliminate dominated, then argmax category),
    # NOT the guarded prompt walkup -- richer sources legitimately trade one
    # panel metric for another, which a do-no-harm guard would wrongly veto.
    winner_key, dominated = select_holistic(cells, PRIMARY, HOLISTIC_METRICS)
    winner_source = winner_key.split("::")[0]
    print(f"\n[stage1] holistic winner: {winner_key}  ->  source={winner_source}")
    if dominated:
        print(f"[stage1] dominated cells: {dominated}")
    print("[stage1] NOTE: confirm/override winner_source in pipeline_state.json before stage 2.")

    for source in SOURCES:
        passes = sum(1 for d in decoys.get(source, []) if d["passed"])
        print(f"[stage1] decoys {source}: {passes}/{len(decoys.get(source, []))} passed")

    return {
        "cells": {k: _panel_json(v) for k, v in cells.items()},
        "dominated": dominated,
        "winner_cell": winner_key,
        "winner_source": winner_source,
        "decoys": decoys,
    }


# =============================================================================
# STAGE 2 — PROMPT (W0..W24)
# =============================================================================


def stage2(dry_run: bool, source: str) -> dict:
    gt, coh = load_gt_and_coherence()
    config = _prepare(load_config(CONFIGS / f"source_{source}.yaml"), "s2_prompt", dry_run)
    config.paths.benchmark_clusters_csv = CLUSTERS_REAL  # prompt walkup on real clusters only
    config.architecture_benchmark.enabled = False
    config.wording_benchmark.enabled = True
    config.wording_benchmark.base_routes = ["single_call"]
    config.wording_benchmark.targets = "all"
    # Evidence source is fixed by the bundle dir; wording targets resolve their
    # own alternate text set, so force_source must stay unset here.
    print(f"\n[stage2] prompt walkup W0..W24 on source={source} -> {config.experiment_id}")
    run_benchmark(config)

    by_wid: dict[str, MetricPanel] = {}
    for route, panel in score_cells(config.experiment_output_dir, gt, coh).items():
        m = _WID_RE.search(route)
        if m:
            by_wid[m.group(1)] = panel

    print("\n[stage2] variant panels:")
    for wid in sorted(by_wid, key=lambda w: int(w[1:])):
        _print_panel(wid, by_wid[wid])

    baseline = by_wid.get("W0")
    if baseline is None:
        raise RuntimeError("W0 baseline missing from stage-2 outputs")
    variants = {k: v for k, v in by_wid.items() if k != "W0"}
    winner = select_winner(baseline, variants, PRIMARY, GUARDS)
    winner_prompt = "W0" if winner == "baseline" else winner
    print(f"\n[stage2] recommended winner: {winner_prompt}")

    return {
        "source": source,
        "cells": {k: _panel_json(v) for k, v in by_wid.items()},
        "winner_prompt": winner_prompt,
    }


# =============================================================================
# STAGE 3 — MODE (single_call / cot / stepwise)
# =============================================================================


def stage3(dry_run: bool, source: str, prompt: str) -> dict:
    gt, coh = load_gt_and_coherence()
    config = _prepare(load_config(CONFIGS / f"source_{source}.yaml"), "s3_mode", dry_run)
    config.paths.benchmark_clusters_csv = CLUSTERS_REAL  # mode axis on real clusters only
    if prompt == "W0":
        # No wording override: run the three modes as a plain architecture axis.
        config.architecture_benchmark.enabled = True
        config.architecture_benchmark.base_routes = list(MODES)
        config.wording_benchmark.enabled = False
    else:
        # Carry the winning prompt into each mode via the wording override system
        # (it always emits the W0 baseline per mode alongside the chosen target).
        config.architecture_benchmark.enabled = False
        config.wording_benchmark.enabled = True
        config.wording_benchmark.base_routes = list(MODES)
        config.wording_benchmark.targets = [prompt]
    print(
        f"\n[stage3] mode axis {MODES} on source={source} prompt={prompt} -> {config.experiment_id}"
    )
    run_benchmark(config)

    cells = score_cells(config.experiment_output_dir, gt, coh)
    # Keep only the chosen prompt's cell per mode; key by mode.
    by_mode: dict[str, MetricPanel] = {}
    for route, panel in cells.items():
        wid = _WID_RE.search(route)
        if prompt != "W0" and (wid is None or wid.group(1) != prompt):
            continue
        for mode in MODES:
            if route.startswith(mode):
                by_mode[mode] = panel

    print("\n[stage3] mode panels:")
    for mode in MODES:
        if mode in by_mode:
            _print_panel(mode, by_mode[mode])

    if "single_call" not in by_mode:
        raise RuntimeError("single_call baseline missing from stage-3 outputs")
    # Mode is holistic like source (delivery format trades metrics off).
    winner_mode, dominated = select_holistic(by_mode, PRIMARY, HOLISTIC_METRICS)
    print(f"\n[stage3] holistic winner: {winner_mode}")
    if dominated:
        print(f"[stage3] dominated modes: {dominated}")

    return {
        "source": source,
        "prompt": prompt,
        "cells": {k: _panel_json(v) for k, v in by_mode.items()},
        "dominated": dominated,
        "winner_mode": winner_mode,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["1", "2", "3", "all"], default="all")
    ap.add_argument("--dry-run", action="store_true", help="mock outputs, zero API cost")
    args = ap.parse_args()

    # Dry-runs write mock data to an isolated dir so they can never clobber real
    # run outputs, state, or figures.
    if args.dry_run:
        global PIPE_DIR, STATE_PATH, GT_PATH
        PIPE_DIR = PIPE_DIR / "_dryrun"
        STATE_PATH = PIPE_DIR / "pipeline_state.json"
        GT_PATH = PIPE_DIR / "consensus_gt.csv"

    state = _load_state()

    if args.stage in ("1", "all"):
        state["stage1"] = stage1(args.dry_run)
        _save_state(state)

    if args.stage in ("2", "all"):
        source = state.get("stage1", {}).get("winner_source")
        if not source:
            raise SystemExit("stage2 needs stage1.winner_source; run --stage 1 first")
        state["stage2"] = stage2(args.dry_run, source)
        _save_state(state)

    if args.stage in ("3", "all"):
        s2 = state.get("stage2", {})
        source, prompt = s2.get("source"), s2.get("winner_prompt")
        if not source or not prompt:
            raise SystemExit("stage3 needs stage2 winner; run --stage 2 first")
        state["stage3"] = stage3(args.dry_run, source, prompt)
        _save_state(state)

    final = {
        "source": state.get("stage1", {}).get("winner_source"),
        "prompt": state.get("stage2", {}).get("winner_prompt"),
        "mode": state.get("stage3", {}).get("winner_mode"),
    }
    print(f"\n=== pipeline selection: {final} ===")


if __name__ == "__main__":
    main()
