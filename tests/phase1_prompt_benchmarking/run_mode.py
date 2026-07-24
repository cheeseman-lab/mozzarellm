#!/usr/bin/env python
"""Step 3 of the pipeline: MODE on the walkup's final assembled prompt.

Runs {single_call, cot, stepwise} on the 8-cluster slate (real clusters drive
selection, the 3 decoys validate the final config), n=3, on the source and
assembled prompt carried from the walkup (runs/walkup). Picks the mode
holistically (delivery format legitimately trades metrics off). Writes the
latest-pointer runs/mode/mode_state.json, archives the state alongside the
timestamped run dir, and pushes the run into the runs submodule.

Component-mapping caveat: the walkup tunes the single_call components
(CAT/GCR/NPR/UPR/PCC). cot/stepwise use different component keys
(cGCR/cPri/cPSC/cVer/...); only CAT (and SC) are shared. So single_call runs the
full tuned prompt, while cot/stepwise carry only the tuned CAT lens and keep
canonical text for their own components. This is a fair mode comparison, not a
verbatim prompt transplant.

--dry-run exercises the plumbing with mock outputs (zero API); --score-only
re-scores the existing run dirs and rewrites state without any API calls.
"""

from __future__ import annotations

import argparse

from architecture_benchmarking_workflow.bench_configparse import load_config
from architecture_benchmarking_workflow.bench_orchestrator import (
    RunSpec,
    _build_config_snapshot,
    _run_benchmark_loop,
)
from architecture_benchmarking_workflow.bench_pipeline_common import (
    CONFIGS,
    HOLISTIC_METRICS,
    MODES,
    OUTPUTS,
    PRIMARY,
    decoy_results,
    latest_run_dir,
    load_gt_and_coherence,
    panel_json,
    prepare,
    print_panel,
    push_run,
    read_carry,
    run_stamp,
    score_cell,
    select_holistic,
    write_state,
)
from architecture_benchmarking_workflow.bench_routes import MODE_REGISTRY

STATE_PATH = OUTPUTS / "mode" / "mode_state.json"


def _overrides_for(mode: str, final: dict[str, str]) -> dict[str, str]:
    """single_call gets the full tuned prompt; cot/stepwise carry only shared CAT."""
    if mode == "single_call":
        return dict(final)
    return {"CAT": final["CAT"]} if final.get("CAT") else {}


def _config_for(source: str, stamp: str, dry_run: bool, score_only: bool):
    """Fresh (run) or non-destructive (score-only) config in runs/mode/<stamp>/mode/.
    experiment_id stays flat ("mode") so run_id / trace paths never contain slashes."""
    cfg = load_config(CONFIGS / f"source_{source}.yaml")
    out_root = OUTPUTS / "mode" / stamp
    if score_only:
        cfg.experiment_id = "mode"
        cfg.paths.output_dir = out_root
        cfg.run.overwrite_outputs = True  # we only read it, no wipe
        return cfg
    return prepare(cfg, "mode", dry_run, out_root=out_root)


def run_mode(dry_run: bool, source: str | None = None, score_only: bool = False) -> None:
    carry = read_carry("walkup")
    source = source or carry.get("source")
    final = carry.get("final_component_texts", {})
    if not source:
        raise SystemExit("run_mode needs a source; run run_walkup.py first (or pass --source).")
    filled = carry.get("components_filled", [])

    if score_only:
        prev = latest_run_dir("mode")
        if prev is None:
            raise SystemExit("[mode] no prior run under runs/mode/ to score")
        stamp = prev.name
        print(f"[mode] --score-only: re-scoring run {stamp}")
    else:
        stamp = run_stamp()
    print(f"[mode] source={source} carrying walkup build: {'+'.join(filled) or 'blank W0'}")

    gt, coh = load_gt_and_coherence()
    cfg = _config_for(source, stamp, dry_run, score_only)
    if not score_only:
        specs = [
            RunSpec(
                route=MODE_REGISTRY[mode],
                condition_name=mode,
                component_overrides=_overrides_for(mode, final),
            )
            for mode in MODES
        ]
        snapshot = _build_config_snapshot(cfg, mode={"source": source, "carried": filled})
        print(f"[mode] {list(MODES)} -> {cfg.experiment_id}")
        _run_benchmark_loop(cfg, specs, snapshot, phase_label="mode")

    out = cfg.experiment_output_dir
    cells = {mode: score_cell(out, gt, coh, mode) for mode in MODES}
    decoys = {mode: decoy_results(out, condition=mode) for mode in MODES}

    print("\n[mode] mode panels:")
    for mode in MODES:
        print_panel(mode, cells[mode])
    for mode in MODES:
        passes = sum(1 for d in decoys[mode] if d["passed"])
        print(f"[mode] decoys {mode}: {passes}/{len(decoys[mode])} passed")

    winner_mode, dominated = select_holistic(cells, PRIMARY, HOLISTIC_METRICS)
    print(f"\n[mode] holistic winner: {winner_mode}")
    if dominated:
        print(f"[mode] dominated modes: {dominated}")

    write_state(
        STATE_PATH,
        step="mode",
        source=source,
        winner=winner_mode,
        decoys=decoys,
        carry={"source": source, "mode": winner_mode},
        stamp=stamp,
        cells={k: panel_json(v) for k, v in cells.items()},
        dominated=dominated,
    )
    archive = OUTPUTS / "mode" / stamp / "mode_state.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_text(STATE_PATH.read_text())
    if not dry_run:
        push_run("mode", f"mode run {stamp}: winner {winner_mode}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="mock outputs, zero API cost")
    ap.add_argument(
        "--score-only", action="store_true", help="re-score existing run dirs, no API calls"
    )
    ap.add_argument("--source", help="override the source carried from the walkup")
    args = ap.parse_args()
    run_mode(args.dry_run, args.source, args.score_only)


if __name__ == "__main__":
    main()
