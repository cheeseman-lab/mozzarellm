#!/usr/bin/env python
"""Step 3 of the pipeline: MODE on the walkup's final assembled prompt.

Runs {single_call, cot, stepwise} on the 8-cluster slate (real clusters drive
selection, the 3 decoys validate the final config), n=3, on the source and
assembled prompt carried from the walkup. Picks the mode holistically (delivery
format legitimately trades metrics off). Writes the latest-pointer
benchmarking_outputs/mode/mode_state.json; each run archives in place under
benchmarking_outputs/mode/<source>_<stamp>/ (nothing overwritten).

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
    """Config for the mode run. Runs archive under benchmarking_outputs/mode/
    <source>_<stamp>/; score-only just loads (dir found via latest_run_dir)."""
    cfg = load_config(CONFIGS / f"source_{source}.yaml")
    if score_only:
        return cfg
    return prepare(cfg, f"{source}_{stamp}", dry_run, out_root=OUTPUTS / "mode")


def run_mode(dry_run: bool, source: str | None = None, score_only: bool = False) -> None:
    carry = read_carry("walkup")
    source = source or carry.get("source")
    final = carry.get("final_component_texts", {})
    if not source:
        raise SystemExit("run_mode needs a source; run run_walkup.py first (or pass --source).")
    filled = carry.get("components_filled", [])
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
    else:
        out = latest_run_dir("mode", source)
        if out is None:
            raise SystemExit(f"[mode] no prior run for {source} under {OUTPUTS / 'mode'}")

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
