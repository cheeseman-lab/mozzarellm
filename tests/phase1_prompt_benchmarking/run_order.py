#!/usr/bin/env python
"""Step 4 of the pipeline: component ORDER on the tuned prompt.

Holds the source, mode, and component *content* fixed (from the walkup + mode
carries) and permutes the component_order to measure positional sensitivity.
Runs the order variants resolve_order_variant_ids("all") = [O, O1, O2, O3, O4]
on the 8-cluster slate (real clusters drive selection, the 3 decoys validate),
n=3, picks the order holistically, and reports the cw-recall spread across
variants (how much order alone moves the result). Writes the latest-pointer
benchmarking_outputs/order/order_state.json; each run archives in place under
benchmarking_outputs/order/<source>_<stamp>/ (nothing overwritten).

Scope (V1): the walkup tunes the single_call component keys, so order is applied
to the single_call route with the walkup's assembled texts injected -- even if
the mode winner was cot/stepwise (those use different component keys, so a
verbatim order transplant is out of scope for now). The mode winner is reported
for context.

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
from architecture_benchmarking_workflow.bench_order import (
    apply_order_variant,
    resolve_order_variant_ids,
)
from architecture_benchmarking_workflow.bench_pipeline_common import (
    CONFIGS,
    HOLISTIC_METRICS,
    OUTPUTS,
    PRIMARY,
    decoy_results,
    latest_run_dir,
    load_gt_and_coherence,
    metric_value,
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

STATE_PATH = OUTPUTS / "order" / "order_state.json"
# V1 permutes the single_call route -- the mode whose component keys the walkup tuned.
BASE_MODE = "single_call"


def _config_for(source: str, stamp: str, dry_run: bool, score_only: bool):
    """Config for the order run. Runs archive under benchmarking_outputs/order/
    <source>_<stamp>/; score-only just loads (dir found via latest_run_dir)."""
    cfg = load_config(CONFIGS / f"source_{source}.yaml")
    if score_only:
        return cfg
    return prepare(cfg, f"{source}_{stamp}", dry_run, out_root=OUTPUTS / "order")


def run_order(dry_run: bool, source: str | None = None, score_only: bool = False) -> None:
    mode_carry = read_carry("mode")
    walkup_carry = read_carry("walkup")
    source = source or mode_carry.get("source") or walkup_carry.get("source")
    if not source:
        raise SystemExit("run_order needs a source; run the mode step first (or pass --source).")
    final = walkup_carry.get("final_component_texts", {})
    mode_winner = mode_carry.get("mode", "?")
    variant_ids = resolve_order_variant_ids("all")
    print(
        f"[order] source={source} mode-winner={mode_winner} "
        f"(order applied to {BASE_MODE}); variants={variant_ids}"
    )

    stamp = run_stamp()
    gt, coh = load_gt_and_coherence()
    cfg = _config_for(source, stamp, dry_run, score_only)
    base_route = MODE_REGISTRY[BASE_MODE]
    if not score_only:
        specs = [
            RunSpec(
                route=apply_order_variant(base_route, vid),
                condition_name=vid,
                component_overrides=dict(final),
            )
            for vid in variant_ids
        ]
        snapshot = _build_config_snapshot(
            cfg, order={"source": source, "mode_winner": mode_winner, "variants": variant_ids}
        )
        print(f"[order] {variant_ids} -> {cfg.experiment_id}")
        _run_benchmark_loop(cfg, specs, snapshot, phase_label="order")
        out = cfg.experiment_output_dir
    else:
        out = latest_run_dir("order", source)
        if out is None:
            raise SystemExit(f"[order] no prior run for {source} under {OUTPUTS / 'order'}")

    cells = {vid: score_cell(out, gt, coh, vid) for vid in variant_ids}
    decoys = {vid: decoy_results(out, condition=vid) for vid in variant_ids}

    print("\n[order] variant panels:")
    for vid in variant_ids:
        print_panel(vid, cells[vid])
    for vid in variant_ids:
        passes = sum(1 for d in decoys[vid] if d["passed"])
        print(f"[order] decoys {vid}: {passes}/{len(decoys[vid])} passed")

    cw = {vid: metric_value(p, PRIMARY) for vid, p in cells.items()}
    spread = max(cw.values()) - min(cw.values())
    winner_order, dominated = select_holistic(cells, PRIMARY, HOLISTIC_METRICS)
    print(f"\n[order] holistic winner: {winner_order}  (cw-recall spread {spread:.3f})")
    if dominated:
        print(f"[order] dominated variants: {dominated}")

    write_state(
        STATE_PATH,
        step="order",
        source=source,
        winner=winner_order,
        decoys=decoys,
        carry={"source": source, "mode": mode_winner, "order": winner_order},
        stamp=stamp,
        cells={k: panel_json(v) for k, v in cells.items()},
        cw_recall=cw,
        spread=round(spread, 4),
        dominated=dominated,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="mock outputs, zero API cost")
    ap.add_argument(
        "--score-only", action="store_true", help="re-score existing run dirs, no API calls"
    )
    ap.add_argument("--source", help="override the source carried from the mode step")
    args = ap.parse_args()
    run_order(args.dry_run, args.source, args.score_only)


if __name__ == "__main__":
    main()
