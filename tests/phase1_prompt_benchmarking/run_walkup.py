#!/usr/bin/env python
"""Step 2 of the pipeline: the build-up prompt WALKUP -- paused, human-gated.

The walkup builds the prompt from a blank W0 by adding one component per stage
(CAT -> GCR -> NPR -> UPR -> PCC). Rather than auto-selecting each stage on a
single metric -- several target metrics are too low-powered on this 5-cluster
benchmark to trust (unchar-subclass n~=6, coherence n=5) -- selection is a human
gate: each stage runs its candidates, prints the FULL metric panel (with the
sub-metric n's exposed so noise is visible), and stops. A person picks the winner
with judgment, weighing the robust axis (category / coverage-weighted recall over
133 genes) heavily and the noisy sub-metrics with skepticism.

Usage (one stage at a time):
    python run_walkup.py --stage CAT              # run CAT candidates, print panel, stop
    python run_walkup.py --select CAT concise     # record the human's choice, carry it forward
    python run_walkup.py --stage GCR              # ... next stage builds on the carried winner
    ...
    python run_walkup.py --finalize               # assemble final prompt + render the figure

Runs on the source chosen by run_source.py (carry.source), or --source, else
affinage. --dry-run exercises the plumbing with mock outputs (zero API).
"""

from __future__ import annotations

import argparse
import json
import shutil

from architecture_benchmarking_workflow.bench_configparse import load_config
from architecture_benchmarking_workflow.bench_evaluator import MetricPanel
from architecture_benchmarking_workflow.bench_orchestrator import (
    RunSpec,
    _build_config_snapshot,
    _run_benchmark_loop,
)
from architecture_benchmarking_workflow.bench_pipeline_common import (
    CLUSTERS_ALL,
    CONFIGS,
    N_REAL_GENES,
    OUTPUTS,
    decoy_results,
    load_gt_and_coherence,
    metric_value,
    panel_json,
    read_carry,
    score_cell,
    validation_specs,
)
from architecture_benchmarking_workflow.bench_routes import MODE_REGISTRY
from architecture_benchmarking_workflow.bench_walkup_candidates import (
    CANDIDATES,
    STAGE_GOAL,
    WALKUP_ORDER,
)

OUT_DIR = OUTPUTS / "walkup"
STATE_PATH = OUT_DIR / "walkup_state.json"

# Reliability note shown at each gate so the human weights metrics correctly.
_METRIC_POWER = "category n=103 robust · novel n~=40 soft · unchar n~=7 NOISE · coherence n=4 NOISE"


def _cand_text(stage: str) -> dict[str, str]:
    return {cid: text for cid, _r, text in CANDIDATES[stage]}


def _base_config(dry_run: bool, source: str, reps: int | None = None):
    cfg = load_config(CONFIGS / f"source_{source}.yaml")
    cfg.paths.output_dir = OUT_DIR
    cfg.paths.benchmark_clusters_csv = CLUSTERS_ALL  # 8-slate: score 103 + validate abstention
    cfg.run.overwrite_outputs = True
    cfg.run.dry_run = dry_run
    if reps is not None:
        cfg.run.num_replicates = reps
    cfg.mcp.preflight = False
    return cfg


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "step": "walkup",
        "source": None,
        "carried": dict.fromkeys(WALKUP_ORDER, ""),
        "stages": [],
        "order": list(WALKUP_ORDER),
    }


def _save_state(st: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    st["components_filled"] = [s for s in WALKUP_ORDER if st["carried"].get(s)]
    STATE_PATH.write_text(json.dumps(st, indent=2))


def _upsert_stage(st: dict, record: dict) -> None:
    st["stages"] = [s for s in st["stages"] if s["stage"] != record["stage"]]
    st["stages"].append(record)
    st["stages"].sort(key=lambda s: WALKUP_ORDER.index(s["stage"]))


def _row(label: str, p: MetricPanel, best: bool, abstain: str) -> str:
    nov, unc, coh = p.novel_subclass, p.unchar_subclass, p.coherence
    return (
        f"   {label:16} cw={metric_value(p, 'coverage_weighted_category'):.3f} "
        f"cat={p.category:.3f} cov={p.n}/{N_REAL_GENES} coh={coh[0]}/{coh[1]} abst={abstain} "
        f"nov={nov[0]}/{nov[1]} unc={unc[0]}/{unc[1]} fail={p.failures}"
        f"{'  <- best cw-recall' if best else ''}"
    )


def _print_panel(stage, prior, cands, prior_ab, cand_ab) -> None:
    all_cells = {"prior": prior, **cands}
    best = max(all_cells, key=lambda k: metric_value(all_cells[k], "coverage_weighted_category"))
    print(f"\n[{stage}] goal={STAGE_GOAL[stage]}   ({_METRIC_POWER})")
    print(_row("prior", prior, best == "prior", prior_ab))
    for cid, _r, _t in CANDIDATES[stage]:
        print(_row(cid, cands[cid], best == cid, cand_ab[cid]))
    print(f"\n   -> choose:  python run_walkup.py --select {stage} <candidate|prior>")


def cmd_stage(stage: str, dry_run: bool, source_arg: str | None, reps: int | None = None) -> None:
    st = _load_state()
    source = source_arg or st.get("source") or read_carry("source").get("source") or "affinage"
    st["source"] = source
    carried = st["carried"]
    print(
        f"[walkup] stage {stage} on source={source}; "
        f"carried build: {'+'.join(st['components_filled']) if st.get('components_filled') else 'blank W0'}"
    )

    gt, coh = load_gt_and_coherence()
    cfg = _base_config(dry_run, source, reps)
    cfg.experiment_id = stage
    out = cfg.experiment_output_dir
    shutil.rmtree(out, ignore_errors=True)

    specs = [
        RunSpec(
            route=MODE_REGISTRY["single_call"],
            condition_name=f"{stage}_prior",
            component_overrides={**carried},
        )
    ]
    for cid, _r, text in CANDIDATES[stage]:
        specs.append(
            RunSpec(
                route=MODE_REGISTRY["single_call"],
                condition_name=f"{stage}_{cid}",
                component_overrides={**carried, stage: text},
            )
        )
    snapshot = _build_config_snapshot(
        cfg, walkup={"stage": stage, "carried": st.get("components_filled", [])}
    )
    _run_benchmark_loop(cfg, specs, snapshot, phase_label=f"walkup:{stage}")

    prior = score_cell(out, gt, coh, f"{stage}_prior")
    cands = {cid: score_cell(out, gt, coh, f"{stage}_{cid}") for cid, _r, _t in CANDIDATES[stage]}
    controls = validation_specs(gt)

    def _abst(cond: str) -> str:
        rows = decoy_results(out, condition=cond, specs=controls)
        return f"{sum(1 for d in rows if d['passed'])}/{len(rows)}"

    prior_ab = _abst(f"{stage}_prior")
    cand_ab = {cid: _abst(f"{stage}_{cid}") for cid, _r, _t in CANDIDATES[stage]}
    _upsert_stage(
        st,
        {
            "stage": stage,
            "goal": STAGE_GOAL[stage],
            "prior": panel_json(prior),
            "prior_abstain": prior_ab,
            "candidates": {cid: panel_json(p) for cid, p in cands.items()},
            "candidate_abstain": cand_ab,
            "selected": None,
        },
    )
    _save_state(st)
    _print_panel(stage, prior, cands, prior_ab, cand_ab)


def cmd_select(stage: str, choice: str) -> None:
    st = _load_state()
    rec = next((s for s in st["stages"] if s["stage"] == stage), None)
    if rec is None:
        raise SystemExit(
            f"stage {stage} has not been run yet (python run_walkup.py --stage {stage})"
        )
    texts = _cand_text(stage)
    if choice != "prior" and choice not in texts:
        raise SystemExit(f"unknown candidate {choice!r}; options: {['prior', *texts]}")
    rec["selected"] = choice
    st["carried"][stage] = "" if choice == "prior" else texts[choice]
    _save_state(st)
    filled = st["components_filled"]
    print(
        f"[walkup] recorded {stage} <- {choice}.  build so far: "
        f"{' + '.join(filled) if filled else 'blank W0'}"
    )
    nxt = [s for s in WALKUP_ORDER if s not in {r["stage"] for r in st["stages"] if r["selected"]}]
    if nxt:
        print(f"[walkup] next: python run_walkup.py --stage {nxt[0]}")
    else:
        print("[walkup] all stages chosen -> python run_walkup.py --finalize")


def cmd_finalize() -> None:
    st = _load_state()
    unresolved = [s["stage"] for s in st["stages"] if s["selected"] is None]
    missing = [s for s in WALKUP_ORDER if s not in {r["stage"] for r in st["stages"]}]
    if unresolved or missing:
        raise SystemExit(f"cannot finalize; unrun={missing} unselected={unresolved}")
    carried = st["carried"]
    filled = st["components_filled"]
    st["winner"] = "+".join(filled) if filled else "blank_W0"
    st["carry"] = {
        "source": st["source"],
        "final_component_texts": {k: v for k, v in carried.items() if v},
        "components_filled": filled,
    }
    st["selected_components"] = {s["stage"]: s["selected"] for s in st["stages"]}
    _save_state(st)
    print(
        f"[walkup] finalized. build: {' + '.join(filled) if filled else 'blank W0 (nothing adopted)'}"
    )
    print(f"[walkup] render figure: figure_walkup_buildup('{STATE_PATH}', out.png)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stage", choices=list(WALKUP_ORDER), help="run one stage's candidates and stop"
    )
    ap.add_argument(
        "--select",
        nargs=2,
        metavar=("STAGE", "CHOICE"),
        help="record the human's pick for a stage (candidate id or 'prior')",
    )
    ap.add_argument(
        "--finalize", action="store_true", help="assemble final prompt from carried picks"
    )
    ap.add_argument("--dry-run", action="store_true", help="mock outputs, zero API cost")
    ap.add_argument("--source", help="override the source carried from run_source.py")
    ap.add_argument(
        "--reps",
        type=int,
        default=None,
        help="override num_replicates for this stage (default: config's 3)",
    )
    args = ap.parse_args()

    if args.stage:
        cmd_stage(args.stage, args.dry_run, args.source, args.reps)
    elif args.select:
        cmd_select(args.select[0], args.select[1])
    elif args.finalize:
        cmd_finalize()
    else:
        ap.error("one of --stage / --select / --finalize is required")


if __name__ == "__main__":
    main()
