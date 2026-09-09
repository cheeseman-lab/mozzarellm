# Phase 1 Prompt Benchmarking

As of 5/13/2026: Implements the full prompt-benchmarking orchestrator for evaluating MozzareLLM cluster analysis (uniprot annotated evidence bundles + pubmed mcp) across architecture variants (Phase 1) and component-order perturbations (Phase 2).

## Directory

```
phase1_prompt_benchmarking/
    README.md
    benchmarking_info.md
    prompt-assembly-routes-info.md
    phase1_benchmark_preprocess.py
    benchmark_clusters_ground_truth.csv

    benchmark_inputs/
        benchmark_clusters.csv
        aconcagua_interphase_screen_context.json
        denali_screen_context.json
        jebel_screen_context.json
        whitney_screen_context.json

    benchmark_evidence_bundles/
        (14 pre-built bundle JSONs + per-screen intermediates)

    architecture_benchmarking_workflow/
        bench_routes.py          -- Route dataclass + 6-route registry
        bench_orchestrator.py         -- Engine: shared loop, prompt construction, execution
        bench_experiment.py           -- Experiment layer: yaml -> runs -> scoring -> state
        bench_configparse.py          -- Benchmark config dataclasses
        bench_metricfns.py            -- Structural, MCP, logical, efficiency metrics
        bench_reportgen.py            -- Markdown report + CSV/JSON aggregates
        bench_trace_parser.py         -- Trace JSON -> gene-level prediction CSVs
        bench_dry_run.py              -- Deterministic mock outputs for dry-run
        order_bench_orderings.py      -- Order variant definitions + route builder

    experiments/
        source.yaml                   -- evidence-source comparison (whole experiment)
        walkup.yaml                   -- staged prompt build-up (whole experiment)
        mode.yaml                     -- delivery-mode comparison on the final build
        order.yaml                    -- component-order sensitivity

    benchmarking_outputs/
        0.comp/                       -- (reserved) component "unit" testing**
        1.arch/                       -- Architecture benchmark outputs
        2.order/                      -- Order sensitivity outputs
        3.wording/                    -- (reserved) wording sensitivity**
```
** more detail in MLLM Benchmarking Plan_3_10_26.docx

## Output Structure (benchmarking_outputs/ Submodule)

Outputs are organized by phase, then by experiment. When `workflow_testing: true` in the config, outputs nest under a `_workflow_testing/` subdirectory to keep dev runs separate from final results.

```
benchmarking_outputs/
    1.arch/
        _workflow_testing/
            {experiment_id}/          -- one dir per experiment run
                ...
    2.order/
        {experiment_id}/
            ...
```

Each **experiment directory** contains:

- **config_snapshot.yaml** -- frozen copy of the config used for this run
- **run_manifest.json** -- summary metadata (routes, clusters, replicates)
- **prompts.jsonl** -- full system + user prompts with hashes
- **raw_outputs.jsonl** -- raw LLM response text, tool calls, steps
- **parsed_outputs.jsonl** -- parsed JSON from model responses
- **metrics.jsonl** -- per-run metric records
- **report.md** -- human-readable aggregate report
- **aggregate_summary.json / .csv** -- per-route summary stats
- **prompts_used/** -- deduplicated system prompt .txt files (one per route)
- **traces/** -- per-run trace JSONs (full audit trail, step-level for stepwise)
- **{experiment_id}_{route}_{date}.csv** -- gene-level analysis output CSVs (from trace parser)

## Terminology

- **route** -- a frozen `Route` dataclass specifying mode, MCP toggle, delivery mechanism, and an ordered tuple of prompt components. Named after the assembly paths in `make_cluster_analysis_system_prompt` (see `prompt-assembly-routes-info.md`).
- **mode** -- how the prompt is structured: *standard* (flat concatenation), *cot* (numbered chain-of-thought steps, single call), or *stepwise* (multi-turn, one API call per step).
- **delivery** -- *single_call* or *multi_turn*. Stepwise routes use multi_turn; standard and cot use single_call.
- **component_order** -- the ordered tuple of shorthand keys (CAT, SC, GCR, NPR, UPR, PCC, O, cPH, cGCR, cPri, cPSC, cVer, cO, LIT) that defines what goes into the prompt and in what sequence.
- **variant** -- a named perturbation of component_order relative to the canonical baseline (Phase 2 only). Defined in `order_bench_orderings.py`.
- **base route** -- the Phase 1 route (e.g. single_call, cot) from which an order variant is derived. The canonical variant preserves the base route's original component_order.
- **replicate** -- repeated execution of the same prompt on the same input. Used to measure reasoning stability at a given temperature.

## Benchmark Phases

Phase 1 -- Architecture: Compares zeroshot prompting, CoT prompting, and stepwise CoT prompting toggling mcp on and off. Mechanistically, this orchestrates 6 routes across 3 modes (standard, cot, stepwise) each with and without MCP. Routes single_call/single_call_mcp use flat concatenation, cot/cot_mcp use numbered CoT steps in a single call, and stepwise/stepwise_mcp deliver steps as separate API turns. Run all 6 with `arch_bench_default.yaml` or pick a subset in a custom config.

Phase 2 -- Order: Holds the mode and MCP constant while permuting component_order. Five variants (canonical + 4 perturbations) are crossed against one or more base routes. Each perturbation tests a specific hypothesis about positional sensitivity. Run with any `order_bench_*.yaml` config.

More detail on the philosophy behind running these phases is written up in MLLM Benchmarking Plan_3_10_26.docx

## Running Benchmarks

Experiments are yaml-driven: one yaml describes one whole experiment (shared model/run
regime + the conditions that vary), and one function runs it end to end -- every condition
through the engine, scored against reviewer-consensus GT, controls validated, the selection
rule applied, and the experiment's state file written (the only metric output).

```bash
python -m tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_experiment \
    tests/phase1_prompt_benchmarking/experiments/source.yaml [--dry-run | --score-only]

# staged experiments (the walkup): one stage per invocation, human-gated selection
... bench_experiment experiments/walkup.yaml --stage CAT [--source affinage]
... bench_experiment experiments/walkup.yaml --select CAT process_guarded
```

- **--dry-run** -- exercise the full plumbing on mock outputs (zero API cost)
- **--score-only** -- re-score the newest archived run dirs, no API calls
- **--stage / --select / --source** -- staged experiments only (see `experiments/walkup.yaml`)

## Experiments

- **experiments/source.yaml** -- uniprot vs affinage evidence on the blank W0 floor; winner carried as `carry.source`
- **experiments/walkup.yaml** -- staged build-up (CAT -> GCR -> NPR -> UPR -> PCC) on the carried source; candidates + rationales live in the yaml
- **experiments/mode.yaml** -- single_call vs cot vs stepwise on the walkup's final build
- **experiments/order.yaml** -- component-order permutations (O-O4) of the tuned single_call prompt

Runs archive under `benchmarking_outputs/<experiment>/<condition-or-stage>_<stamp>/`
(never overwritten); state lives at `benchmarking_outputs/<experiment>/<experiment>_state.json`.

## Running the Trace Parser

After a benchmark run, use the trace parser to extract gene-level predictions from the trace JSONs into aligned CSVs for downstream analysis.

**Usage:**
```bash
python -m tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_trace_parser \
    --traces-dir <path_to_traces_dir> \
    --output-dir <path_to_output_dir> \
    [--experiment-id <experiment_id>] \
    [--overwrite]
```

**Parameters:**
- **--traces-dir** -- path to the `traces/` directory from a benchmark run (e.g. `benchmarking_outputs/2.order/order_bench_full_v1/traces/`)
- **--output-dir** -- where to write the gene-level CSVs (usually the same directory as the traces)
- **--experiment-id** -- optional; auto-detected from trace filenames if not provided
- **--overwrite** -- overwrite existing CSVs; otherwise appends date to filename

**Output:**
Creates one CSV per route per experiment:
- `{experiment_id}_{route}.csv` (if --overwrite)
- `{experiment_id}_{route}_{YYYYMMDD}.csv` (otherwise)

Each CSV contains columns: screen_name, cluster_id, gene_symbol, route, replicate, run_id, predicted_class, predicted_subclass, rationale, evidence, pathway, pathway_confidence, source_trace_path

## Screens

More info on screens can be found in their respective context JSON files in `benchmark_inputs/`.

## Preprocessing

`phase1_benchmark_preprocess.py` was run to generate evidence bundles from the benchmark cluster CSV and screen contexts. This calls the standard bundle-building pipeline with `flat_output=True`. Can be rerun if benchmark dataset is updated.
