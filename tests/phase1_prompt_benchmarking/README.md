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
        bench_orchestrator.py         -- Main loop, CLI, prompt construction, execution
        bench_configparse.py          -- YAML config loader + dataclass sections
        bench_metricfns.py            -- Structural, MCP, logical, efficiency metrics
        bench_reportgen.py            -- Markdown report + CSV/JSON aggregates
        bench_trace_parser.py         -- Trace JSON -> gene-level prediction CSVs
        bench_dry_run.py              -- Deterministic mock outputs for dry-run
        order_bench_orderings.py      -- Order variant definitions + route builder

    configs/
        arch_bench_default.yaml
        arch_bench_dry_run_test.yaml
        arch_bench_full_run_test_denali.yaml
        order_bench_default.yaml
        order_bench_dry_run_test.yaml
        order_bench_full_run_test_denali.yaml
        your-custom-config.yaml

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
- **base route** -- the Phase 1 route (e.g. 3a, 3b) from which an order variant is derived. The canonical variant preserves the base route's original component_order.
- **replicate** -- repeated execution of the same prompt on the same input. Used to measure reasoning stability at a given temperature.

## Benchmark Phases

Phase 1 -- Architecture: Compares zeroshot prompting, CoT prompting, and stepwise CoT prompting toggling mcp on and off. Mechanistically, this orchestrates 6 routes across 3 modes (standard, cot, stepwise) each with and without MCP. Routes 3a/3a_mcp use flat concatenation, 3b/3b_mcp use numbered CoT steps in a single call, and 3c/3c_mcp deliver steps as separate API turns. Run all 6 with `arch_bench_default.yaml` or pick a subset in a custom config.

Phase 2 -- Order: Holds the mode and MCP constant while permuting component_order. Five variants (canonical + 4 perturbations) are crossed against one or more base routes. Each perturbation tests a specific hypothesis about positional sensitivity. Run with any `order_bench_*.yaml` config.

More detail on the philosophy behind running these phases is written up in MLLM Benchmarking Plan_3_10_26.docx

## Running Benchmarks

All benchmarks are run via the orchestrator CLI:

**Usage:**
```bash
python architecture_benchmarking_workflow/bench_orchestrator.py --config configs/<config_file>.yaml [--dry-run]
```

**Parameters:**
- **--config** (required) -- path to YAML config file
- **--dry-run** (optional) -- override config to enable dry-run mode; uses mock outputs instead of API calls

**Example:**
```bash
# Live run with full architecture benchmark
python architecture_benchmarking_workflow/bench_orchestrator.py --config configs/arch_bench_default.yaml

# Dry-run validation of order benchmark
python architecture_benchmarking_workflow/bench_orchestrator.py --config configs/order_bench_dry_run_test.yaml --dry-run
```

## Existing Config Files

- **arch_bench_default.yaml** -- All 6 routes, all screens, 3 replicates
- **arch_bench_dry_run_test.yaml** -- Dry-run validation (2 routes, denali, 1 rep)
- **arch_bench_full_run_test_denali.yaml** -- Live test on denali only - routes to _workflow_output
- **order_bench_default.yaml** -- Order benchmark, single base route, all variants
- **order_bench_dry_run_test.yaml** -- Order dry-run validation
- **order_bench_full_run_test_denali.yaml** -- Order full run on denali

## Config Parameters

All parameters below are YAML keys. Defaults are shown in parentheses. Parsed by `bench_configparse.py`.

**experiment_id** (arch_bench_v1) -- unique name for this run. Determines the output subdirectory name. Please include either `arch_bench_` or `order_bench_` in the name to keep things organized.

**paths:**

---likely constant in most cases---
- **benchmark_inputs_dir** (benchmark_inputs) -- directory containing screen context JSONs
- **benchmark_clusters_csv** (benchmark_inputs/benchmark_clusters.csv) -- CSV with columns: screen_name, cluster_id, gene_symbol
- **evidence_bundles_dir** (benchmark_evidence_bundles) -- directory of pre-built evidence bundle JSONs

----important to adjust as needed----
- **output_dir** (benchmarking_outputs/1.arch) -- root output directory. 

**model:**
- **provider** (anthropic) -- LLM provider
- **model_name** (claude-sonnet-4-5) -- model identifier
- **temperature** (0.2) -- sampling temperature
- **max_tokens** (4000) -- max output tokens
- **top_p** (null) -- nucleus sampling. Leave null to use provider default.
- **top_k** (null) -- top-k sampling. Leave null to use provider default.

**run:**
- **num_replicates** (1) -- how many times to run each route x cluster pair
- **dry_run** (false) -- if true, uses mock outputs instead of calling the API
- **workflow_testing** (false) -- if true, outputs go under `_workflow_testing/` subdirectory
- **overwrite_outputs** (false) -- if true, truncates existing JSONL files on start
- **continue_on_error** (true) -- if true, logs errors and continues to next run rather than aborting
- **save_prompts** (true) -- write prompts.jsonl
- **save_raw_outputs** (true) -- write raw_outputs.jsonl
- **save_parsed_outputs** (true) -- write parsed_outputs.jsonl
- **save_traces** (true) -- write per-run trace JSONs


**routes:** <--Phase 1 Only
- **include** (all 6) -- list of route names to run: 3a, 3a_mcp, 3b, 3b_mcp, 3c, 3c_mcp. 

**screens:** 
- **include** ("all" or list) -- which screens to include. Use "all" or a list like `[denali, whitney]`.

**clusters:**
- **include** ("all" or list) -- use "all" for every cluster in the CSV, or a list of `{screen_name, cluster_id}` objects for selective inclusion.

**mcp:**
- **preflight** (true) -- run MCP server availability check before starting
- **fail_if_unavailable** (false) -- if true, abort when MCP is unreachable. If false, skip MCP routes gracefully.

**evaluation:**
- **structural** (true) -- compute structural metrics (schema compliance, gene completeness, etc.)
- **logical_consistency** (true) -- compute logical metrics (duplicates, mutual exclusivity)
- **efficiency** (true) -- compute token/cost/latency metrics
- **robustness** (true) -- reserved for future robustness metrics

**timing:**
- **track_full_run** (true) -- total wall time per run
- **track_prompt_construction** (true) -- time spent building prompts
- **track_model_latency** (true) -- time waiting on the LLM
- **track_metrics** (true) -- time computing metrics
- **track_io** (true) -- time writing artifacts
- **track_step_latencies** (true) -- per-step timing for stepwise routes
- **track_mcp_tool_latency** (true) -- MCP tool call timing

**order_benchmark:** <--Phase 2 Only
- **enabled** (false) -- when true, ignores `routes.include` and instead builds routes from base_routes x variants
- **base_routes** -- list of Phase 1 route names to permute (e.g. [3a, 3b])
- **variants** -- list of variant names: canonical, late_screen_context, prioritization_before_classification, early_output_format, delayed_task_anchor

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
