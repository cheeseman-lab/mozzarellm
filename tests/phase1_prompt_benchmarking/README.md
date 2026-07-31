# Phase 1 Prompt Benchmarking

Benchmarks MozzareLLM cluster classification against reviewer-consensus ground truth. The benchmark runs as a four-step **pipeline** — `source → walkup → mode → order` — that decides the evidence source, assembles the prompt, picks the delivery format, and checks positional sensitivity. Each step is a runner script: it scores model output against the consensus ground truth and hands its decision to the next step through a small state file. The runners sit on a shared orchestrator engine (routes, config schema, trace parser) documented under **Engine reference** below.

## The benchmark pipeline (source → walkup → mode → order)

The pipeline answers four questions in order, each on the previous step's winner:

1. **source** — which evidence source (`uniprot` / `affinage` / `both`) gives the best classification recall?
2. **walkup** — starting from a blank prompt, which framing of each component (CAT → GCR → NPR → UPR → PCC) to adopt, added one at a time?
3. **mode** — which delivery format (`single_call` / `cot` / `stepwise`) for the final assembled prompt?
4. **order** — does the component *order* matter? Permute it on the tuned prompt and measure positional sensitivity.

> **Status:** all four runners (`run_source.py`, `run_walkup.py`, `run_mode.py`, `run_order.py`) and the shared evaluator are implemented; they consume the same evaluator and state envelope described here.

### 0. Data prep (one-time, upstream of the runners)

The dataset lives in `benchmark_inputs/` (see `benchmark_inputs/README.md`). Two scripts turn it into what the runners consume:

```bash
# reviewer annotations -> one combined per-reviewer ground-truth table
python merge_reviewers.py

# benchmark_input.csv (roles + per-gene features) -> evidence bundles per source
python build_bundles.py            # writes benchmark_bundles/ (master superset bundles)
```

Ground truth is **not** baked into the dataset — it is computed at scoring time from the raw per-reviewer annotations by `bench_evaluator` (`build_consensus_gt`):
- **classification** → strict **≥2-of-3 majority** (every gene resolves; a 1-1-1 split is unscored)
- **subclass** → **ordinal median** on the consensus category's evidence ladder
- **coherence** → per-cluster majority of the reviewers' High/Medium/Low

### 1. `run_source.py` — pick the evidence source

Runs the blank prompt (every component rule empty — the same floor the walkup builds up from) across `{uniprot, affinage, both}` on the 8-cluster slate (5 real clusters drive selection, 3 decoys are validated), n=3, Sonnet 5.

```bash
python run_source.py                # real run  (real API spend)
python run_source.py --dry-run      # mock outputs, zero API cost
python run_source.py --score-only   # re-score existing run dirs, no API
```

Each cell is scored by `bench_evaluator` and the source is picked **holistically on coverage-weighted recall** (correct categories / 103 real genes) so a source can't win by dropping hard genes to inflate raw category. The evaluator's diagnostics (per-class precision/recall/F1, per-cluster recall, reviewer source-preference, inter-reviewer concordance) explain *why* a source wins.

- **reads:** `benchmark_bundles/` (master bundles; each run's source view is derived at prompt assembly via `strip_source_fields`), reviewer annotations, `configs/source_{source}.yaml`
- **writes:** `benchmarking_outputs/source/source_state.json` with `carry = {"source": <winner>}`

### 2. `run_walkup.py` — assemble the prompt (human-gated)

Reads `carry.source` and builds the prompt from the blank floor by adding one component per stage (`CAT → GCR → NPR → UPR → PCC`). Selection is a **human gate**: several target metrics are too low-powered on this 5-cluster benchmark to auto-select on (unchar-subclass n≈7, coherence n=4), so each stage prints the full metric panel — with the sub-metric n's exposed — and stops for a person to choose.

```bash
python run_walkup.py --stage CAT            # run CAT candidates, print panel, stop
python run_walkup.py --select CAT concise   # record the human's choice, carry it forward
python run_walkup.py --stage GCR            # next stage builds on the carried winner
# ... GCR, NPR, UPR, PCC ...
python run_walkup.py --finalize             # assemble the final prompt + render the figure
```

- **reads:** `carry.source` (or `--source`, else affinage), the candidate banks in `bench_walkup_candidates.py`
- **writes:** `benchmarking_outputs/walkup/walkup_state.json` with the carried per-component winning text

### 3. `run_mode.py` — pick the delivery format

Runs `{single_call, cot, stepwise}` on the walkup's final assembled prompt (8-slate, n=3) and picks the mode holistically. `single_call` runs the full tuned prompt; `cot`/`stepwise` use different component keys, so they carry only the shared tuned `CAT` lens and keep canonical text elsewhere — a fair mode comparison, not a verbatim transplant.

- **reads:** the walkup carry (source + assembled prompt)
- **writes:** `benchmarking_outputs/mode/mode_state.json`

### 4. `run_order.py` — component order (positional sensitivity)

Holds source, mode, and component *content* fixed and permutes the `component_order` across the order variants `[O, O1, O2, O3, O4]` (8-slate, n=3). Picks the order holistically and reports the **cw-recall spread** — how much order alone moves the result. V1 applies order to the `single_call` route with the walkup's assembled texts injected (the keys the walkup tuned); a `cot`/`stepwise` order transplant is out of scope for now.

- **reads:** the mode carry (source + winning mode) + the walkup's component texts
- **writes:** `benchmarking_outputs/order/order_state.json` with the order winner + the cw-recall spread

### How the steps chain (the state envelope)

Every step writes a state file through `bench_pipeline_common.write_state` with a shared envelope, and the next step picks it up with `read_carry`:

```json
{ "step": "source", "source": "affinage", "winner": "affinage::single_call",
  "decoys": {...}, "carry": {"source": "affinage"}, "cells": {...}, "diagnostics": {...} }
```

`carry` is the contract between steps; the `extra` keys (`cells`, `diagnostics`, `stages`, …) are the per-step detail the figures and results table read.

### Scoring (`bench_evaluator`)

`bench_evaluator` is the single metric generator for every step. It builds the consensus ground
truth, scores a run's `parsed_outputs.jsonl` against it (modal vote across replicates; a gene hedged
across category lists resolves to the **less-high** call, UNCHAR > NOVEL > EST), validates the decoy
controls, and computes the source diagnostics and pathway agreement. **Everything the report shows is
written to the state file — the report reads it back with no further derivation.** `bench_figures`
renders the figures from the same state files.

## Metrics

The full panel the evaluator emits, and how much weight each one carries for a source/prompt decision.
"Use" reflects what actually discriminates on this benchmark (103 genes / 4 coherent clusters); the
low-power metrics are kept for transparency but should not drive selection alone.

| Metric | What it measures | In `source_state.json` | Use |
|---|---|---|---|
| **novel-role precision / recall / F1** | the goal: does the model recover genes with a *novel role in this cluster* | `diagnostics.per_class.NOVEL_ROLE` | **primary** — the goal class, the only one that separates sources |
| **reviewer source-preference** | which source's bundle a human reviewer found most useful (a direct human vote, independent of the model) | `source_preference` | **primary** — evidence-quality axis; report alongside, don't average in |
| category / coverage-weighted recall | fraction of all genes whose modal call matches consensus | `cells.category` | diagnostic — diluted by the near-saturated ESTABLISHED/UNCHAR classes |
| per-class recall (EST / NOVEL / UNCHAR), consensus & any-reviewer | recall within each consensus class; `any` = matches any single reviewer | `diagnostics.per_class` | diagnostic |
| per-cluster category recall | recall broken out by cluster | `diagnostics.per_cluster` | diagnostic — with 4 clusters the aggregate is **not** cluster-robust |
| decoy validation (abstain / functional) | abstain (Low confidence) on shuffled nonsense; stay functional (High/Med) on a large coherent cluster | `decoys[*].passed` | keep — sanity/robustness control |
| decoy completion | genes classified vs the cluster's true gene count on the functional decoy (output fragility) | `decoys[*].completion` | keep — surfaces dropped/hallucinated genes |
| pathway agreement (substring / loose / semantic) | model `dominant_process` vs reviewers' nominated pathways | `pathway` | diagnostic — small n (per cluster) |
| inter-reviewer concordance / Fleiss κ | ground-truth quality and the recall ceiling (source-independent) | `reviewer_concordance` | context |
| subclass accuracy (novel / unchar) | evidence-ladder sub-call vs consensus | `cells.novel_subclass` / `unchar_subclass` | low signal — reviewer subclass agreement is very low |
| coherence | modal cluster confidence vs consensus High/Med/Low | `cells.coherence` | low signal — 4 clusters, no source separation |

## Directory

```
phase1_prompt_benchmarking/
    README.md
    prompt-assembly-routes-info.md

    run_source.py                     -- pipeline step 1: pick the evidence source
    run_walkup.py                     -- pipeline step 2: assemble the prompt (human-gated)
    run_mode.py                       -- pipeline step 3: pick the delivery format
    run_order.py                      -- pipeline step 4: permute component order
    merge_reviewers.py                -- data prep: reviewer annotations -> combined GT table
    build_bundles.py                  -- data prep: benchmark_input.csv -> evidence bundles

    benchmark_inputs/                 -- the dataset (see benchmark_inputs/README.md)
        README.md
        benchmark_input.csv           -- roles + per-gene up/down features + strength
        benchmark_ground_truth.csv    -- combined per-reviewer annotations (merge_reviewers.py)
        ground_truth/                 -- annotation_{eric,liz,iain}.csv + survey_key.csv
        {screen}_screen_context.json  -- per-screen context

    benchmark_bundles/                -- master evidence bundles (superset; filtered per source at assembly)

    architecture_benchmarking_workflow/
        # --- pipeline (source/walkup/mode) ---
        bench_evaluator.py            -- THE metric generator: consensus GT + scoring + diagnostics + pathway
        bench_pipeline_common.py      -- shared paths, GT loading, state envelope, holistic selection
        bench_figures.py              -- figures rendered from the state files
        bench_walkup_candidates.py    -- per-component candidate banks for the walkup (step 2)
        # --- shared orchestrator engine ---
        bench_routes.py               -- Route dataclass + mode registry
        bench_orchestrator.py         -- main loop, prompt construction, execution
        bench_configparse.py          -- YAML config loader + dataclass sections
        bench_metricfns.py            -- structural / MCP / logical / efficiency metrics
        bench_reportgen.py            -- markdown report + CSV/JSON aggregates
        bench_dry_run.py              -- deterministic mock outputs for dry-run
        bench_order.py                -- component-order variant definitions + route builder
        bench_wording_alternates.py   -- wording alternate-text registry (walkup dep)

    configs/
        source_{uniprot,affinage,both}.yaml   -- one per source (run_source)

    benchmarking_outputs/             -- git submodule (runs archive here; nothing overwritten)
        source/source_state.json      -- step 1 decision + carry
        walkup/walkup_state.json      -- step 2 assembled prompt + carry
        mode/mode_state.json          -- step 3 decision
        order/order_state.json        -- step 4 decision + cw-recall spread
```

---

# Engine reference

Everything below documents the shared orchestrator the pipeline runners sit on — the route/mode model, per-run output layout, config schema, and trace parser. The `source → walkup → mode → order` runners are the entry points; they build their RunSpecs and drive the orchestrator's run loop directly (there is no separate orchestrator CLI).

## Output structure (`benchmarking_outputs/` submodule)

Every runner writes into the `benchmarking_outputs/` submodule under `benchmarking_outputs/<step>/`. Each run gets its own **timestamped** directory — `<label>_<stamp>/`, where the stamp is baked into the run's `experiment_id` so nothing is ever overwritten — plus a stable `<step>_state.json` latest-pointer that `read_carry` and the figures read.

```
benchmarking_outputs/
    source/
        source_state.json             -- latest-pointer (read by read_carry / the figures)
        <source>_<stamp>/             -- one archived run per source, e.g. affinage_20260731_101500
            ...                       -- the per-run outputs below
    walkup/  mode/  order/            -- same shape: one <step>_state.json + timestamped run dirs
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
- **variant** -- a named perturbation of component_order relative to the canonical baseline (order axis only). Defined in `bench_order.py`.
- **base route** -- the route (e.g. `single_call`) from which an order variant is derived. The canonical variant preserves the base route's original component_order.
- **replicate** -- repeated execution of the same prompt on the same input. Used to measure reasoning stability at a given temperature.

## Component libraries (order / wording)

`bench_order.py` (component-order variant definitions) and `bench_wording_alternates.py` (alternate
component text) are libraries reused by the pipeline — `bench_order` by the order phase,
`bench_wording_alternates` by the walkup. There is no standalone engine CLI: the `run_*` pipeline
scripts are the only entry points, and they build their RunSpecs and drive `_run_benchmark_loop`
directly.

## Config files

- **source_{uniprot,affinage,both}.yaml** -- per-source configs for `run_source.py` (the pipeline)

## Config Parameters

All parameters below are YAML keys. Defaults are shown in parentheses. Parsed by `bench_configparse.py`.

**experiment_id** (source_affinage) -- unique name for this run. Determines the output subdirectory name. The pipeline runners set this automatically (e.g. `run_source` sets it to the source name); it must stay slash-free.

**paths:**

---likely constant in most cases---
- **benchmark_inputs_dir** (benchmark_inputs) -- directory containing screen context JSONs
- **benchmark_clusters_csv** (benchmark_inputs/benchmark_clusters.csv) -- CSV with columns: screen_name, cluster_id, gene_symbol
- **evidence_bundles_dir** (benchmark_evidence_bundles) -- directory of pre-built evidence bundle JSONs

----important to adjust as needed----
- **output_dir** (benchmarking_outputs) -- root output directory. The pipeline runners point this at `benchmarking_outputs/<step>` and bake a `<stamp>` into experiment_id; the config value is only a fallback.

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
- **--traces-dir** -- path to the `traces/` directory from a benchmark run (e.g. `benchmarking_outputs/source/affinage_<stamp>/traces/`)
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

Evidence bundles are generated from `benchmark_inputs/benchmark_input.csv` + the screen contexts by `build_bundles.py` (see the pipeline data-prep step above), which calls the standard bundle-building pipeline. Rerun it if the benchmark dataset is updated.
