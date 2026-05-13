# Phase 1 Prompt Benchmarking

Systematic evaluation of prompt architecture, component ordering, and wording sensitivity for the MozzareLLM cluster analysis pipeline.

## Directory 

- **benchmark_inputs/** -- Cluster CSV and screen context JSONs (4 screens, 14 clusters)
- **benchmark_evidence_bundles/** -- Pre-built evidence bundle JSONs
- **architecture_benchmarking_workflow/** -- Core benchmark orchestrator, metrics, reporting, and route definitions
- **configs/** -- YAML configurations for all benchmark runs
- **benchmarking_outputs/** -- Generated outputs organized by phase:
  - *1.architecture* -- Phase 1 architecture comparison (6 routes)
  - *2.order* -- Phase 2 order sensitivity (5 variants per base route)

## Benchmark Phases
relevant terms used throughout this benchmark effort:
- route: a specific combination of prompt components and their order as coordinated by make_cluster_analysis_system_prompt function --- see `prompt-assembly-routes-info.md` for more detailed information.
- variant: a specific combination of prompt components and their order
- base route: the canonical ordering of prompt components

### Phase 1 -- Architecture

- note routes and relevant command to run them all possible combinations

### Phase 2 -- Order

Tests prompt-component order sensitivity using hypothesis-driven perturbations (Planning doc has details). Each variant is applied to one or more base routes and compared against the canonical ordering.

## Running Benchmarks

All benchmarks are run via the orchestrator CLI:

```
python architecture_benchmarking_workflow/bench_orchestrator.py --config configs/<config_file>.yaml
```

Use `--dry-run` to validate infrastructure without API calls.

## Config Files

- **arch_bench_default.yaml** -- All 6 routes, all screens, 3 replicates
- **arch_bench_dry_run_test.yaml** -- Dry-run validation (2 routes, denali, 1 rep)
- **arch_bench_full_run_test_denali.yaml** -- Live test on denali only - routes to _workflow_output
- **order_bench_default.yaml** -- Order benchmark, single base route, all variants
- **order_bench_dry_run_test.yaml** -- Order dry-run validation
- **order_bench_full_run_test_denali.yaml** -- Order full run on denali

## Screens

- **aconcagua_interphase** -- HeLa, 4-channel, interphase, 8 clusters
- **denali** -- HeLa, 2-channel, interphase, 2 clusters
- **jebel** -- RPE1, 5-channel, all phases, 1 cluster
- **whitney** -- HeLa, 4-channel, all phases, 3 clusters
more info on screens can be found in their respective context JSON files in `benchmark_inputs/`.

## Preprocessing

`phase1_benchmark_preprocess.py` was run to generate evidence bundles from the benchmark cluster CSV and screen contexts. This calls the standard bundle-building pipeline with `flat_output=True`. Can be rerun if benchmark dataset is updated.
