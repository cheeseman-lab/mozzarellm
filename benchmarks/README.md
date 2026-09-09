# Phase 1 Prompt Benchmarking

Benchmarks MozzareLLM's gene-cluster classification prompts against expert ground truth.
Eight fixed clusters (103 real genes + 3 negative-control decoys + 1 low-coherence abstain
cluster) are classified by the model and scored against a ≥2-of-3 reviewer consensus built
from the raw annotation sheets. The headline metric is coverage-weighted category recall
(correct consensus categories / all 103 real genes), so accuracy cannot be inflated by
dropping hard genes.

## How it works

One yaml describes one whole experiment (shared model/run regime + the conditions that
vary); one function runs it end to end — every condition through the engine, scored by
the evaluator, controls validated, the selection rule applied, and the experiment's state
file written. **State files are the only metric output**; nothing downstream re-derives.

Experiments chain through `carry`/`uses:`: source picks the evidence source, the walkup
builds the prompt on it one component at a time (human-gated per stage), mode compares
delivery formats on the final build, order permutes its component order.

```bash
python -m benchmarks.workflow.bench_experiment \
    benchmarks/experiments/source.yaml [--dry-run | --score-only]

# staged experiments (the walkup): one stage per invocation, human-gated selection
... bench_experiment experiments/walkup.yaml --stage CAT [--source affinage]
... bench_experiment experiments/walkup.yaml --select CAT process_guarded
```

`--dry-run` exercises the full plumbing on mock outputs (zero API cost); `--score-only`
re-scores the newest archived run dirs without API calls.

## Layout

- `experiments/` — the experiment yamls: `source` (uniprot vs affinage on the blank W0
  floor), `walkup` (staged build-up CAT→GCR→NPR→UPR→PCC; candidates + rationales live in
  the yaml), `mode` (single_call vs cot vs stepwise), `order` (O–O4 permutations).
- `workflow/` — the code: `bench_experiment` (experiment layer:
  yaml → runs → scoring → state), `bench_orchestrator` (engine: shared execution loop),
  `bench_evaluator` (the single metric generator: consensus GT, panels, decoys,
  diagnostics), `bench_routes` (route registry), `bench_configparse` (config dataclasses),
  `bench_orderings` (O-variant catalog), `bench_dry_run`, `bench_metricfns`.
- `inputs/` — `benchmark_input.csv` (one row per screen/cluster/gene, with roles
  and phenotypic features), `ground_truth/annotation_{eric,liz,iain}.csv` (raw reviewer
  sheets) + `survey_key.csv` (source-blinding key), per-screen context JSONs.
- `bundles/` — master evidence bundles (both sources' annotations; reduced to
  one source's view at prompt assembly). Rebuilt by `build_bundles.py`;
  `merge_reviewers.py` merges the reviewer sheets.
- `outputs/` — submodule. Runs archive under
  `<experiment>/<condition-or-stage>_<stamp>/` (never overwritten); state at
  `<experiment>/<experiment>_state.json`.
