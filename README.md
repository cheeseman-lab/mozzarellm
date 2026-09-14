# MozzareLLM

LLM-based analysis of gene clusters from functional genomics screens: pathway
identification, evidence-grounded gene categorization, and prioritization of
understudied genes for experimental follow-up.

## What it does

Given a cluster table (one row per gene, with a cluster assignment) and a short
description of your screen, MozzareLLM builds one evidence bundle per cluster —
functional annotations for every gene, optionally your per-gene phenotypic
features — and asks a large language model to:

1. **Identify the pathway(s)** that best explain why the genes cluster together,
   or state explicitly that no coherent pathway exists.
2. **Categorize every gene relative to that pathway**: ESTABLISHED (documented
   role in the pathway), NOVEL_ROLE (documented function elsewhere — membership
   here is the new evidence), or UNCHARACTERIZED (annotation too sparse to
   judge), each with an evidence-ladder subclass and a written rationale.
3. **Prioritize the understudied genes** — the NOVEL_ROLE and UNCHARACTERIZED
   calls are the deliverable: candidates for follow-up experiments, with the
   evidence trail that produced each call.

The model is used as a recall engine over the assembled evidence, not as an
oracle: every call is grounded in the bundle contents, abstention on incoherent
clusters is a valid outcome, and the full per-cluster record (prompt, response,
literature tool calls, tokens, cost) is written to disk for auditing.

## Installation

Into an existing environment (Python 3.11+); a PyPI release is planned, until
then install from the `cot-mcp` branch:

```bash
python -m pip install "mozzarellm @ git+https://github.com/cheeseman-lab/mozzarellm.git@cot-mcp"
```

Claude models work out of the box; add `[openai]` or `[gemini]` to the
requirement for those providers (`"mozzarellm[openai] @ git+..."`). Use
`python -m pip` explicitly so the install lands in the active interpreter.
For development, clone the repo and use `environment.yml`
(`conda env create -f environment.yml && python -m pip install -e ".[dev]"`).

Add an Anthropic API key (OpenAI and Google models are also supported) to a
`.env` file at the repo root:

```
ANTHROPIC_API_KEY=...
```

## Quick start

Open `examples/analyze_clusters.ipynb`. It walks through setup and three
worked examples — optical pooled screening clusters with phenotypic features,
DepMap co-essentiality modules on the baseline path, and proteomics
co-abundance clusters with PubMed literature validation — and shows how to
point the same two calls at your own screen:

```python
from mozzarellm.clients.llm_api_clients import create_client
from mozzarellm.pipeline.screen_analysis import analyze_screen, prepare_screen_bundles

bundles = prepare_screen_bundles(
    screen_name="my_screen",
    cluster_table="my_clusters.csv",          # gene_symbol, cluster [, feature columns]
    output_dir="output",
    feature_columns=["up_features", "down_features"],  # optional
)
run = analyze_screen(
    screen_name="my_screen",
    cluster_to_bundle_map=bundles,
    client=create_client(model="claude-sonnet-5"),
    run_dir="output/my_screen_analysis/run_01",
    screen_context_path="screen_context.json",  # from examples/screen_context_template.json
    mode="cot",
    include_features=True,                      # requires feature columns
    max_workers=8,                              # clusters analyzed concurrently
)
run["cluster_df"]  # pathway call, confidence, per-category counts per cluster
run["gene_df"]     # one row per gene: category, subclass, rationale, evidence
```

## Outputs

Each run directory contains the complete record and the tables to read:

| File | Contents |
|---|---|
| `<screen>_clusters.csv` | one row per cluster: `cluster_id`, `dominant_process`, `pathway_confidence`, `summary`, the phenotype verdicts when the run had that data (`feature_signature`, `pathway_consistency`, `phenotype_strength`, `confidence_revision`), `n_genes`, `n_classified`, `n_established`, `n_novel_role`, `n_uncharacterized`, per-category gene lists (`;`-joined), `missed_genes`, `classification_completeness` |
| `<screen>_genes.csv` | one row per gene: `gene`, `cluster_id`, `category`, `subclass`, `rationale`, `evidence`, `dominant_process`, `pathway_confidence`, plus `phenotype_strength_rank` (`"N/M"`) when a `strength_column` was given |
| `<screen>_clusters.json` | parsed structured output per cluster; `metadata.schema_version` identifies the structure for downstream readers |
| `traces/cluster_<id>.json` | full per-call record: raw response, tool calls, tokens, cost |

A successful run also writes `latest.json` next to the run directory
(`{"run_dir", "date", "screen_name"}`), so downstream code can find the newest
run without parsing timestamps. The screen context can be passed as a file
(`screen_context_path`) or an in-memory dict (`screen_context`).
`analyze_screen(..., resume=True)` re-reads clusters already answered in that
`run_dir` instead of calling the model again; `dry_run=True` writes every
prompt under `run_dir/prompts_used/` and returns per-cluster input-token and
cost estimates with no API call. `organism_id` (default 9606; 10090 for mouse)
restricts the UniProt lookups. `max_workers` (default 1) analyzes that many
clusters concurrently — a genome-wide panel of ~930 clusters takes about a day
one at a time — and `results`, `errors`, `resumed` and `total_cost_usd` stay
ordered by `cluster_to_bundle_map`, so a run's tables do not depend on how many
workers it used.

## Analysis options

- **Mode** (`mode`): `standard` (flat single-call prompt), `cot` (chain-of-thought
  reasoning steps, the default), `stepwise` (one API turn per reasoning step).
- **Literature gap-fill** (`mcp=True`): the model is given PubMed search tools
  and an evidence-gated step that looks up only the genes whose annotation is
  blank (two tool calls at most). The category-gated variant, which checks
  NOVEL_ROLE and UNCHARACTERIZED calls against the literature, is available
  as `component_overrides={"LIT": COMPONENTS["LITV"]}` (from `mozzarellm.prompts`).
- **Phenotypic features**: pass `feature_columns` (up/down lists per gene —
  imaging features, DE genes, anything list-shaped) and optionally
  `strength_column` (any perturbation-strength metric; converted to scale-free
  `"N/M"` ranks, 1 = strongest vs non-targeting controls) to
  `prepare_screen_bundles`. The model never sees the per-gene lists: each
  bundle carries a bounded cluster-level `feature_coherence` table (features
  covering ≥25% of the cluster, at most 100, with their supporting genes), so
  the prompt cannot grow with the screen's feature space. The matching
  reasoning steps enter the prompt automatically when — and only when — the
  bundles carry the data (`include_features` / `include_strength` default to
  `"auto"`): features add
  a bounded consistency cross-check against the pathway call, strength adds a
  cluster-level informativeness verdict (strong / mixed / weak vs controls);
  neither overturns the call. Describe what your columns mean in
  `screen_context.json` under `phenotype_readout` — the model reads your
  description verbatim. Supported for `mode="cot"` (with or without MCP).
- **Prompt customization**: every text the model sees is a named component in
  `mozzarellm/prompts/components.py`; `mozzarellm/prompts/assembly.py` joins
  them in the default chain for each mode. Reword any component per run with
  `component_overrides={key: text}`, or run your own chain with
  `component_order=[...]` (see the notebook's customization section). The
  shipped wording is the benchmark-selected build, not a constraint.

## Repository layout

- `mozzarellm/` — the package: LLM clients, prompts (`prompts/components.py`
  is the wording, `prompts/assembly.py` the chain), bundle builder, screen
  analysis.
- `examples/` — the analysis notebook, the screen-context template, and the three
  example datasets (OPS, DepMap, proteomics).
- `benchmarks/` — the prompt/evidence benchmarking suite that produced the
  shipped prompt configuration (see `benchmarks/README.md`).
- `tests/` — the test suite (`python -m pytest tests/`).

## License

MIT — see `LICENSE`.
