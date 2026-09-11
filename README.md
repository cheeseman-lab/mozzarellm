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

```bash
git clone https://github.com/cheeseman-lab/mozzarellm.git
cd mozzarellm
conda env create -f environment.yml
conda activate mozzarellm
pip install -e .
```

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
)
run["cluster_df"]  # pathway call, confidence, per-category counts per cluster
run["gene_df"]     # one row per gene: category, subclass, rationale, evidence
```

## Outputs

Each run directory contains the complete record and the tables to read:

| File | Contents |
|---|---|
| `<screen>_clusters.csv` | one row per cluster: `cluster_id`, `dominant_process`, `pathway_confidence`, `summary`, `n_genes`, `n_classified`, `n_established`, `n_novel_role`, `n_uncharacterized`, per-category gene lists (`;`-joined), `missed_genes`, `classification_completeness` |
| `<screen>_genes.csv` | one row per gene: `gene`, `cluster_id`, `category`, `subclass`, `rationale`, `evidence`, `dominant_process`, `pathway_confidence` |
| `<screen>_clusters.json` | parsed structured output per cluster; `metadata.schema_version` identifies the structure for downstream readers |
| `traces/cluster_<id>.json` | full per-call record: raw response, tool calls, tokens, cost |

A successful run also writes `latest.json` next to the run directory
(`{"run_dir", "date", "screen_name"}`), so downstream code can find the newest
run without parsing timestamps. The screen context can be passed as a file
(`screen_context_path`) or an in-memory dict (`screen_context`).

## Analysis options

- **Mode** (`mode`): `standard` (flat single-call prompt), `cot` (chain-of-thought
  reasoning steps, the default), `stepwise` (one API turn per reasoning step).
- **Literature validation** (`mcp=True`): the model is given PubMed search tools
  and a validation step that checks its NOVEL_ROLE and UNCHARACTERIZED calls
  against retrieved literature.
- **Phenotypic features** (`include_features=True`): per-gene feature columns
  from your screen enter the bundles, and feature-interpretation reasoning
  steps require the pathway call to be consistent with the observed phenotypes.
  Currently supported for `mode="cot"` without MCP.
- **Prompt customization** (`component_overrides={key: text}`): replace the
  wording of any prompt component per run (see the notebook's customization
  section); the shipped components are the benchmark-selected defaults, not a
  constraint.

## Repository layout

- `mozzarellm/` — the package: LLM clients, prompt components, bundle builder,
  screen analysis.
- `examples/` — the analysis notebook, the screen-context template, and the three
  example datasets (OPS, DepMap, proteomics).
- `benchmarks/` — the prompt/evidence benchmarking suite that produced the
  shipped prompt configuration (see `benchmarks/README.md`).
- `tests/` — the test suite (`python -m pytest tests/`).

## License

MIT — see `LICENSE`.
