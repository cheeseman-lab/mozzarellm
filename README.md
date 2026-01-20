# mozzarellm

Analyze gene clusters using Large Language Models (LLMs) to identify biological pathways and prioritize novel genes.

## Overview

Mozzarellm is a Python package that leverages Large Language Models to analyze gene clusters for pathway identification and novel gene discovery. It provides a streamlined way to process gene cluster data, classify genes by their pathway involvement, and prioritize candidates for experimental validation.

## Features

- 🧬 **Pathway Identification**: Automatically identify the dominant biological process or pathway in gene clusters
- 🔍 **Gene Classification**: Categorize genes as established pathway members, uncharacterized, or having novel potential roles
- 🏆 **Prioritization**: Assign importance scores to novel gene candidates to guide experimental follow-up
- 🚀 **Multi-Provider Support**: Use OpenAI (GPT-4/o), Anthropic (Claude), or Google (Gemini) models
- 📊 **Structured Output**: Generate JSON and CSV files with detailed analysis at gene and cluster levels

## Installation

```bash
# Create conda environment with uv
conda create -n mozzarellm -c conda-forge python=3.11 uv pip -y
conda activate mozzarellm

# Install dependencies, then package in editable mode
uv pip install -r pyproject.toml
uv pip install -e .
```

Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
```

## Quick Start

```python
import pandas as pd
from mozzarellm import ClusterAnalyzer, reshape_to_clusters

# Load gene-level data (e.g., from clustering output)
gene_df = pd.read_csv("clustering_results.tsv", sep="\t")

# Reshape to cluster format
cluster_df, gene_annotations = reshape_to_clusters(
    input_df=gene_df,
    gene_col="gene_symbol",
    cluster_col="cluster",
    uniprot_col="uniprot_function",  # optional but recommended
    verbose=True,
)

# Initialize analyzer
analyzer = ClusterAnalyzer(model="claude-sonnet-4-5-20250929")

# Run analysis with incremental saving (recommended for production)
results = analyzer.analyze(
    cluster_df,
    gene_annotations=gene_annotations,
    screen_context="Optical pooled screen for mitotic regulators in HeLa cells",
    output_dir="results/mozzarellm",  # enables resume + incremental saving
)

# Results are automatically saved to output_dir:
#   - results/mozzarellm/clusters/          (individual cluster JSONs)
#   - results/mozzarellm/{model}_results.json
#   - results/mozzarellm/{model}_summaries.tsv
#   - results/mozzarellm/{model}_flagged_genes.tsv
```

### Resume Support

When `output_dir` is provided, the analyzer automatically:
- **Saves each cluster immediately** after analysis (crash-safe)
- **Skips already-completed clusters** on restart (resume support)
- **Generates combined outputs** at the end

This means you can safely interrupt and restart long-running analyses.

## Input Format

Your input should be a CSV/TSV file with gene clusters:

| cluster_id | genes                    | uniprot_function                           |
|------------|--------------------------|-------------------------------------------|
| 1          | BRCA1;TP53;PTEN;MLH1     | DNA repair; tumor suppressor; metabolism   |
| 2          | MYC;MAX;MYCN;E2F1;RB1    | transcription factors; cell cycle control  |

Required columns:
- A column with cluster identifiers (default: `cluster_id`)
- A column with gene symbols (default: `genes`)
- Optionally, a column with UniProt function descriptions (helps with analysis)

## Supported Models

Mozzarellm supports models from three providers, auto-detected by model name prefix:

| Provider | Prefix | Examples |
|----------|--------|----------|
| Anthropic | `claude*` | `claude-sonnet-4-5-20250929`, `claude-3-5-haiku-latest` |
| OpenAI | `gpt*`, `o1*`, `o3*`, `o4*` | `gpt-4o`, `o4-mini` |
| Google | `gemini*` | `gemini-2.5-pro`, `gemini-2.5-flash` |

Pass any valid model identifier from these providers to `ClusterAnalyzer(model="...")`.

## Python API

### ClusterAnalyzer

The main interface for analyzing gene clusters:

```python
from mozzarellm import ClusterAnalyzer

analyzer = ClusterAnalyzer(
    model="claude-sonnet-4-5-20250929",  # Model identifier
    temperature=0.0,                      # Temperature (0.0-1.0)
    max_tokens=8000,                      # Max tokens per response
    show_progress=True,                   # Show progress bar
)

results = analyzer.analyze(
    cluster_df,                           # DataFrame with cluster_id and genes columns
    gene_annotations=gene_annotations,    # Optional: DataFrame with gene annotations
    screen_context="...",                 # Optional: Experimental context
    output_dir="results/",                # Optional: Enable incremental saving + resume
)
```

### reshape_to_clusters

Convert gene-level data to cluster format:

```python
from mozzarellm import reshape_to_clusters

cluster_df, gene_annotations = reshape_to_clusters(
    input_df=gene_level_data,          # Gene-level input DataFrame
    gene_col="gene_symbol",            # Gene ID column
    cluster_col="cluster",             # Cluster assignment column
    uniprot_col="uniprot_function",    # Column with gene annotations (optional)
    verbose=True,                      # Show processing information
)
```

## Output Structure

When `output_dir` is provided, the analysis produces:

```
output_dir/
  clusters/                    # Individual cluster results (for resume)
    cluster_0.json
    cluster_1.json
    ...
  {model}_results.json         # Combined analysis with all clusters
  {model}_summaries.tsv        # One row per cluster (pathway, confidence, counts)
  {model}_flagged_genes.tsv    # One row per flagged gene (priority, rationale)
```

The `AnalysisResult` object is also returned with:
- `results.clusters` - Dict mapping cluster_id to ClusterResult
- `results.metadata` - Analysis metadata (model, timestamp, output paths)

## License

MIT License - See LICENSE file for details.