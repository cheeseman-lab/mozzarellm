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
# From PyPI (once published)
pip install mozzarellm

# From GitHub
pip install git+https://github.com/cheeseman-lab/mozzarellm.git

# For development
git clone https://github.com/cheeseman-lab/mozzarellm.git
cd mozzarellm
pip install -e .
```

Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
```

## Quick Start

The easiest way to get started is to run the example notebook:

```bash
# Navigate to the examples directory
cd examples

# Launch Jupyter notebook
jupyter notebook example_notebook.ipynb
```

### Basic Python Usage

```python
import os
import pandas as pd
from mozzarellm import analyze_gene_clusters, reshape_to_clusters
from mozzarellm.prompts import ROBUST_SCREEN_CONTEXT, ROBUST_CLUSTER_PROMPT
from mozzarellm.configs import DEFAULT_OPENAI_REASONING_CONFIG

# Load sample data
sample_data = pd.read_csv("sample_data.csv")

# Process data into cluster format
cluster_df, gene_features = reshape_to_clusters(
    input_df=sample_data, 
    uniprot_col="uniprot_function", 
    verbose=True
)

# Run analysis
results = analyze_gene_clusters(
    # Input data
    input_df=cluster_df,
    # Model configuration
    model_name="o4-mini",  # or other supported models
    config_dict=DEFAULT_OPENAI_REASONING_CONFIG,
    # Analysis context and prompts
    screen_context=ROBUST_SCREEN_CONTEXT,
    cluster_analysis_prompt=ROBUST_CLUSTER_PROMPT,
    # Gene annotations
    gene_annotations_df=gene_features,
    # Options
    batch_size=1,
    save_outputs=True,  # Set to False to avoid writing files
    outputs_to_generate=["json", "clusters", "flagged_genes"]
)

# Explore results
print(results.keys())
cluster_analysis = results["cluster_df"]
gene_analysis = results["gene_df"]
```

### Command Line (deprecated)

Command line usage is currently deprecated as we migrate to a more flexible API-first approach.

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

## Configuration

Mozzarellm comes with default configurations for supported LLM providers:

```python
from mozzarellm.configs import (
    DEFAULT_OPENAI_REASONING_CONFIG,  # For OpenAI models with reasoning
    DEFAULT_OPENAI_CONFIG,            # For standard OpenAI models
    DEFAULT_ANTHROPIC_CONFIG,         # For Anthropic Claude models
    DEFAULT_GEMINI_CONFIG             # For Google Gemini models
)
```

Available models include:
- OpenAI: `o4-mini`, `o3-mini`, `gpt-4.1`, `gpt-4o`
- Anthropic: `claude-3-7-sonnet-20250219`, `claude-3-5-haiku-20241022`
- Google: `gemini-2.5-pro-preview-03-25`, `gemini-2.5-flash-preview-04-17`

## Python API

### Analyze Gene Clusters

```python
from mozzarellm import analyze_gene_clusters
from mozzarellm.configs import DEFAULT_OPENAI_REASONING_CONFIG

results = analyze_gene_clusters(
    # Input options
    input_df=cluster_df,               # DataFrame with cluster data
    # Model configuration
    model_name="o4-mini",              # Model to use
    config_dict=DEFAULT_OPENAI_REASONING_CONFIG,  # Configuration
    # Analysis context 
    screen_context=SCREEN_CONTEXT,     # Context about the experiment
    cluster_analysis_prompt=PROMPT,    # Custom analysis prompt
    # Gene annotations
    gene_annotations_df=gene_features, # Gene annotations DataFrame
    # Processing options
    batch_size=1,                      # Clusters per API call
    # Output options
    output_file="results/analysis",    # Output path prefix (optional)
    save_outputs=True,                 # Whether to write files
    outputs_to_generate=["json", "clusters", "flagged_genes"]  # Output types
)
```

### Process Gene-Level Data

```python
from mozzarellm import reshape_to_clusters

cluster_df, gene_features = reshape_to_clusters(
    input_df=gene_level_data,          # Gene-level input DataFrame
    gene_col="gene_symbol",            # Gene ID column
    cluster_col="cluster",             # Cluster assignment column
    uniprot_col="uniprot_function",    # Column with gene annotations
    verbose=True                       # Show processing information
)
```

## Output Structure

When `save_outputs=True`, the analysis produces three types of output files:
1. **JSON** (`*_clusters.json`): Complete analysis with raw LLM responses
2. **Cluster CSV** (`*_clusters.csv`): One row per cluster with pathway assignments and statistics
3. **Gene CSV** (`*_flagged_genes.csv`): One row per gene with detailed rationales and scores

When `save_outputs=False`, the same data is returned in a dictionary:
```python
{
    'clusters_dict': {...},    # Raw analysis results
    'json_data': {...},        # Processed JSON with metadata
    'cluster_df': pandas.DataFrame(...),  # Cluster-level analysis
    'gene_df': pandas.DataFrame(...)      # Gene-level analysis
}
```

## Examples

See the `examples/example_notebook.ipynb` file for a complete walkthrough of:
- Loading and processing gene data
- Running analysis with different models
- Exploring and visualizing results

## License

MIT License - See LICENSE file for details.