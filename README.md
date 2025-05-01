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

### Basic Usage

```python
import mozzarellm
from dotenv import load_dotenv

# Load API keys
load_dotenv()

# Run analysis
results = mozzarellm.analyze_gene_clusters(
    input_file="data/clusters.csv",
    output_file="results/analysis",
    config_path="config/openai.json",
    gene_column="genes",
    gene_sep=";",
    cluster_id_column="cluster_id"
)
```

### Command Line

```bash
# Run analysis from command line
python -m mozzarellm.main \
  --config config_openai.json \
  --mode cluster \
  --input data/clusters.csv \
  --input_sep "," \
  --gene_column "genes" \
  --gene_sep ";" \
  --cluster_id_column "cluster_id" \
  --output_file results/analysis
```

## Input Format

Your input should be a CSV/TSV file with gene clusters:

| cluster_id | genes                                    | other_columns |
|------------|------------------------------------------|---------------|
| 1          | BRCA1;TP53;PTEN;MLH1                     | ...           |
| 2          | MYC;MAX;MYCN;E2F1;RB1                    | ...           |

Required columns:
- `cluster_id`: Unique identifier for each cluster
- `genes`: Semicolon-separated gene symbols

## Configuration

Create a JSON configuration file:

```json
{
  "MODEL": "gpt-4o",
  "CONTEXT": "You are a bioinformatician analyzing gene clusters.",
  "TEMP": 0.0,
  "MAX_TOKENS": 4000,
  "RATE_PER_TOKEN": 0.00001,
  "DOLLAR_LIMIT": 10.0
}
```

Available models:
- OpenAI: `gpt-4o`, `gpt-4.5`, `gpt-3.5-turbo`
- Anthropic: `claude-3-7-sonnet-20250219`, `claude-3.5-sonnet`
- Google: `gemini-2.0-pro`, `gemini-1.5-pro`

## Python API

### Analyze Gene Clusters

```python
from mozzarellm import analyze_gene_clusters

results = analyze_gene_clusters(
    input_file="data/clusters.csv",        # Input file path
    output_file="results/analysis",        # Output path prefix
    config_path="config.json",             # Configuration file
    model_name="gpt-4o",                   # Override model in config
    gene_column="genes",                   # Column with gene lists
    gene_sep=";",                          # Separator for genes
    cluster_id_column="cluster_id",        # Column with cluster IDs
    batch_size=1,                          # Clusters per API call
    gene_features_path="features.csv",     # Optional gene annotations
    screen_info_path="screen_info.txt"     # Optional screen context
)
```

### Reshape Gene-Level Data

```python
from mozzarellm import reshape_to_clusters

clusters_df = reshape_to_clusters(
    input_file="gene_level_data.csv",      # Gene-level input file
    output_file="clusters.csv",            # Output file path
    gene_col="gene_symbol",                # Gene ID column
    cluster_col="cluster",                 # Cluster assignment column
    gene_sep=";",                          # Separator for output genes
    additional_cols=["condition", "score"] # Additional columns to keep
)
```

## Output Files

The analysis produces three types of output files:

1. **JSON** (`analysis_clusters.json`): Complete analysis with raw LLM responses
2. **Cluster CSV** (`analysis_clusters.csv`): One row per cluster with pathway assignments and statistics
3. **Gene CSV** (`analysis_all_genes.csv`): One row per gene with detailed rationales and scores

## Integration

Mozzarellm can be integrated with other bioinformatics pipelines:

```python
# Example integration with a pipeline
from mozzarellm import analyze_gene_clusters

def my_pipeline_step(data):
    # Process clusters
    results = analyze_gene_clusters(
        input_df=data,
        output_file="results/analysis",
        config_path="config.json"
    )
    
    # Continue pipeline
    return process_results(results)
```

## Examples

See the `examples/` directory for:
- Jupyter notebooks
- Batch processing scripts
- Integration examples

## License

MIT License - See LICENSE file for details.