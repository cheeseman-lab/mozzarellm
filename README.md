# mozzarellm

## Overview

This repository provides tools for analyzing gene clusters using Large Language Models (LLMs). It processes gene cluster data to identify potential biological pathways, categorize genes within those pathways, and prioritize novel gene candidates for further investigation. The primary focus is on helping researchers discover functional relationships between genes and identify promising candidates for experimental validation.

## Key Features

- **Automated Pathway Analysis**: Identifies the dominant biological process or pathway represented by a gene cluster
- **Gene Categorization**: Classifies genes within clusters as established, characterized, or novel
- **Novel Gene Prioritization**: Assigns importance scores to novel genes based on their likelihood of involvement in the identified pathway
- **Batch Processing**: Efficiently processes multiple gene clusters in a single run
- **Flexible Model Support**: Compatible with various LLM providers including OpenAI (GPT models), Anthropic (Claude models), Google (Gemini models), and Perplexity API

## Installation

1. Clone this repository
   ```bash
   git clone https://github.com/yourusername/mozzarellm.git && cd mozzarellm
   ```

2. Create and activate a conda environment
   ```bash
   conda env create -f environment.yml && conda activate mozzarellm
   ```

3. Copy `.env.example` to `.env` and add your API keys:
   ```bash
   cp .env.example .env && nano .env
   ```

4. Modify configuration files to set your preferred models and parameters

## How It Works

The system works through the following process:

1. **Data Input**: Takes a CSV/TSV file containing gene clusters, where each row represents a cluster with a list of gene symbols
2. **LLM Analysis**: Sends each cluster to an LLM with specialized prompts designed to:
   - Identify the dominant biological process or pathway
   - Assess confidence in the pathway identification
   - Categorize genes as established, characterized, or novel in relation to the pathway
   - Rank novel genes based on their likelihood of involvement
3. **Output Generation**: Produces structured outputs including:
   - JSON files with the full analysis
   - CSV files with gene-level and cluster-level analyses
   - Summary statistics and prioritization scores

## Input Data Format

The primary input should be a CSV/TSV file containing gene clusters in the following format:

| cluster_id | genes | additional_columns... |
|------------|-------|------------------------|
| 0 | STAT5A;STAT5B;MLX;HNF1B;PCGF2;GATAD2A;... | ... |
| 1 | ARF1;ATF4;B3GAT3;BATF2;BSCL2;... | ... |

Required columns:
- `cluster_id`: Unique identifier for each cluster
- `genes`: Semicolon-separated list of gene symbols in the cluster

Optional columns:
- `cluster_group`: Grouping or category information (will be preserved in output)
- Any additional columns will be preserved in the output

### Data Preparation

Convert gene-level data to cluster-level format:
```bash
python reshape_clusters.py --input your_gene_table.csv --output data/clusters.csv --sep "," --gene_col "gene_symbol" --cluster_col "cluster" --gene_sep ";" --additional_cols "cluster_group"
```

Example:
```bash
python reshape_clusters.py --input data/df_phate_i.csv --output data/luke_clusters.csv --sep "," --gene_col "gene_symbol" --cluster_col "cluster" --gene_sep ";" --additional_cols "cluster_group"
```

## Usage

### Cluster Analysis

Analyze gene clusters to identify dominant pathways and novel pathway members:

```bash
python main.py --config cluster_config.json --mode cluster --input data/sample_clusters.csv --input_sep "," --set_index "cluster_id" --gene_column "genes" --gene_sep ";" --start 0 --end 5 --output_file results/cluster_analysis
```

With gene features information:
```bash
python main.py --config cluster_config.json --mode cluster --input data/sample_clusters.csv --input_sep "," --set_index "cluster_id" --gene_column "genes" --gene_sep ";" --gene_features data/gene_features.csv --start 0 --end 5 --output_file results/cluster_analysis
```

With batch processing (multiple clusters in one API call):
```bash
python main.py --config cluster_config.json --mode cluster --input data/sample_clusters.csv --input_sep "," --set_index "cluster_id" --gene_column "genes" --gene_sep ";" --batch_size 3 --start 0 --end 5 --output_file results/cluster_analysis_batch
```

Example:
```bash
python main.py --config cluster_config.json --mode cluster --input data/luke_clusters.csv --input_sep "," --set_index "cluster_id" --gene_column "genes" --gene_sep ";" --batch_size 2 --output_file results/aconcagua_cell
```

## Configuration

Create/modify a configuration JSON file (e.g., `cluster_config.json`):

```json
{
  "MODEL": "gpt-4",
  "CONTEXT": "You are a highly skilled bioinformatician analyzing gene clusters.",
  "TEMP": 0.1,
  "MAX_TOKENS": 4000,
  "RATE_PER_TOKEN": 0.00001,
  "DOLLAR_LIMIT": 50.0,
  "LOG_NAME": "cluster_analysis"
}
```

### Using Different LLM Providers

To use different LLM providers, modify the `MODEL` field in your configuration file:

- OpenAI: `"MODEL": "gpt-4"` or `"MODEL": "gpt-3.5-turbo"`
- Anthropic: `"MODEL": "claude-3-opus-20240229"`
- Google: `"MODEL": "gemini-pro"`
- Perplexity: `"MODEL": "mistral-7b-instruct"`

## Output Files

The script generates three types of output files:

1. **JSON output** (`output_file_clusters.json`): Contains the full raw analysis from the LLM for each cluster

2. **Cluster-level CSV** (`output_file_clusters.csv`): Contains one row per cluster with the following columns:
   - `cluster_id`: Unique identifier for the cluster
   - `biological_process`: The identified dominant biological process or pathway
   - `pathway_confidence_level`: Confidence in the pathway identification (High, Medium, Low)
   - `cluster_importance_score`: Composite score reflecting the overall importance of the cluster
   - `established_gene_count`: Number of established pathway members
   - `characterized_gene_count`: Number of characterized genes
   - `novel_gene_count`: Number of novel gene candidates
   - `total_gene_count`: Total number of genes in the cluster
   - `functional_summary`: Brief summary of the pathway function
   - `highest_novel_gene_importance`: Highest importance score among novel genes
   - `average_novel_gene_importance`: Average importance score of novel genes
   - `high_importance_genes`: List of high importance genes (score ≥ 8)
   - `high_importance_gene_count`: Number of high importance genes
   - Plus any additional columns from the original input file

3. **Gene-level CSV** (`output_file_novel_genes.csv`): Contains one row per novel gene with the following columns:
   - `gene_name`: Gene symbol
   - `gene_description`: Rationale for why this gene may be involved in the pathway
   - `gene_importance_score`: Importance score (0-10)
   - `cluster_id`: ID of the cluster this gene belongs to
   - `biological_process`: The identified biological process
   - `pathway_confidence_level`: Confidence in the pathway identification
   - `cluster_importance_score`: Overall importance score of the cluster
   - `functional_summary`: Brief summary of the pathway function
   - Plus additional gene statistics and any columns from the original input file

## Arguments

### Cluster Analysis
- `--config`: Path to configuration JSON file
- `--mode cluster`: Activate cluster analysis mode
- `--input`: Path to input CSV with gene clusters
- `--input_sep`: Separator for input CSV (comma, tab, etc.)
- `--set_index`: Column name for cluster index
- `--gene_column`: Column name containing gene set
- `--gene_sep`: Separator for genes within a set
- `--batch_size`: Number of clusters to analyze in one batch
- `--start`: Start index for processing
- `--end`: End index for processing
- `--gene_features`: Path to CSV with gene features
- `--output_file`: Output file path (without extension)

## Gene Set Analysis (Legacy Mode)

In addition to the primary cluster analysis functionality, this repository also offers a legacy gene set analysis mode for analyzing individual gene sets without clustering information:

```bash
python main.py --config config.json --input data/sample_gene_sets.csv --input_sep "," --gene_column "genes" --gene_sep ";" --start 0 --end 5 --initialize --output_file results/gene_analysis
```

This mode provides a simpler analysis focusing on identifying the most likely function of each gene set without the detailed categorization and prioritization of the cluster analysis mode.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
