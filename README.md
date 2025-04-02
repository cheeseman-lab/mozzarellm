# mozzarellm

# Gene Set and Cluster Analysis with LLMs

This repository contains code to analyze gene sets and gene clusters using various LLM APIs (OpenAI, Google Gemini, Anthropic Claude, and custom models).

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

4. Modify `config.json` to set your preferred models and parameters

## Usage

### Gene Set Analysis (Original)

Analyze individual gene sets to identify their biological function:

```bash
python main.py --config config.json --input data/sample_gene_sets.csv --input_sep "," --gene_column "genes" --gene_sep ";" --start 0 --end 5 --initialize --output_file results/gene_analysis
```

### Cluster Analysis (New)

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

python main.py --config cluster_config.json --mode cluster --input data/luke_clusters.csv --input_sep "," --set_index "cluster_id" --gene_column "genes" --gene_sep ";" --batch_size 5 --output_file results/luke_cluster_analysis
```

### Data Preparation

Convert gene-level data to cluster-level format:
```bash
python reshape_clusters.py --input your_gene_table.csv --output data/clusters.csv --sep "," --gene_col "gene_symbol" --cluster_col "cluster" --gene_sep ";" --additional_cols "cluster_group"

python reshape_clusters.py --input data/df_phate_i.csv --output data/luke_clusters.csv --sep "," --gene_col "gene_symbol" --cluster_col "cluster" --gene_sep ";" --additional_cols "cluster_group"
```

Generate sample data:
```bash
bash create_sample_data.sh
```

## Arguments

### Common Arguments
- `--config`: Path to configuration JSON file
- `--input`: Path to input CSV with gene sets/clusters
- `--input_sep`: Separator for input CSV (comma, tab, etc.)
- `--gene_column`: Column name containing gene set
- `--gene_sep`: Separator for genes within a set
- `--start`: Start index for processing
- `--end`: End index for processing
- `--output_file`: Output file path (without extension)

### Gene Set Analysis
- `--initialize`: Initialize output columns if needed
- `--run_contaminated`: Process contaminated gene sets

### Cluster Analysis
- `--mode cluster`: Activate cluster analysis mode
- `--set_index`: Column name for cluster index
- `--batch_size`: Number of clusters to analyze in one batch
- `--gene_features`: Path to CSV with gene features

## Output

### Gene Set Analysis
- A TSV file with gene set names, scores, and analyses
- A JSON file with full LLM responses

### Cluster Analysis
- A JSON file with structured cluster analyses
- A summary CSV with dominant pathway, confidence, and gene counts
- An automatically generated log file

## Configuration

Two configuration files are provided:
- `config.json`: Original gene set analysis configuration
- `cluster_config.json`: New cluster analysis configuration

The cluster analysis configuration includes specialized context and higher token limits suitable for complex cluster analysis.
