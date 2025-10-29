# Mozzarellm Examples

Benchmarks and examples demonstrating mozzarellm's ability to analyze gene clusters and identify biological pathways.

## Quick Start

```bash
# Run all benchmarks with default models (o1, claude-3-7-sonnet, gemini-2.5-pro, gpt-4o)
python run_all_benchmarks.py

# Run with specific models
python run_all_benchmarks.py --models o1 gpt-4o

# Run individual benchmark
cd ops && python run_benchmark.py
```

## Benchmarks

Three published datasets validate mozzarellm's performance:

| Dataset | Source | Genes | Clusters | Task |
|---------|--------|-------|----------|------|
| **OPS** | Funk et al. (2022) | 140 | 6 | Identify pathways from co-localized genes |
| **DepMap** | Wainberg et al. (2021) | 18 | 2 | Identify pathways from co-essential genes |
| **Proteomics** | Schaffer et al. (2025) | 18 | 2 | Identify pathways from protein assemblies |

Each benchmark validates:
- **Function prediction**: Does the model correctly identify the biological pathway?
- **Gene classification**: Are poorly-characterized genes properly categorized?

## Directory Structure

```
examples/
├── run_all_benchmarks.py     # Run all benchmarks across models
├── ops/                       # Optical pooled screen benchmark
│   ├── funk_2022.csv
│   └── run_benchmark.py
├── depmap/                    # DepMap co-essentiality benchmark
│   ├── wainberg_2021.csv
│   └── run_benchmark.py
├── proteomics/                # Spatial proteomics benchmark
│   ├── schaffer_2025.csv
│   └── run_benchmark.py
└── rag/                       # RAG examples (experimental)
```

## Usage

### Running all benchmarks

```bash
# Default: test all models, save to benchmark_results/
python run_all_benchmarks.py

# Specific models
python run_all_benchmarks.py --models o1 claude-3-7-sonnet-20250219

# Custom output location
python run_all_benchmarks.py --output my_results.json

# Don't save results
python run_all_benchmarks.py --no-save
```

Output example:
```
================================================================================
BENCHMARK SUMMARY
================================================================================
Model                                    Benchmark       Functions    Genes        Status
--------------------------------------------------------------------------------
o1                                       OPS             6/6          7/7          ✓
o1                                       DEPMAP          2/2          2/2          ✓
claude-3-7-sonnet-20250219               OPS             6/6          7/7          ✓
...
```

### Running individual benchmarks

```bash
cd ops/
python run_benchmark.py  # Uses MODEL="gpt-4o" by default
```

To test different models, edit `MODEL` in `run_benchmark.py`:
```python
MODEL = "o1"  # or "claude-3-7-sonnet-20250219", "gemini-2.5-pro-preview-03-25", etc.
```

## How It Works

Each benchmark:
1. Loads gene-wise CSV data (`{author}_{year}.csv`)
2. Reshapes to cluster format with `reshape_to_clusters()`
3. Loads UniProt annotations from `data/knowledge/uniprot_data.tsv`
4. Analyzes with `ClusterAnalyzer` (identifies pathways, classifies genes)
5. Validates predictions against ground truth

Validation data is stored inline in `VALIDATION_DATA` constants:
```python
VALIDATION_DATA = {
    "21": {"function": "ribosome biogenesis", "genes": ["C1orf131"]},
    "149": {"function": "mitochondrial homeostasis", "genes": ["KRAS", "BRAF"]},
}
```

## Data Format

**Input CSV** (gene-wise):
```csv
gene_symbol,cluster
AATF,21
C1orf131,21
KRAS,149
```

**After reshaping** (cluster-wise):
```csv
cluster_id,genes
21,AATF;C1orf131;DDX18;...
149,KRAS;BRAF;CYC1;...
```

## Configuration

All benchmarks use:
- **Temperature**: 0.0 (reproducibility)
- **UniProt annotations**: Loaded from `data/knowledge/uniprot_data.tsv`
- **Screen context**: Custom description of experimental approach
- **API keys**: Loaded from `.env` file via `python-dotenv`

Results are saved to:
- Individual benchmarks: `{benchmark}/results/{model}_results.json`
- Batch runs: `benchmark_results/results_{timestamp}.json`

## References

1. Funk et al. (2022). *Cell* 185(26):4857-4873. DOI: 10.1016/j.cell.2022.11.028
2. Wainberg et al. (2021). *Nature Genetics* 53:638-649. DOI: 10.1038/s41588-021-00840-z
3. Schaffer et al. (2025). *Nature*. DOI: 10.1038/s41586-025-08878-3
