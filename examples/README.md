# Mozzarellm Benchmarks

Benchmarks validating mozzarellm's ability to analyze gene clusters and identify biological pathways.

## Quick Start

```bash
# Run all three benchmarks with default model (claude-sonnet-4-5-20250929)
python run_all_benchmarks.py

# Run with specific models
python run_all_benchmarks.py --models claude-sonnet-4-5-20250929 gpt-4o

# Run individual benchmark
cd ops
python run_benchmark.py --model claude-sonnet-4-5-20250929

# Run RAG comparison (baseline vs enhanced vs concise)
cd rag
python run_benchmark.py  # Runs all 3 modes
```

## Benchmarks

Three published datasets validate mozzarellm's performance:

| Dataset | Source | Genes | Clusters | Task |
|---------|--------|-------|----------|------|
| **OPS** | Funk et al. (2022) | 140 | 7 | Identify pathways from morphological phenotypes |
| **DepMap** | Wainberg et al. (2021) | 18 | 2 | Identify pathways from co-essential genes |
| **Proteomics** | Schaffer et al. (2025) | 18 | 2 | Identify pathways from protein assemblies |

Each benchmark validates:
- **Function prediction**: Does the model correctly identify the biological pathway?
- **Gene classification**: Are poorly-characterized genes properly categorized?

## Directory Structure

```
examples/
├── run_all_benchmarks.py         # Run OPS, DepMap, Proteomics across models
├── benchmark_utils.py             # Shared utilities for all benchmarks
├── ops/                           # Optical pooled screen benchmark
│   ├── funk_2022.csv
│   ├── run_benchmark.py
│   └── uniprot_data.tsv
├── depmap/                        # DepMap co-essentiality benchmark
│   ├── wainberg_2021.csv
│   ├── run_benchmark.py
│   └── uniprot_data.tsv
├── proteomics/                    # Spatial proteomics benchmark
│   ├── schaffer_2025.csv
│   ├── run_benchmark.py
│   └── uniprot_data.tsv
└── rag/                           # RAG methodology comparison
    ├── run_benchmark.py           # Compare baseline vs RAG approaches
    ├── data/knowledge/            # Knowledge files for RAG
    └── uniprot_data.tsv
```

## Usage

### Running All Benchmarks (OPS, DepMap, Proteomics)

Run all three benchmark datasets across one or more models:

```bash
# Default: Run with claude-sonnet-4-5-20250929, gemini-2.5-pro, gpt-4o
python run_all_benchmarks.py

# Specific models
python run_all_benchmarks.py --models claude-sonnet-4-5-20250929

# Multiple models
python run_all_benchmarks.py --models claude-sonnet-4-5-20250929 gpt-4o gemini-2.5-pro

# Custom output location
python run_all_benchmarks.py --output-base my_benchmark_run
```

**Output Structure:**
```
benchmark_results/run_TIMESTAMP/
├── OPS_claude-sonnet-4-5-20250929/
│   ├── quick_validation.csv
│   └── detailed_analysis.csv
├── DepMap_claude-sonnet-4-5-20250929/
├── Proteomics_claude-sonnet-4-5-20250929/
├── master_validation.csv              # Aggregated quick validation, sorted by dataset+cluster
├── master_detailed_analysis.csv       # Aggregated detailed analysis, sorted by dataset+cluster
└── results.json                       # Full metadata and status
```

The master CSVs allow easy comparison of how different models handled the same clusters.

### Running Individual Benchmarks

Each benchmark can be run independently:

```bash
# OPS benchmark
cd ops
python run_benchmark.py --model claude-sonnet-4-5-20250929 --temperature 0.0 --output-dir results

# DepMap benchmark
cd depmap
python run_benchmark.py --model gpt-4o

# Proteomics benchmark
cd proteomics
python run_benchmark.py
```

All benchmarks support the same CLI arguments:
- `--model`: Model to use (default: claude-sonnet-4-5-20250929)
- `--temperature`: Sampling temperature (default: 0.0)
- `--output-dir`: Output directory (default: results)

### RAG Comparison

The RAG benchmark compares three analysis approaches on the OPS dataset:

1. **Baseline**: No RAG, No CoT
2. **Enhanced RAG + CoT**: RAG with 6-step structured reasoning (k=15)
3. **Concise RAG + CoT**: RAG with faster CoT (k=10)

```bash
cd rag

# Run all three modes
python run_benchmark.py

# Run specific mode only
python run_benchmark.py --mode baseline
python run_benchmark.py --mode enhanced
python run_benchmark.py --mode concise

# Customize settings
python run_benchmark.py \
  --model claude-sonnet-4-5-20250929 \
  --knowledge-dir ../../data/knowledge \
  --retriever-k 15
```

**Output Structure:**
```
results/
├── baseline/
│   ├── quick_validation.csv
│   └── detailed_analysis.csv
├── enhanced/
│   ├── quick_validation.csv
│   └── detailed_analysis.csv
├── concise/
│   ├── quick_validation.csv
│   └── detailed_analysis.csv
├── combined_quick_validation.csv      # All 3 modes, sorted by cluster
└── combined_detailed_analysis.csv     # All 3 modes, sorted by cluster
```

The combined CSVs allow direct comparison of how each approach (baseline, enhanced, concise) handled the same clusters.

## How It Works

Each benchmark:
1. Loads gene-wise CSV data (`{author}_{year}.csv`)
2. Reshapes to cluster format with `load_benchmark_data()`
3. Loads UniProt annotations from `uniprot_data.tsv`
4. Analyzes with `ClusterAnalyzer` (identifies pathways, classifies genes)
5. Validates predictions against ground truth in `VALIDATION_DATA`
6. Generates CSV outputs:
   - `quick_validation.csv`: Cluster-level validation (function match, confidence)
   - `detailed_analysis.csv`: Gene-level validation (classification accuracy)

## Validation

Ground truth validation data is defined in each benchmark's `VALIDATION_DATA` constant:

```python
VALIDATION_DATA = {
    "21": {"function": "ribosome biogenesis", "genes": ["C1orf131"]},
    "149": {"function": "mitochondrial homeostasis", "genes": ["KRAS", "BRAF"]},
}
```

The benchmark utilities automatically:
- Check if predicted function matches ground truth
- Validate gene classifications (uncharacterized/novel role)
- Assess pathway confidence levels
- Generate validation CSVs with match/mismatch indicators

## Output Files

### quick_validation.csv
Cluster-level validation summary:
```csv
dataset,model,cluster_id,predicted_function,expected_function,function_match,predicted_confidence,expected_confidence,confidence_match
OPS,claude-sonnet-4-5-20250929,21,ribosome biogenesis,ribosome biogenesis,TRUE,High,,,
```

### detailed_analysis.csv
Gene-level classification details:
```csv
dataset,model,cluster_id,gene,predicted_category,expected_category,category_match,priority,rationale
OPS,claude-sonnet-4-5-20250929,21,C1orf131,Uncharacterized,Uncharacterized,TRUE,9,Minimal functional annotation...
```

## Configuration

All benchmarks use:
- **Default Model**: `claude-sonnet-4-5-20250929`
- **Temperature**: 0.0 (for reproducibility)
- **UniProt Annotations**: Loaded from local `uniprot_data.tsv` files
- **Screen Context**: Custom description of experimental approach
- **API Keys**: Loaded from `.env` file via `python-dotenv`

## References

1. Funk et al. (2022). *Cell* 185(26):4857-4873. DOI: 10.1016/j.cell.2022.11.028
2. Wainberg et al. (2021). *Nature Genetics* 53:638-649. DOI: 10.1038/s41588-021-00840-z
3. Schaffer et al. (2025). *Nature*. DOI: 10.1038/s41586-025-08878-3
