# Mozzarellm Examples

This directory contains examples and benchmarks for mozzarellm.

## Directory Structure

- **ops/** - Optical Pooled Screen benchmark (Funk et al. 2022)
- **depmap/** - DepMap Co-essentiality benchmark (Wainberg et al. 2021)
- **proteomics/** - Spatial Proteomics benchmark (Schaffer et al. 2025)
- **rag/** - RAG (Retrieval-Augmented Generation) examples
- **old/** - Legacy benchmark notebooks and utilities
- **example_notebook.ipynb** - Main usage example
- **sample_data.csv** - Sample data for examples

## Benchmark Datasets

Three benchmark datasets are included, each representing a different experimental approach. Each dataset folder contains:
- `{author}_{year}.csv` - Gene-wise data
- `run_benchmark.py` - Analysis script with inline validation

### OPS (Optical Pooled Screen)

**Source:** Funk et al. (2022)
**Data:** 6 gene clusters from interphase localization screen
**Total genes:** 140
**File:** `ops/funk_2022.csv`

**Expected biological functions:**
- Cluster 21: ribosome biogenesis
- Cluster 37: mTOR signaling/ER-Golgi transport
- Cluster 121: Myc regulation/transcription
- Cluster 149: mitochondrial homeostasis
- Cluster 167: proteasome function
- Cluster 197: m6A mRNA modification

**Validation genes:** C1orf131, C7orf26, SETD2, KRAS, BRAF, AKIRIN2, HNRNPD

### DepMap (Co-essentiality)

**Source:** Wainberg et al. (2021)
**Data:** 2 co-essential gene modules
**Total genes:** 18
**File:** `depmap/wainberg_2021.csv`

**Expected biological functions:**
- Module 2067: clathrin-mediated endocytosis
- Module 2213: ether lipid synthesis

**Validation genes:** C15orf57, TMEM189

### Proteomics (Spatial)

**Source:** Schaffer et al. (2025)
**Data:** 2 protein assemblies
**Total genes:** 18
**File:** `proteomics/schaffer_2025.csv`

**Expected biological functions:**
- Assembly C5255: RNase mitochondrial RNA processing
- Assembly C5415: interferon response regulation

**Validation genes:** C18orf21, DPP9

## Usage

### Running a benchmark

```bash
# Change to dataset directory
cd ops/

# Run analysis (default: gpt-4o)
python run_benchmark.py
```

Each benchmark script:
1. Loads gene-wise data from CSV
2. Reshapes to cluster format using `reshape_to_clusters()`
3. Runs analysis with `ClusterAnalyzer`
4. Saves results to `results/gpt-4o_results.json`
5. Validates against ground truth inline

To test a different model, edit the `MODEL` variable in `run_benchmark.py`:

```python
MODEL = "claude-3-7-sonnet-20250219"  # or "gemini-2.5-pro-preview-03-25", etc.
```

### Example output

```
Loading gene-wise data from: funk_2022_genes.csv
Loaded 140 genes across 6 clusters

Reshaping gene-wise data to cluster format...
Created 6 cluster rows

Initializing ClusterAnalyzer with model: gpt-4o
Running analysis...
...

============================================================
VALIDATION AGAINST GROUND TRUTH
============================================================

Cluster 21:
  Expected: ribosome biogenesis
  Predicted: Ribosomal RNA processing and ribosome assembly
  ✓ Function match
  Validation genes:
    ✓ C1orf131: novel_role

Cluster 149:
  Expected: mitochondrial homeostasis
  Predicted: Mitochondrial function and MAPK signaling
  ✓ Function match
  Validation genes:
    ✓ KRAS: novel_role
    ✓ BRAF: novel_role

============================================================
VALIDATION SUMMARY
============================================================
Function matches: 6/6 (100.0%)
Genes classified: 7/7 (100.0%)

✓ Benchmark complete!
```

## Data Format

### Gene-wise CSV Format

Each benchmark uses a simple gene-wise CSV file:

```csv
gene_symbol,cluster
AATF,21
C1orf131,21
KRAS,149
BRAF,149
```

**Columns:**
- `gene_symbol`: Gene name
- `cluster`: Cluster/module/assembly ID

**Note:** Validation metadata (expected functions and validation genes) is stored in `VALIDATION_DATA` constants within each `run_benchmark.py` script, not in the CSV files

### Reshaping to Cluster Format

The scripts use the built-in `reshape_to_clusters()` function to convert gene-wise data to cluster format:

```python
from mozzarellm import reshape_to_clusters

cluster_df = reshape_to_clusters(
    input_df=gene_df,
    gene_col="gene_symbol",
    cluster_col="cluster",
    verbose=False
)
```

Output format (cluster-wise):
```
cluster_id,genes
21,AATF;C1orf131;DDX18;...
149,KRAS;BRAF;CYC1;...
```

### UniProt Annotations

Each benchmark script loads UniProt gene annotations from `data/knowledge/uniprot_data.tsv` and passes them to the `ClusterAnalyzer`:

```python
# Load UniProt annotations
uniprot_df = pd.read_csv("../../data/knowledge/uniprot_data.tsv", sep="\t")
gene_annotations = uniprot_df[["gene_names", "function"]].copy()

# Pass to analyzer
results = analyzer.analyze(cluster_df, gene_annotations=gene_annotations)
```

The UniProt annotations provide functional descriptions for genes, which the LLM uses to identify biological pathways and classify genes

## Validation

Validation is performed inline within each `run_benchmark.py` script. Ground truth validation points are stored as constants:

```python
VALIDATION_DATA = {
    "21": {"function": "ribosome biogenesis", "genes": ["C1orf131"]},
    "149": {"function": "mitochondrial homeostasis", "genes": ["KRAS", "BRAF"]},
    ...
}
```

**Validation checks:**
1. **Function match**: Does the predicted dominant process match expected function?
2. **Gene classification**: Are validation genes properly categorized (established/novel_role/uncharacterized)?

## Metrics

Each benchmark reports:

1. **Function match rate**: % of clusters where predicted function matches expected
2. **Gene classification rate**: % of validation genes that were properly categorized
3. **Quality metrics**: From ClusterAnalyzer (classification completeness, confidence validation)

## Notes

- All scripts use the `ClusterAnalyzer` API and `reshape_to_clusters` utility
- Temperature is set to 0.0 for reproducibility
- Each dataset has custom `screen_context` describing the experimental approach
- Results are saved as JSON for easy comparison across models
- The `.gitignore` excludes `results/` directories to avoid committing large outputs
- Gene-wise format allows easy tracking of validation genes and expected functions

## References

1. **Funk et al. (2022)** - "Optical pooled screens in human cells"
   *Cell* 185(26):4857-4873. DOI: 10.1016/j.cell.2022.11.028

2. **Wainberg et al. (2021)** - "A genome-wide atlas of co-essential modules assigns function to uncharacterized genes"
   *Nature Genetics* 53:638-649. DOI: 10.1038/s41588-021-00840-z

3. **Schaffer et al. (2025)** - "Multimodal cell maps as a foundation for structural and functional genomics"
   *Nature*. DOI: 10.1038/s41586-025-08878-3
