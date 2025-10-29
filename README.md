# mozzarellm

**LLM-powered analysis of functional genomics experiments for gene function discovery**

Analyze gene clusters from large-scale functional genomics screens (CRISPR, RNAi, proteomics) to identify unexpected gene functions and prioritize candidates for experimental validation.

## Overview

Mozzarellm is a Python package that uses Large Language Models to analyze functional genomics experiments. In these experiments, genes are grouped into clusters based on phenotypic similarity or co-essentiality patterns. Mozzarellm helps researchers:

1. **Identify the dominant biological pathway** in each cluster
2. **Classify genes** into three categories:
   - **Established**: Well-known members of the identified pathway
   - **Uncharacterized**: Genes with minimal functional annotation (novel discovery targets)
   - **Novel Role**: Genes with known functions elsewhere that may have unexpected roles in this pathway
3. **Prioritize genes** for experimental validation based on novelty and pathway fit

## How It Works

### The Functional Genomics Discovery Workflow

```
Functional Genomics Experiment
   ↓
Genes clustered by phenotype/essentiality
   ↓
Mozzarellm LLM Analysis
   ↓
Pathway identification + Gene classification
   ↓
Prioritized candidates for validation
```

**Input**: Gene clusters from your functional genomics screen (CRISPR essentiality, RNAi phenotypes, protein interactions, etc.)

**Analysis**: LLMs analyze each cluster using their knowledge of biological pathways and published literature to identify:
- What pathway/process dominates the cluster
- Which genes are established members (validates the pathway)
- Which genes are completely uncharacterized (highest novelty)
- Which genes might have novel roles in this pathway

**Output**: Ranked lists of genes for experimental follow-up, with detailed rationales and pathway confidence assessments

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
import pandas as pd
from mozzarellm import ClusterAnalyzer, reshape_to_clusters

# Load sample data
sample_data = pd.read_csv("sample_data.csv")

# Process data into cluster format
cluster_df, gene_features = reshape_to_clusters(
    input_df=sample_data,
    uniprot_col="uniprot_function",
    verbose=True
)

# Create analyzer and run analysis
analyzer = ClusterAnalyzer(
    model="gpt-4o",  # or "claude-3-7-sonnet-20250219", "gemini-2.5-pro-preview-03-25"
    temperature=0.0,
    use_retrieval=False,  # Set True to enable RAG with retrieved evidence
)

# Run analysis
results = analyzer.analyze(
    cluster_df=cluster_df,
    gene_annotations=gene_features,
)

# Explore results
print(f"Analyzed {len(results.clusters)} clusters")
for cluster_id, cluster in results.clusters.items():
    print(f"Cluster {cluster_id}: {cluster.dominant_process}")
    print(f"  Confidence: {cluster.pathway_confidence}")
    print(f"  Flagged genes: {len(cluster.get_all_flagged_genes())}")
```

## Analysis Approaches

Mozzarellm offers two analysis modes optimized for different use cases:

### 1. Default Approach (Fast & Simple)

**Best for**: Quick analysis, well-characterized genes, smaller clusters

The LLM directly analyzes gene clusters using its training knowledge:

```python
analyzer = ClusterAnalyzer(
    model="gpt-4o",
    temperature=0.0,
    use_retrieval=False,  # Default approach
)
results = analyzer.analyze(cluster_df)
```

**Advantages**:
- Fast (one API call per cluster)
- Simple setup (no knowledge base required)
- Good for genes in well-studied pathways

**Limitations**:
- May struggle with very novel or recent discoveries
- No external evidence citations
- Limited reasoning transparency

### 2. Chain-of-Thought (CoT) Driven RAG (Evidence-Based)

**Best for**: Complex clusters, novel genes, publication-ready analysis

Combines retrieval-augmented generation with step-by-step reasoning:

```python
analyzer = ClusterAnalyzer(
    model="gpt-4o",
    temperature=0.0,
    use_retrieval=True,  # Enable CoT-driven RAG
    knowledge_dir="data/knowledge",  # Directory with .txt/.md files
    retriever_k=10,  # Number of evidence snippets
    cot_instructions=ENHANCED_COT_INSTRUCTIONS,  # Step-by-step guidance
)
results = analyzer.analyze(cluster_df, gene_annotations=annotations_df)
```

**How it works**:
1. **Retrieval**: Gathers relevant evidence from:
   - Gene annotations you provide
   - Screen context descriptions
   - Local knowledge files (papers, databases, notes)
2. **Chain-of-Thought**: LLM follows structured reasoning steps citing evidence
3. **Output**: Analysis with evidence provenance and reasoning trace

**Advantages**:
- Higher quality for complex cases
- Evidence citations for transparency
- Better handling of novel/recent discoveries
- Structured reasoning helps catch errors

**Limitations**:
- Requires knowledge base setup
- Slightly slower (retrieval overhead)

### When to Use Each

| Scenario | Recommended Approach |
|----------|---------------------|
| Exploratory analysis of new screen data | **Default** |
| Well-studied pathways (DNA repair, cell cycle) | **Default** |
| Publication-quality analysis | **CoT-driven RAG** |
| Novel/poorly characterized genes | **CoT-driven RAG** |
| Need evidence citations | **CoT-driven RAG** |
| Custom knowledge base (papers, databases) | **CoT-driven RAG** |

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

Mozzarellm provides a simple, unified interface for all LLM providers. Just specify the model name and any custom parameters:

```python
from mozzarellm import ClusterAnalyzer

# OpenAI models
analyzer = ClusterAnalyzer(model="gpt-4o", temperature=0.0)
analyzer = ClusterAnalyzer(model="o4-mini", temperature=1.0)  # Reasoning models

# Anthropic models
analyzer = ClusterAnalyzer(model="claude-3-7-sonnet-20250219")

# Google models
analyzer = ClusterAnalyzer(model="gemini-2.5-pro-preview-03-25")
```

Available models include:
- OpenAI: `o4-mini`, `o3-mini`, `gpt-4.1`, `gpt-4o`
- Anthropic: `claude-3-7-sonnet-20250219`, `claude-3-5-haiku-20241022`
- Google: `gemini-2.5-pro-preview-03-25`, `gemini-2.5-flash-preview-04-17`

## Python API

### Analyze Gene Clusters

```python
from mozzarellm import ClusterAnalyzer

# Create analyzer with desired configuration
analyzer = ClusterAnalyzer(
    model="gpt-4o",                    # Model to use
    temperature=0.0,                   # Temperature for generation
    max_tokens=8000,                   # Max tokens per request
    use_retrieval=True,                # Enable RAG (optional)
    knowledge_dir="data/knowledge",    # Knowledge files for RAG (optional)
    retriever_k=10,                    # Number of evidence snippets (optional)
)

# Run analysis
results = analyzer.analyze(
    cluster_df=cluster_df,             # DataFrame with cluster data
    gene_annotations=gene_features,    # Optional gene annotations DataFrame
    screen_context=SCREEN_CONTEXT,     # Optional experiment context (string)
    cluster_analysis_prompt=PROMPT,    # Optional custom analysis prompt
)

# Access results
for cluster_id, cluster in results.clusters.items():
    print(f"Cluster {cluster_id}:")
    print(f"  Process: {cluster.dominant_process}")
    print(f"  Confidence: {cluster.pathway_confidence}")

    # Get high-priority genes
    high_priority = cluster.get_high_priority_genes(threshold=8)
    for gene in high_priority:
        print(f"  - {gene.gene}: priority {gene.priority}")
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

The new API returns structured Pydantic models that provide type safety and convenient access to results:

```python
from mozzarellm import ClusterAnalyzer

analyzer = ClusterAnalyzer(model="gpt-4o")
results = analyzer.analyze(cluster_df)

# Access structured results
print(type(results))  # AnalysisResult

# Get a specific cluster
cluster = results.get_cluster("1")
print(cluster.dominant_process)       # String
print(cluster.pathway_confidence)     # "High" | "Medium" | "Low"
print(cluster.established_genes)      # List[str]
print(cluster.uncharacterized_genes)  # List[GeneClassification]
print(cluster.novel_role_genes)       # List[GeneClassification]

# Helper methods
all_flagged = cluster.get_all_flagged_genes()  # Both uncharacterized + novel_role
high_priority = cluster.get_high_priority_genes(threshold=8)

# Analysis-level methods
high_conf_clusters = results.get_all_high_confidence_clusters()
total_flagged = results.get_total_flagged_genes()

# Convert to DataFrames for saving (optional)
from mozzarellm.utils.llm_analysis_utils import save_cluster_analysis

# Convert to dict format
clusters_dict = {cid: vars(cluster) for cid, cluster in results.clusters.items()}
saved_results = save_cluster_analysis(
    clusters_dict,
    out_file_base="results/analysis",
    save_outputs=True  # Creates JSON, cluster CSV, and gene CSV files
)
```

## Examples

See the `examples/example_notebook.ipynb` file for a complete walkthrough of:
- Loading and processing gene data
- Running analysis with different models
- Exploring and visualizing results

## License

MIT License - See LICENSE file for details.