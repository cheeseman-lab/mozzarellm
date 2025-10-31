"""Run OPS (Funk) benchmark analysis using mozzarellm.

This script analyzes gene clusters from the Funk et al. OPS screen dataset.
Gene-wise data is reshaped to cluster format, analyzed, and validated inline.
"""

import os
import sys

from dotenv import load_dotenv

# Add parent directory to path for benchmark_utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_utils import (
    convert_results_to_dict,
    load_benchmark_data,
    load_uniprot_annotations,
    print_analysis_summary,
    save_benchmark_results,
    validate_results,
)

from mozzarellm import ClusterAnalyzer

# Load environment variables from .env file
load_dotenv()

# Configuration
MODEL = "claude-sonnet-4-5-20250929"  # Change to test different models
TEMPERATURE = 0.0
OUTPUT_DIR = "results"

# Validation constants (from findings.csv)
VALIDATION_DATA = {
    "21": {"function": "ribosome biogenesis", "genes": ["C1orf131"]},
    "37": {
        "function": "mTOR signaling/ER-Golgi transport/Integrator complex",
        "genes": ["C7orf26"],
    },
    "99": {
        "function": "No coherent biological pathway",
        "confidence": "Low",  # Should detect nontargeting controls
        "genes": [],  # Don't validate specific genes for noise cluster
    },
    "121": {"function": "Myc regulation/transcription", "genes": ["SETD2"]},
    "149": {"function": "mitochondrial homeostasis", "genes": ["KRAS", "BRAF"]},
    "167": {"function": "proteasome function", "genes": ["AKIRIN2"]},
    "197": {"function": "m6A mRNA modification", "genes": ["HNRNPD"]},
}

# Screen context for OPS data
SCREEN_CONTEXT = """
This is from an optical pooled screen (OPS).
Genes in each cluster were identified based on similar morphological phenotypes.
Genes grouped within a cluster tend to exhibit similar phenotypes, suggesting they
may participate in the same biological process or pathway.
"""


def main():
    """Run the benchmark analysis."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load and reshape gene-wise data
    csv_path = os.path.join(script_dir, "funk_2022.csv")
    gene_df, cluster_df = load_benchmark_data(
        csv_path,
        additional_cols=["up_features", "down_features", "phenotypic_strength"],
    )

    # Load UniProt annotations
    gene_annotations = load_uniprot_annotations(script_dir)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize analyzer
    print(f"\nInitializing ClusterAnalyzer with model: {MODEL}")
    analyzer = ClusterAnalyzer(model=MODEL, temperature=TEMPERATURE, show_progress=True)

    # Run analysis
    print("\nRunning analysis...")
    results = analyzer.analyze(
        cluster_df, gene_annotations=gene_annotations, screen_context=SCREEN_CONTEXT
    )

    # Convert and save results
    output_base = os.path.join(OUTPUT_DIR, f"{MODEL.replace('/', '_')}_results")
    clusters_dict = convert_results_to_dict(results)
    save_benchmark_results(clusters_dict, output_base, cluster_df)

    # Print analysis summary
    print_analysis_summary(results)

    # Validate against ground truth (check_confidence=True for OPS)
    validate_results(results, VALIDATION_DATA, check_confidence=True)

    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
