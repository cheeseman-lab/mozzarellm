"""Run DepMap (Wainberg) benchmark analysis using mozzarellm.

This script analyzes co-essential modules from the Wainberg et al. DepMap dataset.
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
    "2067": {"function": "clathrin-mediated endocytosis", "genes": ["C15orf57"]},
    "2213": {"function": "ether lipid synthesis", "genes": ["TMEM189"]},
}

# Screen context for DepMap data
SCREEN_CONTEXT = """
This is from a DepMap co-essentiality analysis.
Genes in each module are co-essential across cancer cell lines and likely function in the same
biological pathway or process. Focus on identifying the shared biological function and
classifying genes based on their known vs. novel roles in that process.
"""


def main():
    """Run the benchmark analysis."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load and reshape gene-wise data
    csv_path = os.path.join(script_dir, "wainberg_2021.csv")
    gene_df, cluster_df = load_benchmark_data(csv_path)

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

    # Validate against ground truth
    validate_results(results, VALIDATION_DATA)

    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
