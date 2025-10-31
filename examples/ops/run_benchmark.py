"""Run OPS (Funk) benchmark analysis using mozzarellm.

This script analyzes gene clusters from the Funk et al. OPS screen dataset.
Gene-wise data is reshaped to cluster format, analyzed, and validated with CSV outputs.
"""

import argparse
import os
import sys

from dotenv import load_dotenv

# Add parent directory to path for benchmark_utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_utils import (
    convert_results_to_dict,
    create_detailed_analysis_csv,
    create_quick_validation_csv,
    load_benchmark_data,
    load_uniprot_annotations,
    save_benchmark_results,
)

from mozzarellm import ClusterAnalyzer

# Load environment variables from .env file
load_dotenv()

# Default configuration
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_OUTPUT_DIR = "results"

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
    parser = argparse.ArgumentParser(description="Run OPS benchmark analysis")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Temperature for model (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"OPS Benchmark - Model: {args.model}")

    # Load and reshape gene-wise data
    csv_path = os.path.join(script_dir, "funk_2022.csv")
    gene_df, cluster_df = load_benchmark_data(
        csv_path,
        additional_cols=["up_features", "down_features", "phenotypic_strength"],
    )

    # Load UniProt annotations
    gene_annotations = load_uniprot_annotations(script_dir)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize analyzer
    analyzer = ClusterAnalyzer(model=args.model, temperature=args.temperature, show_progress=True)

    # Run analysis
    print("Running analysis...")
    results = analyzer.analyze(
        cluster_df, gene_annotations=gene_annotations, screen_context=SCREEN_CONTEXT
    )

    # Convert and save standard results
    output_base = os.path.join(args.output_dir, f"{args.model.replace('/', '_')}_results")
    clusters_dict = convert_results_to_dict(results)
    save_benchmark_results(clusters_dict, output_base, cluster_df)

    # Generate validation CSVs
    quick_csv = os.path.join(args.output_dir, "quick_validation.csv")
    detailed_csv = os.path.join(args.output_dir, "detailed_analysis.csv")

    create_quick_validation_csv(
        results, VALIDATION_DATA, "OPS", args.model, quick_csv, check_confidence=True
    )
    create_detailed_analysis_csv(
        results, VALIDATION_DATA, "OPS", args.model, detailed_csv, check_confidence=True
    )

    print(f"✓ CSVs saved to {args.output_dir}/")
    print("  - quick_validation.csv")
    print("  - detailed_analysis.csv")


if __name__ == "__main__":
    main()
