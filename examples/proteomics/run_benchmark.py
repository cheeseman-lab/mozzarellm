"""Run Proteomics (Schaffer) benchmark analysis using mozzarellm.

This script analyzes protein assemblies from the Schaffer et al. U2OS Cell Map dataset.
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
    "C5255": {"function": "RNase mitochondrial RNA processing", "genes": ["C18orf21"]},
    "C5415": {"function": "interferon response regulation", "genes": ["DPP9"]},
}

# Screen context for proteomics data
SCREEN_CONTEXT = """
This is from a spatial proteomics analysis (U2OS Cell Map).
Proteins in each assembly co-localize spatially and likely form functional complexes or
participate in related biological processes. Focus on identifying the shared biological
function and classifying genes based on their known vs. novel roles in that process.
"""


def main():
    """Run the benchmark analysis."""
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description="Run Proteomics benchmark analysis")
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

    print(f"Proteomics Benchmark - Model: {args.model}")

    # Load and reshape gene-wise data
    csv_path = os.path.join(script_dir, "schaffer_2025.csv")
    gene_df, cluster_df = load_benchmark_data(csv_path)

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
        results, VALIDATION_DATA, "Proteomics", args.model, quick_csv, check_confidence=False
    )
    create_detailed_analysis_csv(
        results, VALIDATION_DATA, "Proteomics", args.model, detailed_csv, check_confidence=False
    )

    print(f"✓ CSVs saved to {args.output_dir}/")
    print("  - quick_validation.csv")
    print("  - detailed_analysis.csv")


if __name__ == "__main__":
    main()
