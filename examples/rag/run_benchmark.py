"""Run RAG benchmark comparing different analysis modes.

This script compares three analysis modes on the OPS (Funk) dataset:
1. Baseline: No RAG, no Chain-of-Thought
2. Enhanced RAG + CoT: RAG + Enhanced CoT (6-step structured reasoning)
3. Concise RAG + CoT: RAG + Concise CoT (faster, fewer tokens)

Usage:
    python run_benchmark.py                              # Run all three modes
    python run_benchmark.py --mode baseline              # Run only baseline
    python run_benchmark.py --mode enhanced              # Run only enhanced
    python run_benchmark.py --mode concise               # Run only concise
"""

import argparse
import os
import sys

import pandas as pd
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

from mozzarellm import (
    CONCISE_COT_INSTRUCTIONS,
    ENHANCED_COT_INSTRUCTIONS,
    ClusterAnalyzer,
)

# Load environment variables
load_dotenv()

# Default configuration
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_KNOWLEDGE_DIR = "../../data/knowledge"
DEFAULT_RETRIEVER_K = 15

# Validation constants (from OPS findings.csv)
VALIDATION_DATA = {
    "21": {"function": "ribosome biogenesis", "genes": ["C1orf131"]},
    "37": {
        "function": "mTOR signaling/ER-Golgi transport/Integrator complex",
        "genes": ["C7orf26"],
    },
    "99": {
        "function": "No coherent biological pathway",
        "confidence": "Low",
        "genes": [],
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

# Define all analysis modes
ALL_MODES = {
    "baseline": {
        "name": "Baseline",
        "description": "No RAG, No CoT",
        "use_retrieval": False,
        "cot_instructions": None,
        "retriever_k": None,
    },
    "enhanced": {
        "name": "Enhanced RAG + CoT",
        "description": "RAG + 6-step reasoning",
        "use_retrieval": True,
        "cot_instructions": ENHANCED_COT_INSTRUCTIONS,
        "retriever_k": DEFAULT_RETRIEVER_K,
    },
    "concise": {
        "name": "Concise RAG + CoT",
        "description": "RAG + faster CoT",
        "use_retrieval": True,
        "cot_instructions": CONCISE_COT_INSTRUCTIONS,
        "retriever_k": 10,
    },
}


def run_single_mode(
    mode_config: dict,
    cluster_df,
    gene_annotations: dict,
    model: str,
    temperature: float,
    knowledge_dir: str,
    output_dir: str,
) -> dict:
    """Run analysis for a single mode.

    Args:
        mode_config: Mode configuration dictionary
        cluster_df: Cluster dataframe
        gene_annotations: Gene annotations dictionary
        model: Model to use
        temperature: Temperature setting
        knowledge_dir: Knowledge directory path
        output_dir: Output directory for this mode

    Returns:
        Dictionary with results and validation metrics
    """
    print("\n" + "=" * 70)
    print(f"MODE: {mode_config['name']} ({mode_config['description']})")
    print("=" * 70)

    # Initialize analyzer with mode-specific parameters
    analyzer_kwargs = {
        "model": model,
        "temperature": temperature,
        "show_progress": True,
        "use_retrieval": mode_config["use_retrieval"],
        "cot_instructions": mode_config["cot_instructions"],
    }
    if mode_config["use_retrieval"]:
        analyzer_kwargs["knowledge_dir"] = knowledge_dir
        analyzer_kwargs["retriever_k"] = mode_config["retriever_k"]

    analyzer = ClusterAnalyzer(**analyzer_kwargs)

    # Run analysis
    print("Running analysis...")
    results = analyzer.analyze(
        cluster_df,
        gene_annotations=gene_annotations,
        screen_context=SCREEN_CONTEXT,
    )

    # Create mode-specific output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert and save standard results
    output_base = os.path.join(output_dir, f"{model.replace('/', '_')}_results")
    clusters_dict = convert_results_to_dict(results)
    save_benchmark_results(clusters_dict, output_base, cluster_df)

    # Generate validation CSVs
    quick_csv = os.path.join(output_dir, "quick_validation.csv")
    detailed_csv = os.path.join(output_dir, "detailed_analysis.csv")

    create_quick_validation_csv(
        results,
        VALIDATION_DATA,
        f"RAG-{mode_config['name']}",
        model,
        quick_csv,
        check_confidence=True,
    )
    create_detailed_analysis_csv(
        results,
        VALIDATION_DATA,
        f"RAG-{mode_config['name']}",
        model,
        detailed_csv,
        check_confidence=True,
    )

    print(f"✓ Results saved to {output_dir}/")
    print("  - quick_validation.csv")
    print("  - detailed_analysis.csv")

    return {
        "mode": mode_config["name"],
        "output_dir": output_dir,
        "quick_csv": quick_csv,
        "detailed_csv": detailed_csv,
    }


def main():
    """Run the RAG benchmark analysis."""
    parser = argparse.ArgumentParser(description="Run RAG benchmark analysis")
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
        help=f"Base output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--knowledge-dir",
        type=str,
        default=DEFAULT_KNOWLEDGE_DIR,
        help=f"Knowledge directory for RAG (default: {DEFAULT_KNOWLEDGE_DIR})",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "baseline", "enhanced", "concise"],
        default="all",
        help="Which mode(s) to run (default: all)",
    )

    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("RAG COMPARISON BENCHMARK: OPS (Funk et al.)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print()

    # Load and reshape OPS data
    csv_path = os.path.join(script_dir, "../ops/funk_2022.csv")
    gene_df, cluster_df = load_benchmark_data(
        csv_path,
        additional_cols=["up_features", "down_features", "phenotypic_strength"],
    )

    # Load UniProt annotations
    gene_annotations = load_uniprot_annotations(script_dir)

    # Knowledge directory
    knowledge_dir = os.path.join(script_dir, args.knowledge_dir)
    print(f"Knowledge directory: {knowledge_dir}\n")

    # Determine which modes to run
    modes_to_run = ALL_MODES if args.mode == "all" else {args.mode: ALL_MODES[args.mode]}

    # Run selected modes
    all_mode_results = []
    for mode_key, mode_config in modes_to_run.items():
        mode_output_dir = os.path.join(args.output_dir, mode_key)
        result = run_single_mode(
            mode_config,
            cluster_df,
            gene_annotations,
            args.model,
            args.temperature,
            knowledge_dir,
            mode_output_dir,
        )
        all_mode_results.append(result)

    # Aggregate results if multiple modes were run
    if len(modes_to_run) > 1:
        print("\n" + "=" * 70)
        print("AGGREGATING RESULTS")
        print("=" * 70)

        # Aggregate quick validation CSVs
        quick_dfs = []
        for result in all_mode_results:
            quick_csv_path = result["quick_csv"]
            if os.path.exists(quick_csv_path):
                df = pd.read_csv(quick_csv_path)
                quick_dfs.append(df)

        if quick_dfs:
            combined_quick = pd.concat(quick_dfs, ignore_index=True)
            # Sort by cluster for easy comparison
            combined_quick = combined_quick.sort_values(by="cluster_id")
            combined_quick_path = os.path.join(args.output_dir, "combined_quick_validation.csv")
            combined_quick.to_csv(combined_quick_path, index=False)
            print(f"✓ Combined quick validation: {combined_quick_path}")

        # Aggregate detailed analysis CSVs
        detailed_dfs = []
        for result in all_mode_results:
            detailed_csv_path = result["detailed_csv"]
            if os.path.exists(detailed_csv_path):
                df = pd.read_csv(detailed_csv_path)
                detailed_dfs.append(df)

        if detailed_dfs:
            combined_detailed = pd.concat(detailed_dfs, ignore_index=True)
            # Sort by cluster and gene for side-by-side approach comparison
            if "gene" in combined_detailed.columns:
                combined_detailed = combined_detailed.sort_values(by=["cluster_id", "gene"])
            else:
                combined_detailed = combined_detailed.sort_values(by="cluster_id")
            combined_detailed_path = os.path.join(args.output_dir, "combined_detailed_analysis.csv")
            combined_detailed.to_csv(combined_detailed_path, index=False)
            print(f"✓ Combined detailed analysis: {combined_detailed_path}")

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for result in all_mode_results:
            print(f"✓ {result['mode']:<20} -> {result['output_dir']}/")

    print("\n✓ RAG benchmark complete!")


if __name__ == "__main__":
    main()
