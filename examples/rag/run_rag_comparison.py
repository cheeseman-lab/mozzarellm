"""Compare RAG and CoT approaches on the OPS (Funk) dataset.

This script compares three analysis modes:
1. Baseline: No RAG, no Chain-of-Thought
2. Enhanced RAG: RAG + Enhanced CoT (6-step structured reasoning)
3. Concise RAG: RAG + Concise CoT (faster, fewer tokens)

Usage:
    python run_rag_comparison.py
"""

import json
import os
import sys
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

# Add parent directory to path for benchmark_utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_utils import load_benchmark_data, load_uniprot_annotations, validate_results

from mozzarellm import (
    CONCISE_COT_INSTRUCTIONS,
    ENHANCED_COT_INSTRUCTIONS,
    ClusterAnalyzer,
)

# Load environment variables
load_dotenv()

# Configuration
MODEL = "claude-sonnet-4-5-20250929"
TEMPERATURE = 0.0
KNOWLEDGE_DIR = "../../data/knowledge"
RETRIEVER_K = 15  # Number of evidence snippets to retrieve

# OPS screen context
SCREEN_CONTEXT = """
This is from an optical pooled screen (OPS).
Genes in each cluster were identified based on similar morphological phenotypes.
Genes grouped within a cluster tend to exhibit similar phenotypes, suggesting they
may participate in the same biological process or pathway.
"""

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


def main():
    """Run RAG comparison analysis on OPS data."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("RAG COMPARISON: OPS (Funk et al.)")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Temperature: {TEMPERATURE}")
    print()

    # Load and reshape OPS data (local copy)
    csv_path = os.path.join(script_dir, "funk_2022.csv")
    gene_df, cluster_df = load_benchmark_data(
        csv_path,
        additional_cols=["up_features", "down_features", "phenotypic_strength"],
    )

    # Load UniProt annotations
    gene_annotations = load_uniprot_annotations(script_dir)

    # Knowledge directory
    knowledge_dir = os.path.join(script_dir, KNOWLEDGE_DIR)
    print(f"\nKnowledge directory: {knowledge_dir}")

    # Define analysis modes to compare
    modes = [
        {
            "name": "Baseline",
            "description": "No RAG, No CoT",
            "use_retrieval": False,
            "cot_instructions": None,
            "retriever_k": None,
        },
        {
            "name": "Enhanced RAG + CoT",
            "description": "RAG + 6-step reasoning",
            "use_retrieval": True,
            "cot_instructions": ENHANCED_COT_INSTRUCTIONS,
            "retriever_k": RETRIEVER_K,
        },
        {
            "name": "Concise RAG + CoT",
            "description": "RAG + faster CoT",
            "use_retrieval": True,
            "cot_instructions": CONCISE_COT_INSTRUCTIONS,
            "retriever_k": 10,
        },
    ]

    # Run all modes
    all_results = []
    for mode in modes:
        print("\n" + "=" * 70)
        print(f"MODE: {mode['name']} ({mode['description']})")
        print("=" * 70)

        # Initialize analyzer with mode-specific parameters
        analyzer_kwargs = {
            "model": MODEL,
            "temperature": TEMPERATURE,
            "show_progress": True,
            "use_retrieval": mode["use_retrieval"],
            "cot_instructions": mode["cot_instructions"],
        }
        if mode["use_retrieval"]:
            analyzer_kwargs["knowledge_dir"] = knowledge_dir
            analyzer_kwargs["retriever_k"] = mode["retriever_k"]

        analyzer = ClusterAnalyzer(**analyzer_kwargs)

        # Run analysis
        results = analyzer.analyze(
            cluster_df,
            gene_annotations=gene_annotations,
            screen_context=SCREEN_CONTEXT,
        )

        # Validate and store results (check_confidence=True for OPS data)
        validation = validate_results(
            results, VALIDATION_DATA, check_confidence=True, mode_name=mode["name"]
        )
        all_results.append(validation)

    # Final Comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    comparison_df = pd.DataFrame(all_results)
    print("\n", comparison_df.to_string(index=False))

    # Save results
    output_dir = os.path.join(script_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"rag_comparison_{timestamp}.json")

    output_data = {
        "benchmark": "OPS (Funk et al.)",
        "model": MODEL,
        "timestamp": timestamp,
        "comparison": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    print("\n✓ RAG comparison complete!")


if __name__ == "__main__":
    main()
