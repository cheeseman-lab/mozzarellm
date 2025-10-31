"""Shared utility functions for benchmark scripts.

This module contains common validation, data loading, and result processing
functions used across OPS, DepMap, Proteomics, and RAG benchmark scripts.
"""

import os

import pandas as pd

from mozzarellm import reshape_to_clusters
from mozzarellm.utils.llm_analysis_utils import save_cluster_analysis


def categorize_gene(gene, cluster):
    """Determine which category a gene was classified into.

    Args:
        gene: Gene symbol to categorize
        cluster: Cluster object containing gene classifications

    Returns:
        str: One of "established", "novel_role", "uncharacterized", "not_classified"
    """
    if gene in cluster.established_genes:
        return "established"

    novel_genes = [g.gene for g in cluster.novel_role_genes]
    if gene in novel_genes:
        return "novel_role"

    unchar_genes = [g.gene for g in cluster.uncharacterized_genes]
    if gene in unchar_genes:
        return "uncharacterized"

    return "not_classified"


def validate_results(results, validation_data, check_confidence=False, mode_name=None):
    """Validate analysis results against ground truth.

    Args:
        results: ClusterAnalysisResults object
        validation_data: Dictionary mapping cluster IDs to expected results
        check_confidence: If True, validate confidence levels (for OPS benchmark)
        mode_name: Optional name of analysis mode (for RAG comparison)

    Returns:
        dict: Validation summary with function matches and genes classified
    """
    display_name = mode_name if mode_name else "Results"

    print("\n" + "=" * 70)
    print(f"VALIDATION: {display_name}")
    print("=" * 70)

    total_function_matches = 0
    total_genes_classified = 0
    total_validation_genes = sum(len(v["genes"]) for v in validation_data.values())

    for cluster_id, expected in validation_data.items():
        if cluster_id not in results.clusters:
            print(f"\n✗ Cluster {cluster_id}: Not found in results")
            continue

        cluster = results.clusters[cluster_id]

        # Check function match
        expected_func = expected["function"]
        predicted_func = cluster.dominant_process
        function_match = any(
            term in predicted_func.lower() for term in expected_func.lower().split()
        )

        # Check confidence if specified
        confidence_match = True
        if check_confidence and "confidence" in expected:
            expected_conf = expected["confidence"]
            predicted_conf = cluster.pathway_confidence
            confidence_match = expected_conf == predicted_conf

        # Overall match requires both function and confidence (if checking)
        overall_match = function_match and confidence_match

        if overall_match:
            total_function_matches += 1

        print(f"\nCluster {cluster_id}:")
        print(f"  Expected: {expected_func}")
        print(f"  Predicted: {predicted_func}")
        print(f"  {'✓' if function_match else '✗'} Function match")

        if check_confidence and "confidence" in expected:
            print(f"  Expected confidence: {expected['confidence']}")
            print(f"  Predicted confidence: {cluster.pathway_confidence}")
            print(f"  {'✓' if confidence_match else '✗'} Confidence match")

        # Check validation genes (if any specified)
        if expected["genes"]:
            print("  Validation genes:")
            for gene in expected["genes"]:
                category = categorize_gene(gene, cluster)
                # Validation genes should be classified as novel_role or uncharacterized
                # (they represent novel discoveries, not established genes)
                if category in ["novel_role", "uncharacterized"]:
                    total_genes_classified += 1
                    print(f"    ✓ {gene}: {category}")
                elif category == "established":
                    print(f"    ✗ {gene}: {category} (expected novel_role or uncharacterized)")
                else:
                    print(f"    ✗ {gene}: not classified")
        else:
            print("  (No specific gene validation for this cluster)")

    # Summary
    print("\n" + "=" * 70)
    print(f"VALIDATION SUMMARY: {display_name}")
    print("=" * 70)
    print(
        f"Function matches: {total_function_matches}/{len(validation_data)} "
        f"({100 * total_function_matches / len(validation_data):.1f}%)"
    )
    print(
        f"Genes classified: {total_genes_classified}/{total_validation_genes} "
        f"({100 * total_genes_classified / total_validation_genes:.1f}%)"
    )

    return {
        "mode": mode_name or "Results",
        "function_matches": f"{total_function_matches}/{len(validation_data)}",
        "genes_classified": f"{total_genes_classified}/{total_validation_genes}",
    }


def load_benchmark_data(csv_path, gene_col="gene_symbol", cluster_col="cluster", additional_cols=None):
    """Load and reshape gene-wise data to cluster format.

    Args:
        csv_path: Path to gene-wise CSV file
        gene_col: Column name for gene symbols
        cluster_col: Column name for cluster IDs
        additional_cols: Optional list of additional columns to include

    Returns:
        tuple: (gene_df, cluster_df) - Original and reshaped dataframes
    """
    print(f"Loading gene-wise data from: {csv_path}")
    gene_df = pd.read_csv(csv_path)
    print(f"Loaded {len(gene_df)} genes across {gene_df[cluster_col].nunique()} clusters")

    print("\nReshaping gene-wise data to cluster format...")
    cluster_df = reshape_to_clusters(
        input_df=gene_df,
        gene_col=gene_col,
        cluster_col=cluster_col,
        additional_cols=additional_cols,
        verbose=False,
        return_dataframes=True,
    )
    print(f"Created {len(cluster_df)} cluster rows")

    return gene_df, cluster_df


def load_uniprot_annotations(script_dir=None):
    """Load UniProt gene annotations.

    Args:
        script_dir: Directory of the calling script (for path resolution)
                   If None, uses standard path from current directory

    Returns:
        DataFrame: Gene annotations with columns ["gene_names", "function"]
    """
    if script_dir:
        uniprot_path = os.path.join(script_dir, "..", "..", "data", "knowledge", "uniprot_data.tsv")
    else:
        uniprot_path = "../../data/knowledge/uniprot_data.tsv"

    print(f"\nLoading UniProt annotations from: {uniprot_path}")
    uniprot_df = pd.read_csv(uniprot_path, sep="\t")
    gene_annotations = uniprot_df[["gene_names", "function"]].copy()
    print(f"Loaded {len(gene_annotations)} gene annotations")

    return gene_annotations


def convert_results_to_dict(results):
    """Convert ClusterAnalysisResults to dictionary format.

    Args:
        results: ClusterAnalysisResults object

    Returns:
        dict: Clusters dictionary suitable for save_cluster_analysis
    """
    clusters_dict = {
        cid: {
            "cluster_id": cluster.cluster_id,
            "dominant_process": cluster.dominant_process,
            "pathway_confidence": cluster.pathway_confidence,
            "established_genes": cluster.established_genes,
            "uncharacterized_genes": [
                {"gene": g.gene, "priority": g.priority, "rationale": g.rationale}
                for g in cluster.uncharacterized_genes
            ],
            "novel_role_genes": [
                {"gene": g.gene, "priority": g.priority, "rationale": g.rationale}
                for g in cluster.novel_role_genes
            ],
            "summary": cluster.summary,
            "quality_metrics": cluster.get_quality_summary(),
        }
        for cid, cluster in results.clusters.items()
    }
    return clusters_dict


def save_benchmark_results(clusters_dict, output_base, cluster_df):
    """Save benchmark results using standard format.

    Args:
        clusters_dict: Clusters dictionary from convert_results_to_dict
        output_base: Base path for output files (without extension)
        cluster_df: Original cluster DataFrame

    Returns:
        dict: Saved file paths from save_cluster_analysis
    """
    saved_results = save_cluster_analysis(
        clusters_dict,
        out_file_base=output_base,
        original_df=cluster_df,
        include_raw=False,
        save_outputs=True,
    )

    print("\n✓ Results saved to:")
    print(f"  - {output_base}_clusters.json (cluster data)")
    print(f"  - {output_base}_flagged_genes.csv (gene-level analysis)")
    print(f"  - {output_base}_cluster_summary.csv (cluster-level summary)")

    return saved_results


def print_analysis_summary(results):
    """Print standardized analysis summary.

    Args:
        results: ClusterAnalysisResults object
    """
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    for cluster_id, cluster in results.clusters.items():
        quality = cluster.get_quality_summary()
        print(f"\nCluster {cluster_id}: {cluster.dominant_process}")
        print(f"  Confidence: {cluster.pathway_confidence}")
        print(f"  Established genes: {len(cluster.established_genes)}")
        print(f"  Novel role genes: {len(cluster.novel_role_genes)}")
        print(f"  Uncharacterized genes: {len(cluster.uncharacterized_genes)}")
        print(
            f"  Quality: {'✓' if quality['classification_complete'] else '✗'} complete, "
            f"{'✓' if quality['confidence_validated'] else '✗'} validated"
        )
