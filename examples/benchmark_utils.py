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
                    print(
                        f"    ✗ {gene}: {category} (expected novel_role or uncharacterized)"
                    )
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


def create_quick_validation_csv(
    results,
    validation_data,
    dataset_name,
    model_name,
    output_path,
    check_confidence=False,
):
    """Create a quick validation CSV with one row per cluster.

    Args:
        results: ClusterAnalysisResults object
        validation_data: Dictionary mapping cluster IDs to expected results
        dataset_name: Name of the dataset (e.g., "OPS", "DepMap")
        model_name: Name of the model used
        output_path: Path to save the CSV file
        check_confidence: If True, include confidence validation (for OPS)

    Returns:
        DataFrame: Quick validation metrics
    """
    rows = []

    for cluster_id, expected in validation_data.items():
        if cluster_id not in results.clusters:
            row = {
                "dataset": dataset_name,
                "model": model_name,
                "cluster_id": cluster_id,
                "expected_function": expected["function"],
                "predicted_function": "NOT FOUND",
                "function_match": False,
                "gene_match_rate": "0/0",
            }
            if check_confidence:
                row["expected_confidence"] = expected.get("confidence", "N/A")
                row["predicted_confidence"] = "N/A"
                row["confidence_match"] = False
            rows.append(row)
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

        # Count gene matches
        total_genes = len(expected["genes"])
        matched_genes = 0
        for gene in expected["genes"]:
            category = categorize_gene(gene, cluster)
            if category in ["novel_role", "uncharacterized"]:
                matched_genes += 1

        row = {
            "dataset": dataset_name,
            "model": model_name,
            "cluster_id": cluster_id,
            "expected_function": expected_func,
            "predicted_function": predicted_func,
            "function_match": function_match,
            "gene_match_rate": f"{matched_genes}/{total_genes}"
            if total_genes > 0
            else "0/0",
        }

        if check_confidence:
            row["expected_confidence"] = expected.get("confidence", "N/A")
            row["predicted_confidence"] = cluster.pathway_confidence
            row["confidence_match"] = confidence_match

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def create_detailed_analysis_csv(
    results,
    validation_data,
    dataset_name,
    model_name,
    output_path,
    check_confidence=False,
):
    """Create a detailed analysis CSV with cluster AND gene-level information.

    This table shows BOTH cluster classification and gene classification with model outputs,
    enabling deep inspection of the analysis.

    Args:
        results: ClusterAnalysisResults object
        validation_data: Dictionary mapping cluster IDs to expected results
        dataset_name: Name of the dataset (e.g., "OPS", "DepMap")
        model_name: Name of the model used
        output_path: Path to save the CSV file
        check_confidence: If True, include confidence columns (for OPS)

    Returns:
        DataFrame: Detailed cluster and gene analysis
    """
    rows = []

    for cluster_id, expected in validation_data.items():
        if cluster_id not in results.clusters:
            # Cluster not found - create rows for expected genes if any
            for gene in expected.get("genes", []):
                row = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "cluster_id": cluster_id,
                    "expected_function": expected["function"],
                    "predicted_function": "NOT FOUND",
                    "function_match": False,
                    "gene": gene,
                    "expected_category": "novel_role or uncharacterized",
                    "actual_category": "not_found",
                    "gene_priority": "N/A",
                    "gene_rationale": "Cluster not found in results",
                    "cluster_summary": "N/A",
                }
                if check_confidence:
                    row["expected_confidence"] = expected.get("confidence", "N/A")
                    row["predicted_confidence"] = "N/A"
                    row["confidence_match"] = False
                rows.append(row)
            continue

        cluster = results.clusters[cluster_id]

        # Cluster-level information
        expected_func = expected["function"]
        predicted_func = cluster.dominant_process
        function_match = any(
            term in predicted_func.lower() for term in expected_func.lower().split()
        )

        confidence_match = True
        if check_confidence and "confidence" in expected:
            expected_conf = expected["confidence"]
            predicted_conf = cluster.pathway_confidence
            confidence_match = expected_conf == predicted_conf

        # Gene-level information
        validation_genes = expected.get("genes", [])
        if not validation_genes:
            # No genes to validate - create single row with cluster info
            row = {
                "dataset": dataset_name,
                "model": model_name,
                "cluster_id": cluster_id,
                "expected_function": expected_func,
                "predicted_function": predicted_func,
                "function_match": function_match,
                "gene": "N/A",
                "expected_category": "N/A",
                "actual_category": "N/A",
                "gene_priority": "N/A",
                "gene_rationale": "No validation genes for this cluster",
                "cluster_summary": cluster.summary,
            }
            if check_confidence:
                row["expected_confidence"] = expected.get("confidence", "N/A")
                row["predicted_confidence"] = cluster.pathway_confidence
                row["confidence_match"] = confidence_match
            rows.append(row)
        else:
            # Create one row per validation gene
            for gene in validation_genes:
                category = categorize_gene(gene, cluster)

                # Get priority and rationale if available
                priority = "N/A"
                rationale = "N/A"

                if category == "novel_role":
                    novel_genes = [
                        g for g in cluster.novel_role_genes if g.gene == gene
                    ]
                    if novel_genes:
                        priority = novel_genes[0].priority
                        rationale = novel_genes[0].rationale
                elif category == "uncharacterized":
                    unchar_genes = [
                        g for g in cluster.uncharacterized_genes if g.gene == gene
                    ]
                    if unchar_genes:
                        priority = unchar_genes[0].priority
                        rationale = unchar_genes[0].rationale
                elif category == "established":
                    rationale = "Classified as established gene"

                row = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "cluster_id": cluster_id,
                    "expected_function": expected_func,
                    "predicted_function": predicted_func,
                    "function_match": function_match,
                    "gene": gene,
                    "expected_category": "novel_role or uncharacterized",
                    "actual_category": category,
                    "gene_priority": priority,
                    "gene_rationale": rationale,
                    "cluster_summary": cluster.summary,
                }

                if check_confidence:
                    row["expected_confidence"] = expected.get("confidence", "N/A")
                    row["predicted_confidence"] = cluster.pathway_confidence
                    row["confidence_match"] = confidence_match

                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def load_benchmark_data(
    csv_path, gene_col="gene_symbol", cluster_col="cluster", additional_cols=None
):
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
    print(
        f"Loaded {len(gene_df)} genes across {gene_df[cluster_col].nunique()} clusters"
    )

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
        uniprot_path = os.path.join(
            script_dir, "..", "..", "data", "knowledge", "uniprot_data.tsv"
        )
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
