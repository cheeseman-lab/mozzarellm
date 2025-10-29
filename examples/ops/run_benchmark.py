"""Run OPS (Funk) benchmark analysis using mozzarellm.

This script analyzes gene clusters from the Funk et al. OPS screen dataset.
Gene-wise data is reshaped to cluster format, analyzed, and validated inline.
"""

import json
import os

import pandas as pd

from mozzarellm import ClusterAnalyzer, reshape_to_clusters

# Configuration
MODEL = "gpt-4o"  # Change to test different models
TEMPERATURE = 0.0
OUTPUT_DIR = "results"

# Validation constants (from findings.csv)
VALIDATION_DATA = {
    "21": {"function": "ribosome biogenesis", "genes": ["C1orf131"]},
    "37": {
        "function": "mTOR signaling/ER-Golgi transport/Integrator complex",
        "genes": ["C7orf26"],
    },
    "121": {"function": "Myc regulation/transcription", "genes": ["SETD2"]},
    "149": {"function": "mitochondrial homeostasis", "genes": ["KRAS", "BRAF"]},
    "167": {"function": "proteasome function", "genes": ["AKIRIN2"]},
    "197": {"function": "m6A mRNA modification", "genes": ["HNRNPD"]},
}

# Screen context for OPS data
SCREEN_CONTEXT = """
This is from an optical pooled screen (OPS) analyzing interphase cells.
Genes in each cluster co-localize and likely participate in related biological processes.
Focus on identifying the dominant biological process and classifying genes based on their
known vs. novel roles in that process.
"""


def categorize_gene(gene, cluster):
    """Determine which category a gene was classified into."""
    if gene in cluster.established_genes:
        return "established"

    novel_genes = [g.gene for g in cluster.novel_role_genes]
    if gene in novel_genes:
        return "novel_role"

    unchar_genes = [g.gene for g in cluster.uncharacterized_genes]
    if gene in unchar_genes:
        return "uncharacterized"

    return "not_classified"


def validate_results(results):
    """Validate analysis results against ground truth."""
    print("\n" + "=" * 60)
    print("VALIDATION AGAINST GROUND TRUTH")
    print("=" * 60)

    total_function_matches = 0
    total_genes_classified = 0
    total_validation_genes = sum(len(v["genes"]) for v in VALIDATION_DATA.values())

    for cluster_id, expected in VALIDATION_DATA.items():
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

        if function_match:
            total_function_matches += 1

        print(f"\nCluster {cluster_id}:")
        print(f"  Expected: {expected_func}")
        print(f"  Predicted: {predicted_func}")
        print(f"  {'✓' if function_match else '✗'} Function match")
        print("  Validation genes:")

        # Check validation genes
        for gene in expected["genes"]:
            category = categorize_gene(gene, cluster)
            if category != "not_classified":
                total_genes_classified += 1
                print(f"    ✓ {gene}: {category}")
            else:
                print(f"    ✗ {gene}: not classified")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(
        f"Function matches: {total_function_matches}/{len(VALIDATION_DATA)} "
        f"({100 * total_function_matches / len(VALIDATION_DATA):.1f}%)"
    )
    print(
        f"Genes classified: {total_genes_classified}/{total_validation_genes} "
        f"({100 * total_genes_classified / total_validation_genes:.1f}%)"
    )


def main():
    """Run the benchmark analysis."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load gene-wise data
    gene_data_path = os.path.join(script_dir, "funk_2022.csv")
    print(f"Loading gene-wise data from: {gene_data_path}")
    gene_df = pd.read_csv(gene_data_path)
    print(f"Loaded {len(gene_df)} genes across {gene_df['cluster'].nunique()} clusters")

    # Reshape to cluster format using built-in function
    print("\nReshaping gene-wise data to cluster format...")
    cluster_df = reshape_to_clusters(
        input_df=gene_df,
        gene_col="gene_symbol",
        cluster_col="cluster",
        verbose=False,
        return_dataframes=True,
    )
    print(f"Created {len(cluster_df)} cluster rows")

    # Load UniProt annotations
    uniprot_path = os.path.join(script_dir, "..", "..", "data", "knowledge", "uniprot_data.tsv")
    print(f"\nLoading UniProt annotations from: {uniprot_path}")
    uniprot_df = pd.read_csv(uniprot_path, sep="\t")
    gene_annotations = uniprot_df[["gene_names", "function"]].copy()
    print(f"Loaded {len(gene_annotations)} gene annotations")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize analyzer
    print(f"\nInitializing ClusterAnalyzer with model: {MODEL}")
    analyzer = ClusterAnalyzer(
        model=MODEL, temperature=TEMPERATURE, screen_context=SCREEN_CONTEXT, show_progress=True
    )

    # Run analysis
    print("\nRunning analysis...")
    results = analyzer.analyze(cluster_df, gene_annotations=gene_annotations)

    # Save results
    output_file = os.path.join(OUTPUT_DIR, f"{MODEL.replace('/', '_')}_results.json")

    results_dict = {
        "clusters": {
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
        },
        "metadata": results.metadata,
    }

    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    # Print analysis summary
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

    # Validate against ground truth
    validate_results(results)

    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
