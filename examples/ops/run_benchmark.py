"""Run OPS (Funk) benchmark analysis using mozzarellm.

This script analyzes gene clusters from the Funk et al. OPS screen dataset.
Gene-wise data is reshaped to cluster format, analyzed, and validated inline.
"""

import os

import pandas as pd
from dotenv import load_dotenv

from mozzarellm import ClusterAnalyzer, reshape_to_clusters
from mozzarellm.utils.llm_analysis_utils import save_cluster_analysis

# Load environment variables from .env file
load_dotenv()

# Configuration
MODEL = "claude-sonnet-4-5-20250929"  # Modified by run_all_benchmarks.py
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

        # Check confidence if specified
        confidence_match = True
        if "confidence" in expected:
            expected_conf = expected["confidence"]
            predicted_conf = cluster.pathway_confidence
            confidence_match = expected_conf == predicted_conf

        # Overall match requires both function and confidence (if specified)
        overall_match = function_match and confidence_match

        if overall_match:
            total_function_matches += 1

        print(f"\nCluster {cluster_id}:")
        print(f"  Expected: {expected_func}")
        print(f"  Predicted: {predicted_func}")
        print(f"  {'✓' if function_match else '✗'} Function match")
        if "confidence" in expected:
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
        additional_cols=["up_features", "down_features", "phenotypic_strength"],
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
    analyzer = ClusterAnalyzer(model=MODEL, temperature=TEMPERATURE, show_progress=True)

    # Run analysis
    print("\nRunning analysis...")
    results = analyzer.analyze(
        cluster_df, gene_annotations=gene_annotations, screen_context=SCREEN_CONTEXT
    )

    # Save results (JSON + CSVs)
    output_base = os.path.join(OUTPUT_DIR, f"{MODEL.replace('/', '_')}_results")

    # Convert results to dictionary format for save_cluster_analysis
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

    # Save using built-in function (creates JSON + 2 CSVs)
    save_cluster_analysis(
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
