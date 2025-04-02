import pandas as pd
import argparse


def reshape_to_clusters(
    input_file,
    output_file,
    sep=",",
    gene_col="gene_symbol",
    cluster_col="cluster",
    gene_sep=";",
    additional_cols=None,
):
    """
    Reshape a gene-level table to a cluster-level table where each row is a cluster
    and contains all genes belonging to that cluster.

    Args:
        input_file: Path to input CSV/TSV file with gene-level data
        output_file: Path to save the output cluster-level file
        sep: Separator used in the input file (comma or tab)
        gene_col: Column name containing gene identifiers
        cluster_col: Column name containing cluster assignments
        gene_sep: Separator to use between genes in the output file
        additional_cols: List of additional columns to include as cluster-level metadata
    """
    print(f"Reading input file: {input_file}")

    # Read the input file
    df = pd.read_csv(input_file, sep=sep)

    print(f"Found {len(df)} genes across {df[cluster_col].nunique()} clusters")

    # Create a dictionary to store cluster information
    clusters = {}

    # Group genes by cluster
    for cluster, group in df.groupby(cluster_col):
        # Get list of genes in this cluster
        genes = group[gene_col].tolist()
        genes_text = gene_sep.join(genes)

        # Start with cluster ID and genes
        cluster_info = {
            "cluster_id": cluster,
            "genes": genes_text,
            "gene_count": len(genes),
        }

        # Add additional cluster-level metadata if specified
        if additional_cols:
            for col in additional_cols:
                if col in df.columns:
                    # For columns like cluster_group that should be the same for all genes in a cluster
                    # Take the first non-null value or the most common value
                    values = group[col].dropna().tolist()
                    if values:
                        if len(set(values)) == 1:  # All values are the same
                            cluster_info[col] = values[0]
                        else:
                            # Get the most common value
                            most_common = pd.Series(values).value_counts().index[0]
                            cluster_info[col] = most_common
                    else:
                        cluster_info[col] = None

        clusters[cluster] = cluster_info

    # Convert to DataFrame
    cluster_df = pd.DataFrame.from_dict(clusters, orient="index").reset_index(drop=True)

    # Sort by cluster_id
    if "cluster_id" in cluster_df.columns:
        try:
            cluster_df["cluster_id"] = pd.to_numeric(cluster_df["cluster_id"])
            cluster_df = cluster_df.sort_values("cluster_id")
        except Exception as e:
            # If conversion to numeric fails, sort as strings
            print(f"Could not convert cluster_id to numeric: {e}. Sorting as strings.")
            cluster_df = cluster_df.sort_values("cluster_id")

    # Save to output file
    print(f"Writing {len(cluster_df)} clusters to output file: {output_file}")
    cluster_df.to_csv(output_file, index=False)
    print("Done!")

    return cluster_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reshape gene-level data to cluster-level data."
    )
    parser.add_argument("--input", required=True, help="Input file path")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument(
        "--sep", default=",", help="Input file separator (default: comma)"
    )
    parser.add_argument(
        "--gene_col", default="gene_symbol", help="Column name for gene identifiers"
    )
    parser.add_argument(
        "--cluster_col", default="cluster", help="Column name for cluster assignments"
    )
    parser.add_argument(
        "--gene_sep", default=";", help="Separator for genes in output file"
    )
    parser.add_argument(
        "--additional_cols",
        default=None,
        help="Comma-separated list of additional columns to include",
    )

    args = parser.parse_args()

    # Convert additional_cols to list if provided
    add_cols = args.additional_cols.split(",") if args.additional_cols else None

    # Run the reshaping
    reshape_to_clusters(
        args.input,
        args.output,
        sep="," if args.sep == "comma" else "\t" if args.sep == "tab" else args.sep,
        gene_col=args.gene_col,
        cluster_col=args.cluster_col,
        gene_sep=args.gene_sep,
        additional_cols=add_cols,
    )
