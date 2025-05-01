# utils/cluster_utils.py
import pandas as pd
import os

def reshape_to_clusters(
    input_file=None,
    output_file=None,
    input_df=None,
    sep=",",
    gene_col="gene_symbol",
    cluster_col="cluster",
    gene_sep=";",
    additional_cols=None,
    uniprot_col=None,
    gene_features_output=None,
    verbose=True
):
    """
    Reshape a gene-level table to a cluster-level table where each row is a cluster
    and contains all genes belonging to that cluster.
    
    Args:
        input_file: Path to input CSV/TSV file with gene-level data
        output_file: Path to save the output cluster-level file (optional)
        input_df: DataFrame to use instead of loading from file (optional)
        sep: Separator used in the input file (comma or tab)
        gene_col: Column name containing gene identifiers
        cluster_col: Column name containing cluster assignments
        gene_sep: Separator to use between genes in the output file
        additional_cols: List of additional columns to include as cluster-level metadata
        uniprot_col: Column name containing UniProt data (optional)
        gene_features_output: Path to save gene features data (optional)
        verbose: Whether to print progress messages
        
    Returns:
        cluster_df: DataFrame with reshaped clusters
        
    Note:
        Either input_file or input_df must be provided
    """
    # Handle input - either from file or DataFrame
    if input_df is not None:
        df = input_df
        if verbose:
            print(f"Using provided DataFrame with {len(df)} rows")
    elif input_file is not None:
        if verbose:
            print(f"Reading input file: {input_file}")
        # Read the input file
        df = pd.read_csv(input_file, sep=sep)
    else:
        raise ValueError("Either input_file or input_df must be provided")
    
    if verbose:
        print(f"Found {len(df)} genes across {df[cluster_col].nunique()} clusters")
    
    # Handle gene features if UniProt column is specified
    if uniprot_col is not None and uniprot_col in df.columns:
        if verbose:
            print(f"Extracting gene features from {uniprot_col} column")
        
        # Extract gene features (gene to UniProt mapping)
        gene_features = df[[gene_col, uniprot_col]].drop_duplicates()
        
        # Fill NaN values with empty string
        gene_features.fillna('', inplace=True)
        
        # Save gene features if output path is provided
        if gene_features_output:
            if verbose:
                print(f"Saving gene features to {gene_features_output}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(gene_features_output), exist_ok=True)
            
            # Save gene features
            gene_features.to_csv(gene_features_output, index=False)
    
    # Create a dictionary to store cluster information
    clusters = {}
    
    # Group genes by cluster
    for cluster, group in df.groupby(cluster_col):
        # Get list of genes in this cluster
        genes = group[gene_col].tolist()
        genes_text = gene_sep.join(genes)
        
        # Start with just cluster ID and genes
        cluster_info = {
            "cluster_id": cluster,
            "genes": genes_text,
        }
        
        # Add additional cluster-level metadata if specified
        if additional_cols:
            # Add gene count if requested
            if "gene_count" in additional_cols:
                cluster_info["gene_count"] = len(genes)
            
            # Process other additional columns
            for col in additional_cols:
                if col != "gene_count" and col in df.columns:
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
            if verbose:
                print(f"Could not convert cluster_id to numeric: {e}. Sorting as strings.")
            cluster_df = cluster_df.sort_values("cluster_id")
    
    # Save to output file if provided
    if output_file:
        if verbose:
            print(f"Writing {len(cluster_df)} clusters to output file: {output_file}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save cluster data
        cluster_df.to_csv(output_file, index=False)
        if verbose:
            print("Done!")
    
    return cluster_df


def parse_additional_cols(additional_cols_str):
    """
    Parse a comma-separated string of additional columns.
    
    Args:
        additional_cols_str: Comma-separated string of column names
        
    Returns:
        List of column names or None if input is None
    """
    if additional_cols_str:
        return [col.strip() for col in additional_cols_str.split(",")]
    return None