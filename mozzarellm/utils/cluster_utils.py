import pandas as pd
from pathlib import Path


def aggregate_genes_by_cluster(
    input_df: pd.DataFrame,
    gene_col: str = "gene_symbol",
    cluster_col: str = "cluster",
    additional_cols: list[str] | None = None,
    gene_sep: str = ";",
) -> pd.DataFrame:
    """Reshape a gene-level DataFrame into a cluster-level DataFrame.

    Each row in the output represents one cluster, with genes joined by `gene_sep`.
    Additional columns (e.g. feature columns) are also joined by `gene_sep`.

    Args:
        input_df: Gene-level DataFrame with one row per gene.
        gene_col: Column containing gene symbols.
        cluster_col: Column containing cluster IDs.
        additional_cols: Extra columns to aggregate alongside genes.
        gene_sep: Separator used when joining gene symbols (default: ";").

    Returns:
        Cluster-level DataFrame with columns: cluster_id, genes, [additional_cols...].
    """
    agg: dict[str, object] = {gene_col: lambda x: gene_sep.join(x.astype(str))}
    if additional_cols:
        for col in additional_cols:
            if col in input_df.columns:
                agg[col] = lambda x: gene_sep.join(x.astype(str))

    cluster_df = (
        input_df.groupby(cluster_col, sort=False)
        .agg(agg)
        .reset_index()
        .rename(columns={cluster_col: "cluster_id", gene_col: "genes"})
    )
    return cluster_df


def cluster_chunker(df: pd.DataFrame, cluster_id_column: str) -> list[pd.DataFrame]:
    """Chunk a gene-level table into smaller per-cluster DataFrames slices.

    Returns:
        List of DataFrames, one for each cluster.

    Raises:
        ValueError: If cluster_id_column is not found in DataFrame.

    Note:
        Will handle both sorted and interleaved cluster IDs. Relative order of genes within each cluster is preserved from the original DataFrame.
    """
    if cluster_id_column not in df.columns:
        raise ValueError(f"Cluster ID column '{cluster_id_column}' not found in DataFrame.")
    # Single-pass split; sort=False preserves first-seen cluster order; row order within each chunk is preserved.
    return [group for _cluster_id, group in df.groupby(cluster_id_column, sort=False)]


def find_feature_overlaps(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, str]:
    """Find overlapping comma-separated features in specified columns.

    Returns a mapping from column name to a comma-separated string of features that
    appear in 2+ rows.
    """
    overlaps: dict[str, str] = {}
    for col in feature_columns:
        if col not in df.columns:
            overlaps[col] = ""
            continue

        feature_counts: dict[str, int] = {}
        for feature_str in df[col].dropna():
            if not isinstance(feature_str, str) or not feature_str:
                continue
            features = [f.strip() for f in feature_str.split(",") if f.strip()]
            for feature in features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1

        overlapping = [f for f, count in feature_counts.items() if count >= 2]
        overlapping.sort(key=lambda f: (-feature_counts[f], f))
        overlaps[col] = ",".join(overlapping) if overlapping else ""

    return overlaps


def build_cluster_id_to_bundle_path(
    evidence_bundle_dir: Path,
    screen_name: str,
) -> dict[str, Path]:
    """Construct a cluster-to-prompt map using the cluster ID column and the evidence bundle directory.

    Greps the evidence bundle directory for all files with the name pattern {screen_name}__cluster_{#}__bundle.json and creates
    a dictionary mapping cluster ID to bundle path."""

    pattern = f"{screen_name}__cluster_*__bundle.json"

    bundle_files = list(evidence_bundle_dir.glob(pattern))

    cluster_id_to_bundle_path: dict[str, Path] = {}
    for f in bundle_files:
        name = f.name
        if "__cluster_" not in name or not name.endswith("__bundle.json"):
            continue
        cluster_id = name.split("__cluster_", 1)[1].split("__bundle.json", 1)[0]
        if cluster_id:
            cluster_id_to_bundle_path[str(cluster_id)] = f

    return cluster_id_to_bundle_path
