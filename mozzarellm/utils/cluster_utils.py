import pandas as pd


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
