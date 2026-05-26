from pathlib import Path

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


def compute_feature_coherence(
    df: pd.DataFrame,
    feature_columns: list[str],
    gene_column: str,
) -> dict:
    """Compute per-feature gene-coverage across a cluster with supporting gene lists.

    Each column in `feature_columns` is a comma-separated per-gene list of features
    differentially significant in a given direction. Direction is inferred from the
    column name — substring "up" → up, substring "down" → down.

    Returns a dict with `n_genes_in_cluster` and a `features` array, one row per
    feature called by at least one gene, with `n_up`/`frac_up`/`up_genes` and
    `n_down`/`frac_down`/`down_genes`. Sorted by aggregate signal (most-covered first).
    """
    n_genes = len(df)
    up_col = next((c for c in feature_columns if "up" in c.lower()), None)
    down_col = next((c for c in feature_columns if "down" in c.lower()), None)

    def _gene_to_features(col: str | None) -> dict[str, list[str]]:
        if col is None or col not in df.columns:
            return {}
        out: dict[str, list[str]] = {}
        for _, row in df.iterrows():
            gene = row[gene_column]
            features_str = row.get(col, "")
            if not isinstance(features_str, str) or not features_str:
                continue
            for f in (x.strip() for x in features_str.split(",")):
                if f:
                    out.setdefault(f, []).append(gene)
        return out

    up_map = _gene_to_features(up_col)
    down_map = _gene_to_features(down_col)

    rows = []
    for feat in sorted(set(up_map) | set(down_map)):
        ups = up_map.get(feat, [])
        downs = down_map.get(feat, [])
        rows.append(
            {
                "feature": feat,
                "n_up": len(ups),
                "frac_up": round(len(ups) / n_genes, 3) if n_genes else 0.0,
                "up_genes": ups,
                "n_down": len(downs),
                "frac_down": round(len(downs) / n_genes, 3) if n_genes else 0.0,
                "down_genes": downs,
            }
        )
    rows.sort(key=lambda r: (-(r["n_up"] + r["n_down"]), r["feature"]))

    return {"n_genes_in_cluster": n_genes, "features": rows}


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
