from pathlib import Path

import pandas as pd

STRENGTH_RANK_COL = "phenotype_strength_rank"


def attach_strength_ranks(
    df: pd.DataFrame, strength_column: str, *, higher_is_stronger: bool = True
) -> pd.DataFrame:
    """Replace a raw perturbation-strength column with scale-free "N/M" ranks.

    Any metric works (AUC, e-distance, normalized fate distance, ...): values
    are ranked across the table's scored genes, 1 = strongest, and each gene
    gets ``phenotype_strength_rank = "N/M"``. A column already holding "N/M"
    strings passes through unchanged. Genes with missing strength carry no
    rank. The raw column is dropped — raw strength values never enter bundles.
    """
    if strength_column not in df.columns:
        raise ValueError(f"strength_column {strength_column!r} not in cluster table")
    df = df.copy()
    col = df[strength_column]
    nonnull = col.notna()
    if not nonnull.any():
        raise ValueError(f"strength_column {strength_column!r} has no values")
    ranks = pd.Series(pd.NA, index=df.index, dtype=object)
    strs = col[nonnull].astype(str).str.strip()
    if strs.str.fullmatch(r"\d+/\d+").all():
        ranks[nonnull] = strs
    else:
        vals = pd.to_numeric(col, errors="coerce")
        scored = vals.notna()
        if not scored.any():
            raise ValueError(
                f"strength_column {strength_column!r} is neither numeric nor 'N/M' rank strings"
            )
        m = int(scored.sum())
        r = vals[scored].rank(ascending=not higher_is_stronger, method="min").astype(int)
        ranks[scored] = r.map(lambda n: f"{n}/{m}")
    df[STRENGTH_RANK_COL] = ranks
    return df.drop(columns=[strength_column])


def compute_phenotype_strength(df: pd.DataFrame, gene_column: str) -> dict:
    """Summarize a cluster's phenotype-strength ranks as a discrete table.

    Mirrors compute_feature_coherence: the aggregate the model reads is computed
    here, so the strength step is recall over a table rather than arithmetic.
    Ranks are the "N/M" strings in ``STRENGTH_RANK_COL``; genes without one are
    left out of the summary.
    """
    ranked = []
    screen_size = None
    for _, row in df.iterrows():
        value = row.get(STRENGTH_RANK_COL)
        if not isinstance(value, str) or "/" not in value:
            continue
        n, m = (int(x) for x in value.split("/"))
        screen_size = m
        ranked.append((n, str(row[gene_column])))
    ranked.sort()
    if not ranked:
        return {"n_ranked": 0, "screen_size": None, "ranked_genes": []}
    ranks = [n for n, _ in ranked]
    quartile = screen_size / 4
    median = ranks[len(ranks) // 2]
    return {
        "n_ranked": len(ranked),
        "screen_size": screen_size,
        "median_rank": f"{median}/{screen_size}",
        "strongest_quartile_frac": round(sum(1 for n in ranks if n <= quartile) / len(ranks), 3),
        "weakest_quartile_frac": round(
            sum(1 for n in ranks if n > 3 * quartile) / len(ranks), 3
        ),
        "ranked_genes": [{"gene": g, "rank": f"{n}/{screen_size}"} for n, g in ranked],
    }


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
