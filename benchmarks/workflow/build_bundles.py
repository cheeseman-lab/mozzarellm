#!/usr/bin/env python
"""
Batch data preprocessing for the prompt benchmark.

Reads inputs/benchmark_input.csv (screen_name, cluster_id, role, gene_symbol,
up_features, down_features, phenotypic_strength), matches each screen to its
<screen_name>_screen_context.json in inputs/, resolves stable accessions via
UniProt, and builds one master evidence bundle per cluster: the superset of
evidence (UniProt + Affinage annotations + feature columns). Per-source /
per-feature views are derived at prompt-assembly time (strip_source_fields /
strip_feature_fields in mozzarellm.utils.prompt_factory).

Usage:
    python workflow/build_bundles.py                      # full rebuild (UniProt/Affinage)
    python workflow/build_bundles.py --augment-phenotype  # add phenotype blocks in place, no API
"""

import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from mozzarellm.pipeline.bundle_builder import (
    build_evidence_bundles,
    get_or_append_stable_accession,
)
from mozzarellm.utils.cluster_utils import (
    STRENGTH_RANK_COL,
    attach_strength_ranks,
    build_cluster_id_to_bundle_path,
    compute_feature_coherence,
    compute_phenotype_strength,
)
from mozzarellm.utils.io import load_table, write_bundle
from mozzarellm.utils.screen_context_utils import load_screen_context_json

load_dotenv()  # walks upward to find .env automatically


############### configuration ###############
BENCH_DIR = Path(__file__).resolve().parents[1]
INPUTS_DIR = BENCH_DIR / "inputs"
OUTPUT_DIR = BENCH_DIR / "bundles"
BENCHMARK_CSV = INPUTS_DIR / "benchmark_input.csv"
SCREEN_COL = "screen_name"
CLUSTER_COL = "cluster_id"
GENE_COL = "gene_symbol"
ORGANISM_ID = 9606  # human

# non-bundle columns dropped before bundling; per-gene features pass through
NON_BUNDLE_COLS = [SCREEN_COL, "role"]


# per-screen processing
def process_screen(
    screen_name: str, screen_df: pd.DataFrame, output_dir: Path = OUTPUT_DIR
) -> dict:
    """Assert that screen context is present and well-formed, run accession lookup, and build evidence bundles."""
    # locate screen context JSON
    ctx_path = INPUTS_DIR / f"{screen_name}_screen_context.json"
    if not ctx_path.exists():
        raise FileNotFoundError(f"Missing screen context: {ctx_path}")
    screen_ctx = load_screen_context_json(ctx_path)
    print(f"  screen_context: {ctx_path.name}  ({len(screen_ctx)} keys)")

    # drop the non-bundle columns - downstream needs cluster_id + gene_symbol + features
    cluster_df = screen_df.drop(columns=[c for c in NON_BUNDLE_COLS if c in screen_df.columns])
    if "phenotypic_strength" in cluster_df.columns:
        cluster_df["phenotypic_strength"] = cluster_df["phenotypic_strength"].astype("string")

    # accession lookup (UniProt: primary annotation source + accession authority)
    acc_df = get_or_append_stable_accession(
        screen_name=screen_name,
        cluster_df=cluster_df,
        gene_column=GENE_COL,
        organism_id=ORGANISM_ID,
        warn_on_fallback=False,
        output_dir=output_dir,
    )
    print(f"  accessions resolved: {acc_df.shape}")

    # build master evidence bundles (flat: directly into output_dir)
    build_evidence_bundles(
        screen_name=screen_name,
        acc_cluster_df=acc_df,
        gene_column=GENE_COL,
        cluster_id_column=CLUSTER_COL,
        stable_accession_col="accession",
        feature_columns=[],  # per-gene features pass through as columns; no numeric coherence
        source="both",
        output_dir=output_dir,
        flat_output=True,
    )

    # collect bundle path mapping
    bundle_map = build_cluster_id_to_bundle_path(output_dir, screen_name=screen_name)
    print(f"  bundles: {len(bundle_map)} clusters → {output_dir.name}/")

    return {
        "screen_name": screen_name,
        "screen_ctx": screen_ctx,
        "acc_df": acc_df,
        "bundle_map": bundle_map,
        "bundles_dir": output_dir,
    }


def augment_phenotype(bundles_dir: Path = OUTPUT_DIR, benchmark_csv: Path = BENCHMARK_CSV) -> int:
    """Add the phenotype evidence blocks to existing bundles, without any API call.

    The master bundles were built with the per-gene up/down feature columns and
    the raw strength passing through. This attaches what the phenotype
    reasoning steps read: a feature_coherence table from the up/down lists, and
    per-gene phenotype_strength_rank ("N/M", ranked per screen over the
    benchmark's scored genes -- M is the benchmark's gene set for that screen,
    not the full screen) plus the cluster-level phenotype_strength table. The
    raw strength value is dropped so it never reaches the model. Annotation
    fields are untouched, so non-phenotype prompts stay byte-identical.
    """
    df = load_table(benchmark_csv)
    ranks: dict[tuple[str, str], str] = {}
    for screen_name, screen_df in df.groupby(SCREEN_COL):
        if screen_df["phenotypic_strength"].notna().any():
            ranked = attach_strength_ranks(screen_df, "phenotypic_strength")
            for _, row in ranked.iterrows():
                if isinstance(row[STRENGTH_RANK_COL], str):
                    ranks[(screen_name, row[GENE_COL])] = row[STRENGTH_RANK_COL]

    n = 0
    for path in sorted(bundles_dir.glob("*__bundle.json")):
        bundle = json.loads(path.read_text(encoding="utf-8"))
        genes = bundle["cluster_genes"]
        for gene in genes:
            gene.pop("phenotypic_strength", None)
            rank = ranks.get((bundle["screen_name"], gene[GENE_COL]))
            if rank:
                gene[STRENGTH_RANK_COL] = rank
        chunk = pd.DataFrame(genes)
        feature_cols = [c for c in ("up_features", "down_features") if c in chunk.columns]
        if feature_cols:
            bundle["feature_coherence"] = compute_feature_coherence(
                chunk, feature_cols, gene_column=GENE_COL
            )
        if STRENGTH_RANK_COL in chunk.columns:
            bundle["phenotype_strength"] = compute_phenotype_strength(chunk, gene_column=GENE_COL)
        write_bundle(bundle, path)
        n += 1
    return n


#    main
def main():
    print(f"Inputs:    {INPUTS_DIR}")
    print(f"Output:    {OUTPUT_DIR}")
    print()

    # load master benchmark table
    df = load_table(BENCHMARK_CSV)
    screens = df[SCREEN_COL].unique()
    print(f"Loaded {BENCHMARK_CSV.name}: {df.shape[0]} rows, {len(screens)} screens")
    print(f"Screens: {list(screens)}\n")

    results = {}
    for screen_name in screens:
        screen_df = df[df[SCREEN_COL] == screen_name].copy()
        n_clusters = screen_df[CLUSTER_COL].nunique()
        n_genes = len(screen_df)
        print(f"[{screen_name}] {n_genes} genes across {n_clusters} clusters")

        results[screen_name] = process_screen(screen_name, screen_df)
        print()

    # summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, res in results.items():
        print(f"  {name}: {len(res['bundle_map'])} bundles in {res['bundles_dir']}")
    print(f"\nDone. {sum(len(r['bundle_map']) for r in results.values())} total bundles.")

    return results


if __name__ == "__main__":
    import sys

    if "--augment-phenotype" in sys.argv:
        print(f"Augmented {augment_phenotype()} bundles with phenotype blocks.")
    else:
        main()
