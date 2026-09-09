#!/usr/bin/env python
"""
Batch data preprocessing for phase-1 prompt benchmarking.

Reads inputs/benchmark_input.csv (screen_name, cluster_id, role, gene_symbol,
up_features, down_features, phenotypic_strength), matches each screen to its
<screen_name>_screen_context.json in inputs/, resolves stable accessions via
UniProt, and builds one master evidence bundle per cluster: the superset of
evidence (UniProt + Affinage annotations + feature columns). Per-source /
per-feature views are derived at prompt-assembly time (strip_source_fields /
strip_feature_fields in mozzarellm.utils.prompt_factory).

Usage:
    python build_bundles.py
"""

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from mozzarellm.pipeline.bundle_builder import (
    build_evidence_bundles,
    get_or_append_stable_accession,
)
from mozzarellm.utils.cluster_utils import build_cluster_id_to_bundle_path
from mozzarellm.utils.io import load_table
from mozzarellm.utils.screen_context_utils import load_screen_context_json

load_dotenv()  # walks upward to find .env automatically


############### configuration ###############
SCRIPT_DIR = Path(__file__).resolve().parent
INPUTS_DIR = SCRIPT_DIR / "benchmark_inputs"
OUTPUT_DIR = SCRIPT_DIR / "benchmark_bundles"
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
    main()
