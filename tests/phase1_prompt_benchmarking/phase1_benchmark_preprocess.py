#!/usr/bin/env python
"""
Batch data preprocessing for phase-1 prompt benchmarking.

Reads inputs/benchmark_clusters.csv (screen_name, cluster_id, gene_symbol),
matches each screen to its <screen_name>_screen_context.json in inputs/,
resolves stable accessions via UniProt, and builds evidence bundles.

Usage:
    python data_preprocess.py
"""

from pathlib import Path
import argparse
import json
import os
import sys

import pandas as pd
from dotenv import load_dotenv

from mozzarellm.utils.io import load_table
from mozzarellm.utils.screen_context_utils import load_screen_context_json
from mozzarellm.pipeline.bundle_builder import (
    build_evidence_bundles,
    get_or_append_stable_accession,
)
from mozzarellm.utils.cluster_utils import build_cluster_id_to_bundle_path

load_dotenv()  # walks upward to find .env automatically


############### configuration ###############
SCRIPT_DIR = Path(__file__).resolve().parent
INPUTS_DIR = SCRIPT_DIR / "benchmark_inputs"
OUTPUT_DIRS = {
    "uniprot": SCRIPT_DIR / "benchmark_evidence_bundles_uniprot",
    "affinage": SCRIPT_DIR / "benchmark_evidence_bundles_affinage",
    "both": SCRIPT_DIR / "benchmark_evidence_bundles_both",
}
BENCHMARK_CSV = INPUTS_DIR / "benchmark_clusters.csv"
SCREEN_COL = "screen_name"
CLUSTER_COL = "cluster_id"
GENE_COL = "gene_symbol"
ORGANISM_ID = 9606  # human

# feature columns if present (for later run maybe)
FEATURE_COLUMNS: list[str] = []


# per-screen processing
def process_screen(
    screen_name: str, screen_df: pd.DataFrame, source: str = "uniprot", output_dir: Path = None
) -> dict:
    """Assert that screen context is present and well-formed, run accession lookup, and build evidence bundles."""
    if output_dir is None:
        output_dir = OUTPUT_DIRS["uniprot"]
    # locate screen context JSON
    ctx_path = INPUTS_DIR / f"{screen_name}_screen_context.json"
    if not ctx_path.exists():
        raise FileNotFoundError(f"Missing screen context: {ctx_path}")
    screen_ctx = load_screen_context_json(ctx_path)
    print(f"  screen_context: {ctx_path.name}  ({len(screen_ctx)} keys)")

    # drop the screen_name column - downstream needs only cluster_id + gene_symbol
    cluster_df = screen_df.drop(columns=[SCREEN_COL])

    # accession lookup (UniProt: primary for uniprot/both, backup for affinage)
    acc_df = get_or_append_stable_accession(
        screen_name=screen_name,
        cluster_df=cluster_df,
        gene_column=GENE_COL,
        organism_id=ORGANISM_ID,
        warn_on_fallback=False,
        output_dir=output_dir,
    )
    print(f"  accessions resolved: {acc_df.shape}")

    # build evidence bundles (flat: directly into output_dir)
    build_evidence_bundles(
        screen_name=screen_name,
        acc_cluster_df=acc_df,
        gene_column=GENE_COL,
        cluster_id_column=CLUSTER_COL,
        stable_accession_col="accession",
        feature_columns=FEATURE_COLUMNS,
        source=source,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["uniprot", "affinage", "both"], default="uniprot")
    args = parser.parse_args()
    output_dir = OUTPUT_DIRS[args.source]

    print(f"Inputs:    {INPUTS_DIR}")
    print(f"Source:    {args.source}")
    print(f"Output:    {output_dir}")
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

        results[screen_name] = process_screen(
            screen_name, screen_df, source=args.source, output_dir=output_dir
        )
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
