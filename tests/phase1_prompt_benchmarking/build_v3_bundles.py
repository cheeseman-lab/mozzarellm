#!/usr/bin/env python
"""Build pure-source evidence bundles for a cluster slate.

Reproducible builder for the benchmark bundles. Reads a (screen_name,
cluster_id, gene_symbol) CSV and writes one flat `*__bundle.json` per cluster
into `benchmark_bundles_{source}/` (or an explicit --output-dir). Used for both
the 5 real clusters (benchmark_clusters_5real.csv) and the decoys
(benchmark_clusters_decoys.csv); pass --csv to pick the slate.
"""

import argparse
from pathlib import Path

from architecture_benchmarking_workflow.bench_orchestrator import (
    _client_from_config,  # noqa: F401 (ensure pkg importable)
)
from dotenv import load_dotenv

from mozzarellm.pipeline.bundle_builder import (
    build_evidence_bundles,
    get_or_append_stable_accession,
)
from mozzarellm.utils.cluster_utils import build_cluster_id_to_bundle_path
from mozzarellm.utils.io import load_table
from mozzarellm.utils.screen_context_utils import load_screen_context_json

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
INPUTS_DIR = SCRIPT_DIR / "benchmark_inputs_v3"
DEFAULT_CSV = INPUTS_DIR / "benchmark_clusters_5real.csv"
SCREEN_COL, CLUSTER_COL, GENE_COL = "screen_name", "cluster_id", "gene_symbol"
ORGANISM_ID = 9606


def process_screen(screen_name, screen_df, source, output_dir):
    ctx_path = INPUTS_DIR / f"{screen_name}_screen_context.json"
    load_screen_context_json(ctx_path)  # assert present/well-formed
    cluster_df = screen_df.drop(columns=[SCREEN_COL])
    acc_df = get_or_append_stable_accession(
        screen_name=screen_name,
        cluster_df=cluster_df,
        gene_column=GENE_COL,
        organism_id=ORGANISM_ID,
        warn_on_fallback=False,
        output_dir=output_dir,
    )
    build_evidence_bundles(
        screen_name=screen_name,
        acc_cluster_df=acc_df,
        gene_column=GENE_COL,
        cluster_id_column=CLUSTER_COL,
        stable_accession_col="accession",
        feature_columns=[],
        source=source,
        output_dir=output_dir,
        flat_output=True,
    )
    return build_cluster_id_to_bundle_path(output_dir, screen_name=screen_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["uniprot", "affinage", "both"], default="uniprot")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="cluster slate CSV")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="bundle output dir (default: benchmark_bundles_{source})",
    )
    args = ap.parse_args()
    output_dir = args.output_dir or SCRIPT_DIR / f"benchmark_bundles_{args.source}"
    df = load_table(args.csv)
    print(f"Source={args.source} csv={args.csv.name} -> {output_dir.name}  ({df.shape[0]} genes)")
    total = 0
    for screen_name in df[SCREEN_COL].unique():
        sdf = df[df[SCREEN_COL] == screen_name].copy()
        bmap = process_screen(screen_name, sdf, args.source, output_dir)
        print(f"  {screen_name}: {len(bmap)} bundles")
        total += len(bmap)
    print(f"Done. {total} bundles.")


if __name__ == "__main__":
    main()
