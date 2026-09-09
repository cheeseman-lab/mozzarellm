#!/usr/bin/env python
"""Merge the three per-reviewer annotation tables into one consensus-agnostic
ground-truth table.

Reads benchmark_inputs/ground_truth/annotation_{eric,liz,iain}.csv and writes
benchmark_inputs/benchmark_ground_truth.csv, keyed on (screen, cluster, gene)
with per-reviewer columns (classification_eric, subclass_liz, ...). No consensus
is baked in — the raw reviewer labels are the ground truth; consensus is a
downstream reduction computed by the evaluator.
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
GT_DIR = HERE / "benchmark_inputs" / "ground_truth"
OUT = HERE / "benchmark_inputs" / "benchmark_ground_truth.csv"

REVIEWERS = ("eric", "liz", "iain")
KEY = ["screen", "cluster", "gene"]
PER_REVIEWER = [
    "classification", "subclass", "pathway", "coherence",
    "cluster_notes", "preferred_source", "used_web",
]


def main() -> None:
    merged = None
    for r in REVIEWERS:
        df = pd.read_csv(GT_DIR / f"annotation_{r}.csv")
        keep = [c for c in PER_REVIEWER if c in df.columns]
        sub = df[KEY + keep].rename(columns={c: f"{c}_{r}" for c in keep})
        merged = sub if merged is None else merged.merge(sub, on=KEY, how="outer")
    merged = merged.sort_values(KEY).reset_index(drop=True)
    merged.to_csv(OUT, index=False)
    print(f"{len(merged)} gene rows x {len(REVIEWERS)} reviewers -> {OUT}")
    print(f"columns: {list(merged.columns)}")


if __name__ == "__main__":
    main()
