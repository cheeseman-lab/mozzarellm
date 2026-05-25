"""Cross-field evaluator: join prediction CSVs against the ground-truth CSV
and compute per-route / per-case-type match rates.

Consumes the gene-level prediction CSVs produced by `bench_trace_parser.py`
(one CSV per route in an experiment directory) and joins them against
`benchmark_clusters_ground_truth.csv` on (screen_name, cluster_id, gene_symbol).

Match flags are computed against the active human reviewers (default eric +
iain; operon is excluded by default because it is an LLM annotator, not a human
reviewer — scoring against it is LLM-evaluating-LLM). Because the ground-truth
CSV's `consensus_classification` column was precomputed over all three reviewers,
this evaluator RECOMPUTES consensus from the active reviewers rather than trusting
that column.

With two reviewers, classification is scored three ways so concordance is visible:
- classification_match_eric / _iain: vs each reviewer individually
- classification_match_either: matches at least one reviewer (ceiling 100%)
- classification_match_consensus: matches the agreed label, scored only on genes
  where the reviewers agree (None elsewhere; ceiling 100% on that subset)
- experts_agree: do the active reviewers unanimously agree (the per-gene ceiling)

Aggregates per route and per (route × case_type), emits:
- eval_per_gene.csv
- eval_per_route.csv
- eval_per_route_per_case_type.csv
- eval_report.md

Usage:
    python -m architecture_benchmarking_workflow.bench_evaluator \\
        --experiment-dir benchmarking_outputs/1.arch/<experiment_id> \\
        --ground-truth benchmark_clusters_ground_truth.csv \\
        --reviewers eric,iain
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

PHASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GROUND_TRUTH = PHASE_DIR / "benchmark_clusters_ground_truth.csv"

# Human reviewers used for scoring. operon is intentionally excluded — it is an
# LLM annotator, so including it makes the benchmark LLM-evaluating-LLM.
DEFAULT_REVIEWERS = ("eric", "iain")
JOIN_KEYS = ("screen_name", "cluster_id", "gene_symbol")

# Confidence ordering for tie-breaks (mode ties resolve toward higher confidence).
_CONF_ORDER = {"high": 3, "medium": 2, "low": 1}

# Columns in the prediction CSVs we always expect (from bench_trace_parser.py)
PRED_COLS_REQUIRED = (
    "screen_name",
    "cluster_id",
    "gene_symbol",
    "route",
    "replicate",
    "run_id",
    "predicted_class",
    "predicted_subclass",
    "pathway",
    "pathway_confidence",
)


def _norm_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _norm_lower(x: Any) -> str:
    return _norm_str(x).lower()


def _norm_class(x: Any) -> str:
    """Classification labels: trim + uppercase. Empty stays empty."""
    return _norm_str(x).upper()


def consensus_of(values: list[str]) -> str:
    """Consensus over present (non-empty) values: the agreed value if all present
    values are equal, else "" (no consensus). Requires at least one present value."""
    present = [v for v in values if v]
    if not present:
        return ""
    return present[0] if len(set(present)) == 1 else ""


def majority_confidence(row: pd.Series, reviewers: tuple[str, ...]) -> str:
    """Most-frequent expert_confidence_* across active reviewers. Ties (incl. the
    2-reviewer disagreement case) resolve toward higher confidence. Empty if no votes."""
    votes = [_norm_str(row.get(f"expert_confidence_{r}")) for r in reviewers]
    votes = [v for v in votes if v]
    if not votes:
        return ""
    counter = Counter(votes)
    top_count = max(counter.values())
    top = [v for v, c in counter.items() if c == top_count]
    if len(top) == 1:
        return top[0]
    return max(top, key=lambda v: _CONF_ORDER.get(v.lower(), 0))


def pathway_substring_match(predicted: str, experts: list[str]) -> bool:
    """Any expert pathway is a case-insensitive substring of the prediction."""
    pred = _norm_lower(predicted)
    if not pred:
        return False
    for exp in experts:
        exp_lower = _norm_lower(exp)
        if exp_lower and exp_lower in pred:
            return True
    return False


def pathway_loose_match(predicted: str, experts: list[str]) -> bool:
    """Bidirectional substring: prediction in expert OR expert in prediction."""
    pred = _norm_lower(predicted)
    if not pred:
        return False
    for exp in experts:
        exp_lower = _norm_lower(exp)
        if not exp_lower:
            continue
        if exp_lower in pred or pred in exp_lower:
            return True
    return False


def load_predictions(experiment_dir: Path) -> pd.DataFrame:
    """Stack every prediction CSV in the experiment dir into one frame."""
    csvs = sorted(experiment_dir.glob("*_3*.csv"))
    csvs = [c for c in csvs if c.name not in {"aggregate_summary.csv"}]
    if not csvs:
        raise FileNotFoundError(
            f"No per-route prediction CSVs in {experiment_dir} "
            f"(expected names like '<experiment_id>_3a.csv')"
        )
    frames = []
    for c in csvs:
        df = pd.read_csv(c, dtype={"cluster_id": str, "gene_symbol": str})
        missing = [col for col in PRED_COLS_REQUIRED if col not in df.columns]
        if missing:
            raise ValueError(f"{c.name} missing required columns: {missing}")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    # Normalize join keys
    for k in JOIN_KEYS:
        out[k] = out[k].astype(str).str.strip()
    return out


def load_ground_truth(path: Path) -> pd.DataFrame:
    gt = pd.read_csv(path, dtype={"cluster_id": str, "gene_symbol": str})
    for k in JOIN_KEYS:
        gt[k] = gt[k].astype(str).str.strip()
    return gt


def annotate_matches(joined: pd.DataFrame, reviewers: tuple[str, ...]) -> pd.DataFrame:
    """Add per-row match flag columns scored against the active reviewers."""
    out = joined.copy()
    out["predicted_class_norm"] = out["predicted_class"].apply(_norm_class)

    # --- Classification: per-reviewer, either, recomputed consensus ---------
    for r in reviewers:
        out[f"expert_class_{r}"] = out[f"expert_classification_{r}"].apply(_norm_class)
        out[f"classification_match_{r}"] = (
            (out[f"expert_class_{r}"] != "")
            & (out["predicted_class_norm"] == out[f"expert_class_{r}"])
        )

    def _expert_classes(row: pd.Series) -> list[str]:
        return [row[f"expert_class_{r}"] for r in reviewers if row[f"expert_class_{r}"]]

    out["classification_match_either"] = out.apply(
        lambda r: r["predicted_class_norm"] != ""
        and r["predicted_class_norm"] in _expert_classes(r),
        axis=1,
    )
    out["experts_agree"] = out.apply(
        lambda r: len(_expert_classes(r)) >= 1
        and len(set(_expert_classes(r))) == 1
        and len(_expert_classes(r)) == len([rv for rv in reviewers if r[f"expert_class_{rv}"]]),
        axis=1,
    )
    out["consensus_class"] = out.apply(
        lambda r: consensus_of([r[f"expert_class_{rv}"] for rv in reviewers]), axis=1
    )

    def _class_consensus(row: pd.Series) -> bool | None:
        if not row["consensus_class"]:
            return None  # reviewers disagree → not scored under consensus rule
        return row["predicted_class_norm"] == row["consensus_class"]

    out["classification_match_consensus"] = out.apply(_class_consensus, axis=1)

    # --- Confidence: per-reviewer + majority (2-reviewer ties → higher) -----
    out["majority_expert_confidence"] = out.apply(
        lambda r: majority_confidence(r, reviewers), axis=1
    )
    for r in reviewers:
        out[f"confidence_match_{r}"] = (
            (out[f"expert_confidence_{r}"].apply(_norm_lower) != "")
            & (
                out["pathway_confidence"].apply(_norm_lower)
                == out[f"expert_confidence_{r}"].apply(_norm_lower)
            )
        )
    out["confidence_consensus_match"] = (
        (out["majority_expert_confidence"].apply(_norm_lower) != "")
        & (
            out["pathway_confidence"].apply(_norm_lower)
            == out["majority_expert_confidence"].apply(_norm_lower)
        )
    )

    # --- Pathway: per-reviewer + either (any active reviewer) ---------------
    def _expert_pathways(row: pd.Series) -> list[str]:
        vals = [row.get(f"expert_pathway_{r}") for r in reviewers]
        return [v for v in vals if isinstance(v, str) and v.strip()]

    for r in reviewers:
        out[f"pathway_match_substring_{r}"] = out.apply(
            lambda x, rv=r: pathway_substring_match(x.get("pathway"), [x.get(f"expert_pathway_{rv}")]),
            axis=1,
        )
        out[f"pathway_match_loose_{r}"] = out.apply(
            lambda x, rv=r: pathway_loose_match(x.get("pathway"), [x.get(f"expert_pathway_{rv}")]),
            axis=1,
        )
    out["pathway_match_substring"] = out.apply(
        lambda r: pathway_substring_match(r.get("pathway"), _expert_pathways(r)), axis=1
    )
    out["pathway_match_loose"] = out.apply(
        lambda r: pathway_loose_match(r.get("pathway"), _expert_pathways(r)), axis=1
    )

    # --- Subclass: gated on recomputed consensus, voted over active reviewers -
    def _subclass_match(row: pd.Series) -> bool | None:
        if row["consensus_class"] not in {"NOVEL_ROLE", "UNCHARACTERIZED"}:
            return None  # not applicable (no consensus, or consensus is ESTABLISHED)
        pred_sc = _norm_class(row.get("predicted_subclass"))
        if not pred_sc:
            return False
        experts = [_norm_class(row.get(f"expert_subclass_{r}")) for r in reviewers]
        experts = [e for e in experts if e]
        if not experts:
            return None
        return pred_sc in experts

    out["subclass_match"] = out.apply(_subclass_match, axis=1)

    return out


# Metric columns whose values may be None (scored only on applicable rows).
NULLABLE_METRICS = ("classification_match_consensus", "subclass_match")


def _route_metric_cols(reviewers: tuple[str, ...]) -> list[str]:
    cols = [
        "classification_match_consensus",
        "classification_match_either",
    ]
    cols += [f"classification_match_{r}" for r in reviewers]
    cols += [
        "pathway_match_substring",
        "pathway_match_loose",
        "confidence_consensus_match",
        "subclass_match",
    ]
    return cols


def _agg_rate(grp: pd.DataFrame, metric: str) -> tuple[float | None, int]:
    """Mean of a boolean/None metric column over applicable rows; returns (rate, n)."""
    if metric in NULLABLE_METRICS:
        applicable = grp[grp[metric].notna()]
        n = len(applicable)
        return (round(applicable[metric].mean(), 3) if n else None), n
    return round(grp[metric].mean(), 3), len(grp)


def aggregate_per_route(joined: pd.DataFrame, reviewers: tuple[str, ...]) -> pd.DataFrame:
    """Per-route match rates across all genes."""
    metric_cols = _route_metric_cols(reviewers)
    rows = []
    for route, grp in joined.groupby("route", sort=True):
        rec: dict[str, Any] = {"route": route, "n_genes": len(grp)}
        for m in metric_cols:
            rate, n = _agg_rate(grp, m)
            rec[f"{m}_rate"] = rate
            if m in NULLABLE_METRICS:
                rec[f"{m}_n"] = n
        rows.append(rec)
    return pd.DataFrame(rows)


def aggregate_per_route_per_case_type(
    joined: pd.DataFrame, reviewers: tuple[str, ...]
) -> pd.DataFrame:
    """Per-route per-case-type match rates."""
    metric_cols = [
        "classification_match_consensus",
        "classification_match_either",
        "pathway_match_substring",
        "pathway_match_loose",
        "confidence_consensus_match",
    ]
    rows = []
    for (route, case_type), grp in joined.groupby(["route", "case_type"], sort=True):
        rec = {"route": route, "case_type": case_type, "n_genes": len(grp)}
        for m in metric_cols:
            rate, _ = _agg_rate(grp, m)
            rec[f"{m}_rate"] = rate
        rows.append(rec)
    return pd.DataFrame(rows)


def inter_reviewer_concordance(joined: pd.DataFrame, reviewers: tuple[str, ...]) -> dict[str, float]:
    """Pairwise + unanimous classification agreement among reviewers (the ceiling).
    Computed on the unique gene set (one route) to avoid counting genes once per route."""
    one_route = joined[joined["route"] == joined["route"].iloc[0]]
    stats: dict[str, float] = {}
    revs = list(reviewers)
    for i in range(len(revs)):
        for j in range(i + 1, len(revs)):
            a, b = revs[i], revs[j]
            both = (one_route[f"expert_class_{a}"] != "") & (one_route[f"expert_class_{b}"] != "")
            agree = both & (one_route[f"expert_class_{a}"] == one_route[f"expert_class_{b}"])
            stats[f"{a}_vs_{b}"] = round(agree.sum() / max(both.sum(), 1), 3)
    stats["unanimous"] = round(one_route["experts_agree"].mean(), 3)
    return stats


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown table without requiring `tabulate`."""
    if df.empty:
        return "_(empty)_"

    def _fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.3f}" if not pd.isna(v) else "—"
        if pd.isna(v):
            return "—"
        return str(v)

    cols = list(df.columns)
    rows = [[_fmt(v) for v in row] for row in df.itertuples(index=False)]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def write_report(
    out_path: Path,
    experiment_dir: Path,
    ground_truth_path: Path,
    reviewers: tuple[str, ...],
    joined: pd.DataFrame,
    per_route: pd.DataFrame,
    per_route_case_type: pd.DataFrame,
    concordance: dict[str, float],
) -> None:
    n_genes = len(joined)
    n_routes = joined["route"].nunique()
    n_clusters = joined[["screen_name", "cluster_id"]].drop_duplicates().shape[0]
    rev_str = " + ".join(reviewers)
    sections = []
    sections.append(f"# Evaluation report — `{experiment_dir.name}`\n")
    sections.append(f"- Experiment dir: `{experiment_dir}`")
    sections.append(f"- Ground truth: `{ground_truth_path}`")
    sections.append(f"- Reviewers scored: **{rev_str}** (operon excluded — LLM annotator)")
    sections.append(f"- Joined rows: {n_genes}  ({n_routes} routes × {n_clusters} clusters)\n")

    sections.append("## Inter-reviewer concordance (scoring ceiling)")
    sections.append(
        "Classification agreement among reviewers — a model cannot exceed this when "
        "scored against a consensus that requires agreement."
    )
    sections.append(_df_to_markdown(pd.DataFrame([concordance])))
    sections.append("")

    sections.append("## How classification is scored (2 reviewers)")
    sections.append(f"- `classification_match_{reviewers[0]}` / `_{reviewers[-1]}`: vs each reviewer individually.")
    sections.append("- `classification_match_either`: prediction matches at least one reviewer (ceiling 100%).")
    sections.append("- `classification_match_consensus`: matches the agreed label, scored ONLY on genes where reviewers agree (`_n` column gives the applicable count; ceiling 100% on that subset).")
    sections.append("")

    sections.append("## Heuristics (caveats)")
    sections.append(f"- `pathway_match_substring`: any active expert pathway is a case-insensitive substring of `pathway`. Model names are typically verbose; expert names short. Imperfect.")
    sections.append("- `pathway_match_loose`: bidirectional substring (catches inverse cases).")
    sections.append(f"- `confidence_consensus_match`: mode of `expert_confidence_*` over {rev_str}; 2-reviewer ties broken toward higher confidence.")
    sections.append("- `subclass_match`: counted only for genes whose recomputed consensus is NOVEL_ROLE or UNCHARACTERIZED.")
    sections.append("- Pathway matching is heuristic, not semantic. Swap in embeddings later if needed.\n")

    sections.append("## Per-route match rates")
    sections.append(_df_to_markdown(per_route))
    sections.append("")

    sections.append("## Per-route × per-case-type match rates")
    sections.append(_df_to_markdown(per_route_case_type))
    sections.append("")

    out_path.write_text("\n".join(sections), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate prediction CSVs against ground truth.")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Directory containing per-route prediction CSVs (e.g. benchmarking_outputs/1.arch/<id>)",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH,
        help="Path to benchmark_clusters_ground_truth.csv",
    )
    parser.add_argument(
        "--reviewers",
        default=",".join(DEFAULT_REVIEWERS),
        help="Comma-separated reviewers to score against (default 'eric,iain'; operon excluded).",
    )
    parser.add_argument(
        "--output-prefix",
        default="eval",
        help="Prefix for output files (default 'eval' → eval_per_gene.csv, etc.)",
    )
    args = parser.parse_args()

    reviewers = tuple(r.strip() for r in args.reviewers.split(",") if r.strip())
    if not reviewers:
        print("  [ERROR] No reviewers specified.")
        return 1

    preds = load_predictions(args.experiment_dir)
    gt = load_ground_truth(args.ground_truth)
    missing_gt = [
        f"expert_classification_{r}" for r in reviewers if f"expert_classification_{r}" not in gt.columns
    ]
    if missing_gt:
        print(f"  [ERROR] Ground truth missing reviewer columns: {missing_gt}")
        return 1

    joined = preds.merge(gt, on=list(JOIN_KEYS), how="inner", suffixes=("", "_gt"))
    if len(joined) == 0:
        print(
            f"  [WARN] No rows joined. predictions={len(preds)} ground_truth={len(gt)}\n"
            f"  Check that (screen_name, cluster_id, gene_symbol) match between sources."
        )
        return 1

    print(
        f"Joined {len(joined)} rows ({preds['route'].nunique()} routes × "
        f"{joined[['screen_name', 'cluster_id']].drop_duplicates().shape[0]} clusters). "
        f"Reviewers: {', '.join(reviewers)}."
    )

    joined = annotate_matches(joined, reviewers)
    concordance = inter_reviewer_concordance(joined, reviewers)
    per_route = aggregate_per_route(joined, reviewers)
    per_route_case_type = aggregate_per_route_per_case_type(joined, reviewers)

    per_gene_path = args.experiment_dir / f"{args.output_prefix}_per_gene.csv"
    per_route_path = args.experiment_dir / f"{args.output_prefix}_per_route.csv"
    per_case_type_path = args.experiment_dir / f"{args.output_prefix}_per_route_per_case_type.csv"
    report_path = args.experiment_dir / f"{args.output_prefix}_report.md"

    joined.to_csv(per_gene_path, index=False)
    per_route.to_csv(per_route_path, index=False)
    per_route_case_type.to_csv(per_case_type_path, index=False)
    write_report(
        report_path,
        args.experiment_dir,
        args.ground_truth,
        reviewers,
        joined,
        per_route,
        per_route_case_type,
        concordance,
    )

    print(f"  per-gene:     {per_gene_path}")
    print(f"  per-route:    {per_route_path}")
    print(f"  per-case-type: {per_case_type_path}")
    print(f"  report:       {report_path}")
    print(f"\n=== Inter-reviewer concordance (ceiling) ===\n  {concordance}")
    print("\n=== Per-route match rates ===")
    print(per_route.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
