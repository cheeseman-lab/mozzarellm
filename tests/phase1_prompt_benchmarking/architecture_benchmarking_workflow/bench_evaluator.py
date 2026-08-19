"""Benchmark evaluator: consensus ground truth + per-run metric generation.

Builds consensus ground truth from the raw per-reviewer annotations
(>=2-of-3 majority classification, ordinal-median subclass, majority coherence)
and scores a run's parsed_outputs.jsonl against it:

- per-run scoring: gene-category recall by modal vote across replicates (a
  hedged gene resolves to the less-high call), novel/uncharacterized subclass,
  and cluster coherence -> MetricPanel;
- decoy validation: abstain/functional expectations on the negative-control
  clusters, with output-completeness reporting;
- diagnostics: per-consensus-class recall, lenient any-reviewer recall,
  inter-reviewer concordance (pairwise / Fleiss / by level), the reviewers'
  de-blinded source-preference tally, and audit-flag engagement;
- pathway agreement: the model's modal dominant process vs reviewers' nominated
  pathways (substring / bidirectional substring / optional semantic cosine via
  a sentence-transformer).

Library consumed by the pipeline runners -- no CLI. State files written by the
runners are the record; nothing is re-derived downstream.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

# Optional semantic pathway scoring via sentence-transformers.
# Degrades gracefully: if not installed, semantic columns are omitted and
# a single informational message is printed.
try:
    import numpy as _np
    from sentence_transformers import SentenceTransformer

    _SEMANTIC_AVAILABLE = True
except ImportError:
    _SEMANTIC_AVAILABLE = False

_SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"
_semantic_model: Any = None  # lazy-loaded on first use

PHASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GROUND_TRUTH = PHASE_DIR / "benchmark_inputs" / "benchmark_combined_expert_annotations.csv"

# Human reviewers used for scoring. operon is intentionally excluded — it is an
# LLM annotator, so including it makes the benchmark LLM-evaluating-LLM.
DEFAULT_REVIEWERS = ("eric", "iain", "liz")
JOIN_KEYS = ("screen_name", "cluster_id", "gene_symbol")

# Stable suffixes of evaluator output CSVs (independent of the phase prefix), used
# to exclude them when globbing prediction CSVs so re-runs don't re-ingest them.
_EVAL_OUTPUT_SUFFIXES = (
    "_per_gene.csv",
    "_per_route.csv",
    "_per_route_per_case_type.csv",
    "_negative_abstention.csv",
    "_coverage.csv",
    "_output_fragility.csv",
)

# Shuffled clusters carry no expert pathway annotation by construction; scored
# on whether the model correctly emits "no coherent pathway" + empty arrays.
NEGATIVE_CLUSTERS = (
    ("aconcagua_interphase_shuffled", "244", "no_coherent_pathway"),
    ("aconcagua_interphase_shuffled", "3", "no_coherent_pathway"),
    ("aconcagua_interphase_shuffled", "17", "no_coherent_pathway"),
)

# Output-fragility diagnostic: a large coherent cluster (jebel/0, 147 genes)
# with no expert annotations. Reported separately from accuracy / abstention.
OUTPUT_FRAGILITY_CLUSTER = ("jebel", "0")

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
    """Most-frequent pathway_confidence_{r} across active reviewers. Ties resolve
    toward higher confidence. Empty if no votes."""
    votes = [_norm_str(row.get(f"pathway_confidence_{r}")) for r in reviewers]
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


def _get_semantic_model() -> Any:
    """Lazy-load the sentence-transformer model (~80 MB download on first use)."""
    global _semantic_model
    if _semantic_model is None:
        print(f"  [semantic] Loading {_SEMANTIC_MODEL_NAME} (downloads ~80 MB on first use)...")
        _semantic_model = SentenceTransformer(_SEMANTIC_MODEL_NAME)
    return _semantic_model


def compute_semantic_scores(
    joined: pd.DataFrame,
    reviewers: tuple[str, ...],
    threshold: float = 0.70,
) -> pd.DataFrame:
    """Batch-encode all pathway strings and annotate with per-gene semantic similarity.

    Adds two columns to the returned DataFrame:
    - pathway_semantic_score: max cosine similarity to any active reviewer's
      nominated_pathway (float, NaN when either side is empty)
    - pathway_semantic_match: score >= threshold (bool)

    All unique non-empty pathway strings (predicted + all reviewer nominations)
    are encoded in a single batch call — one model forward pass regardless of
    the number of routes or rows.

    Returns joined unchanged (NaN/False semantic columns) when sentence-transformers
    is not installed.
    """
    out = joined.copy()
    nan_fill = [float("nan")] * len(out)
    false_fill = [False] * len(out)

    if not _SEMANTIC_AVAILABLE:
        out["pathway_semantic_score"] = nan_fill
        out["pathway_semantic_match"] = false_fill
        return out

    model = _get_semantic_model()
    expert_cols = [f"nominated_pathway_{r}" for r in reviewers]

    # Collect all unique non-empty strings across predictions + all expert columns.
    all_strings: set[str] = set()
    for s in out["pathway"].dropna():
        s = str(s).strip()
        if s:
            all_strings.add(s)
    for col in expert_cols:
        if col in out.columns:
            for s in out[col].dropna():
                s = str(s).strip()
                if s:
                    all_strings.add(s)

    if not all_strings:
        out["pathway_semantic_score"] = nan_fill
        out["pathway_semantic_match"] = false_fill
        return out

    str_list = sorted(all_strings)
    str_to_idx = {s: i for i, s in enumerate(str_list)}

    print(
        f"  [semantic] Encoding {len(str_list)} unique pathway strings "
        f"with {_SEMANTIC_MODEL_NAME}..."
    )
    # numpy arrays (n_strings, d) — avoids torch tensor handling entirely.
    embeddings = model.encode(str_list, convert_to_tensor=False, show_progress_bar=False)
    # Pre-normalise all rows once so similarity = dot product.
    norms = _np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / _np.maximum(norms, 1e-9)

    scores: list[float] = []
    for _, row in out.iterrows():
        pred = str(row.get("pathway") or "").strip()
        if not pred or pred not in str_to_idx:
            scores.append(float("nan"))
            continue

        expert_indices = [
            str_to_idx[str(row.get(col) or "").strip()]
            for col in expert_cols
            if str(row.get(col) or "").strip() in str_to_idx
        ]
        if not expert_indices:
            scores.append(float("nan"))
            continue

        pred_vec = embeddings[str_to_idx[pred]]  # (d,) normalised
        exp_vecs = embeddings[expert_indices]  # (n, d) normalised
        sims = exp_vecs @ pred_vec  # (n,) dot products = cosine sims
        scores.append(round(float(_np.max(sims)), 3))

    out["pathway_semantic_score"] = scores
    out["pathway_semantic_match"] = [
        bool(s >= threshold) if s == s else False  # NaN check via self-equality
        for s in out["pathway_semantic_score"]
    ]
    out["pathway_semantic_match_loose"] = [
        bool(s >= 0.60) if s == s else False for s in out["pathway_semantic_score"]
    ]
    return out


def load_predictions(experiment_dir: Path) -> pd.DataFrame:
    """Stack every prediction CSV in the experiment dir into one frame.

    Excludes eval_*.csv (evaluator outputs) and aggregate_summary.csv.
    """
    csvs = sorted(experiment_dir.glob("*.csv"))
    # Exclude evaluator outputs (any prefix, e.g. eval_*, arch_eval_*) by their
    # stable suffixes, plus the aggregate summary, so re-runs never re-ingest them.
    csvs = [
        c
        for c in csvs
        if not c.name.startswith("eval_")
        and not c.name.endswith(_EVAL_OUTPUT_SUFFIXES)
        and c.name != "aggregate_summary.csv"
    ]
    if not csvs:
        raise FileNotFoundError(
            f"No per-route prediction CSVs in {experiment_dir}. "
            f"Run bench_trace_parser.py first to generate them."
        )
    frames = []
    for c in csvs:
        df = pd.read_csv(c, dtype={"cluster_id": str, "gene_symbol": str})
        missing = [col for col in PRED_COLS_REQUIRED if col not in df.columns]
        if missing:
            raise ValueError(f"{c.name} missing required columns: {missing}")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    for k in JOIN_KEYS:
        out[k] = out[k].astype(str).str.strip()
    return out


def load_ground_truth(path: Path) -> pd.DataFrame:
    """Load the ground-truth CSV.

    Renames `sheet` → `screen_name` to match the prediction CSV join key.
    """
    gt = pd.read_csv(path, dtype={"cluster_id": str, "gene_symbol": str})
    if "sheet" in gt.columns and "screen_name" not in gt.columns:
        gt = gt.rename(columns={"sheet": "screen_name"})
    for k in JOIN_KEYS:
        gt[k] = gt[k].astype(str).str.strip()
    return gt


def annotate_matches(joined: pd.DataFrame, reviewers: tuple[str, ...]) -> pd.DataFrame:
    """Add per-row match flag columns scored against the active reviewers."""
    out = joined.copy()
    out["predicted_class_norm"] = out["predicted_class"].apply(_norm_class)

    # --- Classification: per-reviewer, either, recomputed consensus ----------
    for r in reviewers:
        out[f"expert_class_{r}"] = out[f"expert_classification_{r}"].apply(_norm_class)
        out[f"classification_match_{r}"] = (out[f"expert_class_{r}"] != "") & (
            out["predicted_class_norm"] == out[f"expert_class_{r}"]
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

    # --- Confidence: per-reviewer + majority (ties → higher) ----------------
    # NOTE: ground truth uses pathway_confidence_{r}; predictions use pathway_confidence.
    # After the merge, pathway_confidence = model output; pathway_confidence_{r} = expert.
    out["majority_expert_confidence"] = out.apply(
        lambda r: majority_confidence(r, reviewers), axis=1
    )
    for r in reviewers:
        out[f"confidence_match_{r}"] = (out[f"pathway_confidence_{r}"].apply(_norm_lower) != "") & (
            out["pathway_confidence"].apply(_norm_lower)
            == out[f"pathway_confidence_{r}"].apply(_norm_lower)
        )
    out["confidence_consensus_match"] = (
        out["majority_expert_confidence"].apply(_norm_lower) != ""
    ) & (
        out["pathway_confidence"].apply(_norm_lower)
        == out["majority_expert_confidence"].apply(_norm_lower)
    )

    # --- Pathway: per-reviewer + any-reviewer --------------------------------
    # Ground truth uses nominated_pathway_{r} for expert pathway nominations.
    def _expert_pathways(row: pd.Series) -> list[str]:
        vals = [row.get(f"nominated_pathway_{r}") for r in reviewers]
        return [v for v in vals if isinstance(v, str) and v.strip()]

    for r in reviewers:
        out[f"pathway_match_substring_{r}"] = out.apply(
            lambda x, rv=r: pathway_substring_match(
                x.get("pathway"), [x.get(f"nominated_pathway_{rv}")]
            ),
            axis=1,
        )
        out[f"pathway_match_loose_{r}"] = out.apply(
            lambda x, rv=r: pathway_loose_match(
                x.get("pathway"), [x.get(f"nominated_pathway_{rv}")]
            ),
            axis=1,
        )
    out["pathway_match_substring"] = out.apply(
        lambda r: pathway_substring_match(r.get("pathway"), _expert_pathways(r)), axis=1
    )
    out["pathway_match_loose"] = out.apply(
        lambda r: pathway_loose_match(r.get("pathway"), _expert_pathways(r)), axis=1
    )

    # --- Subclass: gated on recomputed consensus for NOVEL_ROLE / UNCHARACTERIZED ---
    def _subclass_match(row: pd.Series) -> bool | None:
        if row["consensus_class"] not in {"NOVEL_ROLE", "UNCHARACTERIZED"}:
            return None  # not applicable
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
    has_semantic = "pathway_semantic_score" in joined.columns
    rows = []
    for route, grp in joined.groupby("route", sort=True):
        rec: dict[str, Any] = {"route": route, "n_genes": len(grp)}
        for m in metric_cols:
            rate, n = _agg_rate(grp, m)
            rec[f"{m}_rate"] = rate
            if m in NULLABLE_METRICS:
                rec[f"{m}_n"] = n
        if has_semantic:
            valid = grp["pathway_semantic_score"].dropna()
            rec["pathway_semantic_score_mean"] = (
                round(float(valid.mean()), 3) if len(valid) else float("nan")
            )
            rec["pathway_semantic_match_rate"] = round(
                float(grp["pathway_semantic_match"].mean()), 3
            )
            if "pathway_semantic_match_loose" in grp.columns:
                rec["pathway_semantic_match_loose_rate"] = round(
                    float(grp["pathway_semantic_match_loose"].mean()), 3
                )
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
    has_semantic = "pathway_semantic_score" in joined.columns
    rows = []
    for (route, case_type), grp in joined.groupby(["route", "benchmark_case_type"], sort=True):
        rec = {"route": route, "case_type": case_type, "n_genes": len(grp)}
        for m in metric_cols:
            rate, _ = _agg_rate(grp, m)
            rec[f"{m}_rate"] = rate
        if has_semantic:
            valid = grp["pathway_semantic_score"].dropna()
            rec["pathway_semantic_score_mean"] = (
                round(float(valid.mean()), 3) if len(valid) else float("nan")
            )
            rec["pathway_semantic_match_rate"] = round(
                float(grp["pathway_semantic_match"].mean()), 3
            )
        rows.append(rec)
    return pd.DataFrame(rows)


def inter_reviewer_concordance(
    joined: pd.DataFrame, reviewers: tuple[str, ...]
) -> dict[str, float]:
    """Pairwise + unanimous classification agreement among reviewers (the ceiling).

    Computed on the unique gene set (one route) to avoid inflating by route count.
    """
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


def compute_negative_abstention(experiment_dir: Path) -> pd.DataFrame:
    """Per-route abstention rate on the shuffled negative-control clusters.

    Reads parsed_outputs.jsonl directly so cells with empty gene arrays (correct
    abstention) are counted. Abstain iff `dominant_process` contains "no coherent"
    AND `pathway_confidence` is Low AND all three gene-classification arrays are
    empty.
    """
    parsed_path = experiment_dir / "parsed_outputs.jsonl"
    if not parsed_path.exists():
        return pd.DataFrame()
    neg_lookup = {(s, str(c).strip()): ct for s, c, ct in NEGATIVE_CLUSTERS}
    cell_rows: list[dict] = []
    for line in parsed_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        run_id = d.get("run_id", "")
        route = d.get("route", "")
        parts = run_id.split("__")
        if len(parts) < 5:
            continue
        screen = parts[-3]
        cluster = parts[-2].replace("cluster_", "")
        if (screen, cluster.strip()) not in neg_lookup:
            continue
        po = d.get("parsed_output") or {}
        proc = _norm_lower(po.get("dominant_process") or "")
        conf = _norm_lower(po.get("pathway_confidence") or "")
        n_genes = (
            len(po.get("established_genes") or [])
            + len(po.get("novel_role_genes") or [])
            + len(po.get("uncharacterized_genes") or [])
        )
        abstain = ("no coherent" in proc) and (conf == "low") and (n_genes == 0)
        cell_rows.append(
            {
                "route": route,
                "case_type": neg_lookup[(screen, cluster.strip())],
                "abstain": abstain,
            }
        )
    if not cell_rows:
        return pd.DataFrame()
    cells = pd.DataFrame(cell_rows)
    rows = []
    for (route, ct), grp in cells.groupby(["route", "case_type"]):
        rate = float(grp["abstain"].mean())
        rows.append(
            {
                "route": route,
                "case_type": ct,
                "n_cells": len(grp),
                "abstain_rate": round(rate, 3),
                "fabrication_rate": round(1 - rate, 3),
            }
        )
    return pd.DataFrame(rows)


def compute_coverage(
    preds: pd.DataFrame,
    clusters_csv_path: Path,
    ground_truth: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-route coverage on unique gene symbols.

    Reports three rates so the "coverage drop" signal isn't confounded by the
    shuffled-controls case where lower coverage is correct:
    - coverage_rate_all: across every cluster in the benchmark
    - coverage_rate_real: excludes the shuffled negative-control clusters
    - coverage_rate_categorized: also excludes clusters with benchmark_case_type=NaN
      (PI consensus failed; abstaining there is a defensible call)
    """
    if not clusters_csv_path.exists():
        return pd.DataFrame()
    clusters = pd.read_csv(clusters_csv_path, dtype={"cluster_id": str, "gene_symbol": str})
    if "sheet" in clusters.columns and "screen_name" not in clusters.columns:
        clusters = clusters.rename(columns={"sheet": "screen_name"})
    if not {"screen_name", "cluster_id", "gene_symbol"}.issubset(clusters.columns):
        return pd.DataFrame()
    expected_per_cluster = (
        clusters.drop_duplicates(["screen_name", "cluster_id", "gene_symbol"])
        .groupby(["screen_name", "cluster_id"])
        .size()
    )
    shuffled_keys = {(s, str(c).strip()) for s, c, _ in NEGATIVE_CLUSTERS}
    uncategorized_keys: set[tuple[str, str]] = set()
    if ground_truth is not None and "benchmark_case_type" in ground_truth.columns:
        defined = {
            (s, str(c).strip())
            for s, c in ground_truth.dropna(subset=["benchmark_case_type"])
            .drop_duplicates(["screen_name", "cluster_id"])[["screen_name", "cluster_id"]]
            .itertuples(index=False)
        }
        all_clusters = {(s, str(c).strip()) for (s, c) in expected_per_cluster.index}
        uncategorized_keys = all_clusters - defined - shuffled_keys
    full_expected = {(s, str(c).strip()): n for (s, c), n in expected_per_cluster.items()}
    rows = []
    for route, grp in preds.groupby("route", sort=True):
        # defense/future-proofing: use actual IDs rather than range(1, N+1)
        replicate_ids = sorted(grp["replicate"].unique())
        per_cell = (
            grp.groupby(["screen_name", "cluster_id", "replicate"])["gene_symbol"]
            .nunique()
            .reset_index(name="n_unique_pred")
        )
        all_pairs = [
            (s, str(c).strip(), rep)
            for (s, c) in expected_per_cluster.index
            for rep in replicate_ids
        ]
        all_df = pd.DataFrame(all_pairs, columns=["screen_name", "cluster_id", "replicate"])
        merged = all_df.merge(per_cell, on=["screen_name", "cluster_id", "replicate"], how="left")
        merged["n_unique_pred"] = merged["n_unique_pred"].fillna(0)
        merged["expected"] = merged.apply(
            lambda r: full_expected.get((r["screen_name"], r["cluster_id"]), 0), axis=1
        )
        merged["is_shuffled"] = merged.apply(
            lambda r: (r["screen_name"], r["cluster_id"]) in shuffled_keys, axis=1
        )
        merged["is_uncategorized"] = merged.apply(
            lambda r: (r["screen_name"], r["cluster_id"]) in uncategorized_keys, axis=1
        )
        total_pred_all = int(merged["n_unique_pred"].sum())
        total_exp_all = int(merged["expected"].sum())
        real = merged[~merged["is_shuffled"]]
        cat = merged[~merged["is_shuffled"] & ~merged["is_uncategorized"]]
        rows.append(
            {
                "route": route,
                "n_replicates": len(replicate_ids),
                "coverage_rate_all": round(total_pred_all / total_exp_all, 3)
                if total_exp_all
                else 0.0,
                "coverage_rate_real": round(
                    int(real["n_unique_pred"].sum()) / int(real["expected"].sum()), 3
                )
                if int(real["expected"].sum())
                else 0.0,
                "coverage_rate_categorized": round(
                    int(cat["n_unique_pred"].sum()) / int(cat["expected"].sum()), 3
                )
                if int(cat["expected"].sum())
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compute_output_fragility(preds: pd.DataFrame, clusters_csv_path: Path) -> pd.DataFrame:
    """Per-route diagnostic on the output_fragility cluster (jebel/0, 147 genes).

    No expert annotations exist for this cluster, so it is not scored on accuracy
    or abstention. Reports per-route coverage and pathway-consistency to surface
    output-structure failures on a large coherent input.
    """
    screen, cid = OUTPUT_FRAGILITY_CLUSTER
    cell = preds[(preds["screen_name"] == screen) & (preds["cluster_id"].astype(str) == cid)].copy()
    if cell.empty:
        return pd.DataFrame()
    expected = 0
    if clusters_csv_path.exists():
        clusters = pd.read_csv(clusters_csv_path, dtype={"cluster_id": str, "gene_symbol": str})
        if "sheet" in clusters.columns and "screen_name" not in clusters.columns:
            clusters = clusters.rename(columns={"sheet": "screen_name"})
        expected = int(
            clusters[
                (clusters["screen_name"] == screen) & (clusters["cluster_id"].astype(str) == cid)
            ]
            .drop_duplicates("gene_symbol")
            .shape[0]
        )
    rows = []
    for route, grp in cell.groupby("route", sort=True):
        n_replicates = int(grp["replicate"].nunique())
        per_rep_counts = grp.groupby("replicate")["gene_symbol"].nunique()
        median_per_cell = float(per_rep_counts.median()) if not per_rep_counts.empty else 0.0
        total_pred = int(per_rep_counts.sum())
        expected_total = expected * max(n_replicates, 1)
        coverage = total_pred / expected_total if expected_total else 0.0
        per_rep_consistent = grp.drop_duplicates("replicate").assign(
            _ok=lambda d: d["pathway"]
            .astype(str)
            .str.lower()
            .apply(lambda s: "ribosom" in s or "translation" in s)
        )
        consistency_rate = (
            float(per_rep_consistent["_ok"].mean()) if len(per_rep_consistent) else 0.0
        )
        rows.append(
            {
                "route": route,
                "n_replicates": n_replicates,
                "median_genes_per_cell": round(median_per_cell, 1),
                "n_expected_per_cell": expected,
                "coverage_rate": round(float(coverage), 3),
                "pathway_consistency_rate": round(consistency_rate, 3),
            }
        )
    return pd.DataFrame(rows)
