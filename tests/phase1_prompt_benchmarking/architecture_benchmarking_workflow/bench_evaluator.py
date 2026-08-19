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

import csv
import json
from collections import Counter, defaultdict
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

# Ordinal evidence ladders per category (rank ascending = more support for the
# call). CONTRADICTORY_EVIDENCE sits at the bottom of the NOVEL_ROLE ladder --
# most skeptical -- so a CONTRADICTORY majority is respected by the median while
# scattered votes resolve to the conservative middle. DARK_GENE < ANNOTATED_ONLY.
NOVEL_LADDER = {
    "CONTRADICTORY_EVIDENCE": 0,
    "NO_EVIDENCE": 1,
    "INDIRECT_EVIDENCE": 2,
    "PARTIAL_EVIDENCE": 3,
}
UNCHAR_LADDER = {"DARK_GENE": 0, "ANNOTATED_ONLY": 1}
_SUBCLASS_LADDERS = {"NOVEL_ROLE": NOVEL_LADDER, "UNCHARACTERIZED": UNCHAR_LADDER}

REAL_COLUMNS = [
    "screen",
    "cluster",
    "cluster_role",
    "gene",
    "consensus_class",
    "unanimous",
    "n_agree",
    "consensus_subclass",
    "pref_affinage",
    "pref_uniprot",
    "pref_both",
    "pref_neither",
]

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


def consensus_of(classifications: list[str]) -> tuple[str, bool, int]:
    """Strict >=2-of-3 majority over classification labels: (winner, unanimous, n_agree).

    Returns ("", False, top_count) when no label reaches a >=2 majority (a 1-1-1
    three-way split) -- the gene is then unresolved and excluded from consensus
    scoring. On the 133-gene benchmark there are no such splits.
    """
    present = [c for c in classifications if c]
    if not present:
        return "", False, 0
    counts = Counter(present)
    winner, n_agree = counts.most_common(1)[0]
    if n_agree < 2:
        return "", False, n_agree
    return winner, len(counts) == 1, n_agree


def _consensus_subclass(subclasses: list[str], consensus_class: str) -> str:
    """Ordinal-median subclass over votes on the consensus category's ladder.

    Only NOVEL_ROLE / UNCHARACTERIZED carry a subclass. Votes off the relevant
    ladder (e.g. a NOVEL_ROLE subclass on an UNCHARACTERIZED-consensus gene) are
    dropped; the median of the remaining ranks is taken with the lower median on
    even counts (the more conservative call). Empty if no on-ladder vote exists.
    """
    ladder = _SUBCLASS_LADDERS.get(consensus_class)
    if ladder is None:
        return ""
    ranks = sorted(ladder[s] for s in subclasses if s in ladder)
    if not ranks:
        return ""
    median_rank = ranks[(len(ranks) - 1) // 2]
    return {v: k for k, v in ladder.items()}[median_rank]


def _coherence_label(coherence_text: str) -> str | None:
    """Map a reviewer's free-text coherence to High/Medium/Low, or None."""
    text = coherence_text.lower()
    for level in ("high", "medium", "low"):
        if level in text:
            return level.capitalize()
    return None


def consensus_coherence(reviewer_csvs: dict[str, Path]) -> dict[tuple, str]:
    """Majority-vote per-cluster coherence (High/Medium/Low) across reviewers.

    Reads the cluster-level `coherence` free text from each reviewer's rows,
    maps it to a level, and returns {(screen, cluster): level} for clusters with
    a strict majority. Ties (e.g. one High, one Medium, one Low) are excluded.
    """
    per_cluster: dict[tuple, list[str]] = defaultdict(list)
    for path in reviewer_csvs.values():
        with open(path, newline="") as fh:
            seen: set[tuple] = set()
            for row in csv.DictReader(fh):
                cluster_key = (row["screen"].strip(), row["cluster"].strip())
                if cluster_key in seen:
                    continue  # one coherence vote per reviewer per cluster
                seen.add(cluster_key)
                label = _coherence_label(row["coherence"])
                if label:
                    per_cluster[cluster_key].append(label)

    result: dict[tuple, str] = {}
    for cluster_key, labels in per_cluster.items():
        counts = Counter(labels)
        winner, top = counts.most_common(1)[0]
        if sum(1 for c in counts.values() if c == top) == 1:
            result[cluster_key] = winner
    return result


def _deblind(preferred_source: str, key_row: dict) -> str:
    """Map a reviewer's blinded 'a'/'b' pick to its real source via the key row."""
    choice = preferred_source.strip().lower()
    if choice == "a":
        return key_row["srcA_src"].strip().lower()
    if choice == "b":
        return key_row["srcB_src"].strip().lower()
    return choice


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


def build_consensus_gt(
    reviewer_csvs: dict[str, Path],
    key_csv: Path,
    decoy_specs: list[dict],
    out_csv: Path,
) -> None:
    """Build the consensus ground-truth CSV from reviewer annotations + blinding key.

    Args:
        reviewer_csvs: mapping of reviewer name -> path to their annotation CSV.
        key_csv: path to the blinding key (screen, cluster, gene, srcA_src,
            srcB_src).
        decoy_specs: list of {"screen", "cluster", "decoy_type", "genes"} dicts;
            each gene becomes a blank negative-control row.
        out_csv: destination path for the consensus ground-truth CSV.
    """
    key: dict[tuple, dict] = {}
    with open(key_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            key[(row["screen"].strip(), row["cluster"].strip(), row["gene"].strip())] = row

    per_gene: dict[tuple, list[dict]] = defaultdict(list)
    for path in reviewer_csvs.values():
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                gene_key = (row["screen"].strip(), row["cluster"].strip(), row["gene"].strip())
                per_gene[gene_key].append(row)

    rows = []
    for (screen, cluster, gene), revs in per_gene.items():
        key_row = key[(screen, cluster, gene)]

        consensus_class, unanimous, n_agree = consensus_of(
            [r["classification"].strip() for r in revs]
        )
        consensus_subclass = _consensus_subclass(
            [r["subclass"].strip() for r in revs], consensus_class
        )

        pref_counts = Counter(_deblind(r["preferred_source"], key_row) for r in revs)

        rows.append(
            {
                "screen": screen,
                "cluster": cluster,
                "cluster_role": "real",
                "gene": gene,
                "consensus_class": consensus_class,
                "unanimous": unanimous,
                "n_agree": n_agree,
                "consensus_subclass": consensus_subclass,
                "pref_affinage": pref_counts.get("affinage", 0),
                "pref_uniprot": pref_counts.get("uniprot", 0),
                "pref_both": pref_counts.get("both", 0),
                "pref_neither": pref_counts.get("neither", 0),
            }
        )

    for spec in decoy_specs:
        for gene in spec["genes"]:
            rows.append(
                {
                    "screen": spec["screen"],
                    "cluster": spec["cluster"],
                    "cluster_role": "decoy",
                    "gene": gene,
                    "consensus_class": "",
                    "unanimous": "",
                    "n_agree": "",
                    "consensus_subclass": "",
                    "pref_affinage": "",
                    "pref_uniprot": "",
                    "pref_both": "",
                    "pref_neither": "",
                }
            )

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REAL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def load_consensus_gt(csv_path: Path) -> dict[tuple, dict]:
    """Load a consensus ground-truth CSV into {(screen, cluster, gene): row_dict}."""
    result = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            result[(row["screen"], row["cluster"], row["gene"])] = row
    return result


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
