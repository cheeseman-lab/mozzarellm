"""Build and load reviewer-consensus ground truth for gene-cluster classification.

Consumes per-reviewer annotation CSVs (annotator, screen, cluster, pathway,
coherence, cluster_notes, gene, classification, subclass, preferred_source,
used_web) plus a blinding key CSV (screen, cluster, gene, srcA_src, srcB_src,
flagged, case_type) and produces one consensus row per (screen, cluster, gene),
majority-voting classification/subclass across reviewers and de-blinding
preferred_source into pref_affinage/pref_uniprot/pref_both/pref_neither tallies.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

REAL_COLUMNS = [
    "screen",
    "cluster",
    "cluster_role",
    "case_type",
    "flagged",
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


def _deblind(preferred_source: str, key_row: dict) -> str:
    """Map a reviewer's blinded 'a'/'b' pick to its real source via the key row."""
    choice = preferred_source.strip().lower()
    if choice == "a":
        return key_row["srcA_src"].strip().lower()
    if choice == "b":
        return key_row["srcB_src"].strip().lower()
    return choice


def _consensus_class(classifications: list[str]) -> tuple[str, bool, int]:
    """Majority vote over classification labels: (winner, unanimous, n_agree)."""
    counts = Counter(classifications)
    winner, n_agree = counts.most_common(1)[0]
    unanimous = len(counts) == 1
    return winner, unanimous, n_agree


def _consensus_subclass(subclasses: list[str]) -> str:
    """Majority vote over non-empty subclass labels; empty if none present."""
    present = [s for s in subclasses if s]
    if not present:
        return ""
    return Counter(present).most_common(1)[0][0]


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
            srcB_src, flagged, case_type).
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

        consensus_class, unanimous, n_agree = _consensus_class(
            [r["classification"].strip() for r in revs]
        )
        consensus_subclass = _consensus_subclass([r["subclass"].strip() for r in revs])

        pref_counts = Counter(_deblind(r["preferred_source"], key_row) for r in revs)

        rows.append(
            {
                "screen": screen,
                "cluster": cluster,
                "cluster_role": "real",
                "case_type": key_row["case_type"],
                "flagged": key_row["flagged"],
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
                    "case_type": "negative_control",
                    "flagged": "",
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
