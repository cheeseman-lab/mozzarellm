"""Score a model run's parsed_outputs.jsonl against reviewer-consensus ground truth.

Reproduces the metric panel (category accuracy, novel/uncharacterized subclass
accuracy, cluster coherence) used to validate the benchmarking harness against
known cached numbers before spending API tokens on new runs.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

_RUN_ID_RE = re.compile(r"^(?P<screen>.+)__cluster_(?P<cluster>[^_]+)__rep_(?P<rep>.+)$")


@dataclass
class MetricPanel:
    category: float
    novel_subclass: tuple
    unchar_subclass: tuple
    coherence: tuple
    n: int
    failures: int


def _parse_run_id(run_id: str, route: str) -> tuple[str, str]:
    """Extract (screen, cluster) from a run_id, anchored on the cell's own route."""
    marker = f"__{route}__"
    idx = run_id.find(marker)
    if idx == -1:
        raise ValueError(f"route {route!r} not found in run_id {run_id!r}")
    remainder = run_id[idx + len(marker) :]
    match = _RUN_ID_RE.match(remainder)
    if not match:
        raise ValueError(f"could not parse screen/cluster from run_id {run_id!r}")
    return match.group("screen"), match.group("cluster")


def _gene_symbol(item) -> str:
    """Normalize an established_genes entry to a bare gene symbol."""
    if isinstance(item, dict):
        return next(iter(item))
    return item


def score_run(
    run_dir: Path,
    gt: dict,
    cluster_coherence: dict | None = None,
    route_excludes: str = "mcp",
    route_includes: str | None = None,
    route_equals: str | None = None,
) -> MetricPanel:
    """Score one run directory's parsed_outputs.jsonl against consensus ground truth.

    Route filtering precedence (first that is set wins): route_equals keeps only
    rows whose route matches exactly (used to isolate one variant when several
    share a prefix, e.g. W5 vs W51); route_includes keeps rows whose route
    contains it (convenience for the mcp cell, a superset of the non-mcp name);
    otherwise skip rows matching the route_excludes substring.
    """
    run_dir = Path(run_dir)

    gene_votes: dict[tuple, Counter] = defaultdict(Counter)
    subclass_votes: dict[tuple, dict] = defaultdict(lambda: defaultdict(Counter))
    cluster_confidence_votes: dict[tuple, Counter] = defaultdict(Counter)
    failures = 0

    with open(run_dir / "parsed_outputs.jsonl") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cell = json.loads(line)
            route = cell.get("route", "")
            if route_equals is not None:
                if route != route_equals:
                    continue
            elif route_includes is not None:
                if route_includes not in route:
                    continue
            elif route_excludes and route_excludes in route:
                continue

            parsed = cell.get("parsed_output")
            if parsed is None or parsed.get("dominant_process") is None:
                failures += 1
                continue

            screen, cluster = _parse_run_id(cell["run_id"], route)
            cluster_key = (screen, cluster)

            # A gene landing in more than one category list within this single
            # reply is a self-contradiction (the model wasn't decisive) — that
            # reply casts no vote for that gene. Collect this cell's per-gene
            # category -> subclass mapping first so we can detect and drop
            # those before folding anything into the cross-replicate tally.
            cell_categories: dict[str, dict[str, str]] = defaultdict(dict)

            for item in parsed.get("established_genes") or []:
                gene = _gene_symbol(item)
                cell_categories[gene]["ESTABLISHED"] = ""

            for item in parsed.get("novel_role_genes") or []:
                cell_categories[item["gene"]]["NOVEL_ROLE"] = item.get("class", "")

            for item in parsed.get("uncharacterized_genes") or []:
                cell_categories[item["gene"]]["UNCHARACTERIZED"] = item.get("class", "")

            for gene, categories in cell_categories.items():
                if len(categories) != 1:
                    continue  # self-contradicting reply: drop this vote
                category, subclass = next(iter(categories.items()))
                gene_votes[(screen, cluster, gene)][category] += 1
                if subclass:
                    subclass_votes[(screen, cluster, gene)][category][subclass] += 1

            confidence = parsed.get("pathway_confidence")
            if confidence:
                cluster_confidence_votes[cluster_key][confidence] += 1

    category_correct = 0
    category_n = 0
    novel_correct = 0
    novel_n = 0
    unchar_correct = 0
    unchar_n = 0

    for (screen, cluster, gene), votes in gene_votes.items():
        row = gt.get((screen, cluster, gene))
        if row is None or row.get("cluster_role") != "real":
            continue

        modal_category = votes.most_common(1)[0][0]
        category_n += 1
        consensus_class = row.get("consensus_class", "")
        if modal_category == consensus_class:
            category_correct += 1

        consensus_subclass = row.get("consensus_subclass", "")

        if (
            consensus_class == "NOVEL_ROLE"
            and consensus_subclass
            and modal_category == "NOVEL_ROLE"
        ):
            sub_counter = subclass_votes[(screen, cluster, gene)]["NOVEL_ROLE"]
            modal_subclass = sub_counter.most_common(1)[0][0] if sub_counter else None
            novel_n += 1
            if modal_subclass == consensus_subclass:
                novel_correct += 1

        if (
            consensus_class == "UNCHARACTERIZED"
            and consensus_subclass
            and modal_category == "UNCHARACTERIZED"
        ):
            sub_counter = subclass_votes[(screen, cluster, gene)]["UNCHARACTERIZED"]
            modal_subclass = sub_counter.most_common(1)[0][0] if sub_counter else None
            unchar_n += 1
            if modal_subclass == consensus_subclass:
                unchar_correct += 1

    category_score = category_correct / category_n if category_n else 0.0

    coherence_correct = 0
    coherence_n = 0
    if cluster_coherence:
        for cluster_key, confidence_votes in cluster_confidence_votes.items():
            if cluster_key not in cluster_coherence:
                continue
            modal_confidence = confidence_votes.most_common(1)[0][0]
            coherence_n += 1
            if modal_confidence == cluster_coherence[cluster_key]:
                coherence_correct += 1

    return MetricPanel(
        category=category_score,
        novel_subclass=(novel_correct, novel_n),
        unchar_subclass=(unchar_correct, unchar_n),
        coherence=(coherence_correct, coherence_n),
        n=category_n,
        failures=failures,
    )


@dataclass
class DecoyResult:
    screen: str
    cluster: str
    expectation: str  # "abstain" | "functional"
    reps: int
    failures: int
    modal_confidence: str | None
    passed: bool


def score_decoys(
    run_dir: Path,
    decoy_specs: dict[tuple, str],
    route_equals: str | None = None,
    route_includes: str | None = None,
    route_excludes: str = "mcp",
) -> list[DecoyResult]:
    """Validate negative-control decoy clusters (not consensus-scored).

    decoy_specs maps (screen, cluster) -> expectation:
      - "abstain"    (nonsense / control-heavy clusters): PASS when the model
        returns a valid answer whose modal pathway_confidence is "Low" -- it
        recognized there is no discernible cluster. A parse failure is a crash,
        not abstention, so it does NOT pass.
      - "functional" (large coherent cluster, e.g. jebel/0): PASS when every
        replicate returned a valid answer (no error / truncation) AND modal
        confidence is High or Medium -- it handled the big cluster without
        erroring out.

    Route filtering mirrors score_run (route_equals > route_includes >
    route_excludes).
    """
    run_dir = Path(run_dir)
    reps: dict[tuple, int] = defaultdict(int)
    failures: dict[tuple, int] = defaultdict(int)
    conf_votes: dict[tuple, Counter] = defaultdict(Counter)

    with open(run_dir / "parsed_outputs.jsonl") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cell = json.loads(line)
            route = cell.get("route", "")
            if route_equals is not None:
                if route != route_equals:
                    continue
            elif route_includes is not None:
                if route_includes not in route:
                    continue
            elif route_excludes and route_excludes in route:
                continue

            screen, cluster = _parse_run_id(cell["run_id"], route)
            key = (screen, cluster)
            if key not in decoy_specs:
                continue
            reps[key] += 1
            parsed = cell.get("parsed_output")
            if parsed is None or parsed.get("dominant_process") is None:
                failures[key] += 1
                continue
            confidence = parsed.get("pathway_confidence")
            if confidence:
                conf_votes[key][confidence] += 1

    results = []
    for key, expectation in decoy_specs.items():
        screen, cluster = key
        votes = conf_votes[key]
        modal = votes.most_common(1)[0][0] if votes else None
        n_reps = reps[key]
        n_fail = failures[key]
        if expectation == "abstain":
            passed = modal == "Low"
        elif expectation == "functional":
            passed = n_reps > 0 and n_fail == 0 and modal in ("High", "Medium")
        else:
            raise ValueError(f"unknown decoy expectation {expectation!r} for {key}")
        results.append(
            DecoyResult(screen, cluster, expectation, n_reps, n_fail, modal, passed)
        )
    return results
