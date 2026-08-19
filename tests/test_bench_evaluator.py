"""Unit tests for bench_evaluator.py"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import pandas as pd
from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_evaluator import (  # noqa: E402
    _agg_rate,
    _consensus_subclass,
    _EVAL_OUTPUT_SUFFIXES,
    aggregate_per_route,
    build_consensus_gt,
    compute_coverage,
    compute_negative_abstention,
    compute_output_fragility,
    consensus_coherence,
    consensus_of,
    load_consensus_gt,
    load_predictions,
    NEGATIVE_CLUSTERS,
    OUTPUT_FRAGILITY_CLUSTER,
    pathway_loose_match,
    pathway_substring_match,
    PRED_COLS_REQUIRED,
)

REVIEWERS = ("eric", "iain", "liz")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

P = Path(__file__).resolve().parent / "phase1_prompt_benchmarking"
GT = P / "benchmark_inputs" / "ground_truth"


def _reviewers() -> dict[str, Path]:
    return {r: GT / f"annotation_{r}.csv" for r in ("eric", "liz", "iain")}


def _gt(tmp_path):
    out = tmp_path / "gt.csv"
    build_consensus_gt(_reviewers(), GT / "survey_key.csv", [], out)
    return load_consensus_gt(out)


# ===========================================================================
# Utility functions
# ===========================================================================


class TestUtilityFunctions:
    def test_consensus_all_agree(self):
        assert consensus_of(["A", "A", "A"]) == ("A", True, 3)

    def test_consensus_majority(self):
        # >=2-of-3 resolves even when one reviewer differs.
        assert consensus_of(["A", "B", "A"]) == ("A", False, 2)

    def test_consensus_disagreement(self):
        # A 1-1-1 three-way split is unresolved ("").
        assert consensus_of(["A", "B", "C"]) == ("", False, 1)

    def test_consensus_blank_ignored(self):
        assert consensus_of(["A", "", "A"]) == ("A", True, 2)

    def test_consensus_all_empty(self):
        assert consensus_of(["", "", ""]) == ("", False, 0)

    def test_pathway_substring_match_expert_in_pred(self):
        assert pathway_substring_match("DNA damage response pathway", ["DNA damage"]) is True

    def test_pathway_substring_match_pred_in_expert(self):
        assert pathway_substring_match("DNA damage", ["DNA damage response pathway"]) is False

    def test_pathway_loose_match_bidirectional(self):
        assert pathway_loose_match("DNA damage", ["DNA damage response pathway"]) is True

    def test_pathway_substring_empty_pred(self):
        assert pathway_substring_match("", ["anything"]) is False

    def test_pathway_loose_empty_experts(self):
        assert pathway_loose_match("something", ["", ""]) is False


# ===========================================================================
# 3.4 Aggregation
# ===========================================================================


class TestAggregation:
    def test_agg_rate_nullable_with_none(self):
        df = pd.DataFrame({"subclass_match": [True, None, False, None]})
        rate, n = _agg_rate(df, "subclass_match")
        assert n == 2
        assert rate == pytest.approx(0.5)

    def test_agg_rate_non_nullable(self):
        df = pd.DataFrame({"classification_match_either": [True, False, True]})
        rate, n = _agg_rate(df, "classification_match_either")
        assert n == 3
        assert rate == pytest.approx(0.667)


# ===========================================================================
# 3.5 Loading -- use tmp_path
# ===========================================================================


class TestLoading:
    def _write_pred_csv(self, path: Path, extra_rows: int = 3):
        """Write a valid prediction CSV."""
        records = []
        for i in range(extra_rows):
            records.append(
                {
                    "screen_name": "screen1",
                    "cluster_id": f"c{i}",
                    "gene_symbol": f"gene{i}",
                    "route": "route_A",
                    "replicate": "rep1",
                    "run_id": "run1",
                    "predicted_class": "ESTABLISHED",
                    "predicted_subclass": "",
                    "pathway": "some pathway",
                    "pathway_confidence": "High",
                }
            )
        pd.DataFrame(records).to_csv(path, index=False)

    def test_load_predictions_excludes_eval_files(self, tmp_path):
        self._write_pred_csv(tmp_path / "predictions_route_A.csv")
        # Write eval files that should be excluded
        pd.DataFrame({"x": [1]}).to_csv(tmp_path / "eval_per_gene.csv", index=False)
        pd.DataFrame({"x": [1]}).to_csv(tmp_path / "arch_eval_per_route.csv", index=False)
        pd.DataFrame({"x": [1]}).to_csv(tmp_path / "aggregate_summary.csv", index=False)

        result = load_predictions(tmp_path)
        assert len(result) == 3
        assert list(result.columns) == list(PRED_COLS_REQUIRED)

    def test_load_predictions_missing_columns_raises(self, tmp_path):
        df = pd.DataFrame(
            {
                "screen_name": ["s1"],
                "cluster_id": ["c1"],
                # gene_symbol is missing
                "route": ["r1"],
                "replicate": ["rep1"],
                "run_id": ["run1"],
                "predicted_class": ["ESTABLISHED"],
                "predicted_subclass": [""],
                "pathway": ["p"],
                "pathway_confidence": ["High"],
            }
        )
        df.to_csv(tmp_path / "bad.csv", index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_predictions(tmp_path)


# ===========================================================================
# Helpers for new-addition tests
# ===========================================================================


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in records),
        encoding="utf-8",
    )


def _clusters_csv(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "benchmark_clusters.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


# ===========================================================================
# 3.6 compute_negative_abstention
# ===========================================================================


class TestComputeNegativeAbstention:
    """Tests derived from docstring:

    Per-route abstention rate on the shuffled negative-control clusters.
    Reads parsed_outputs.jsonl directly so cells with empty gene arrays (correct
    abstention) are counted. Abstain iff `dominant_process` contains "no coherent"
    AND `pathway_confidence` is Low AND all three gene-classification arrays are
    empty.
    """

    def test_returns_empty_when_no_file(self, tmp_path):
        result = compute_negative_abstention(tmp_path)
        assert result.empty

    def test_correct_abstention_detected(self, tmp_path):
        """A cell that meets all three abstention criteria should be scored as abstain=True."""
        screen, cluster, case_type = NEGATIVE_CLUSTERS[0]
        record = {
            "run_id": f"exp__route_A__{screen}__cluster_{cluster}__rep_1",
            "route": "route_A",
            "parsed_output": {
                "dominant_process": "no coherent pathway identified",
                "pathway_confidence": "Low",
                "established_genes": [],
                "novel_role_genes": [],
                "uncharacterized_genes": [],
            },
        }
        _write_jsonl(tmp_path / "parsed_outputs.jsonl", [record])
        result = compute_negative_abstention(tmp_path)
        assert len(result) == 1
        assert result.iloc[0]["abstain_rate"] == 1.0
        assert result.iloc[0]["fabrication_rate"] == 0.0

    def test_non_abstention_when_genes_present(self, tmp_path):
        """If gene arrays are non-empty, abstention condition fails even with correct text."""
        screen, cluster, case_type = NEGATIVE_CLUSTERS[0]
        record = {
            "run_id": f"exp__route_A__{screen}__cluster_{cluster}__rep_1",
            "route": "route_A",
            "parsed_output": {
                "dominant_process": "no coherent pathway identified",
                "pathway_confidence": "Low",
                "established_genes": ["TP53"],
                "novel_role_genes": [],
                "uncharacterized_genes": [],
            },
        }
        _write_jsonl(tmp_path / "parsed_outputs.jsonl", [record])
        result = compute_negative_abstention(tmp_path)
        assert result.iloc[0]["abstain_rate"] == 0.0
        assert result.iloc[0]["fabrication_rate"] == 1.0

    def test_non_abstention_when_confidence_not_low(self, tmp_path):
        """If confidence is not Low, abstention condition fails."""
        screen, cluster, case_type = NEGATIVE_CLUSTERS[0]
        record = {
            "run_id": f"exp__route_A__{screen}__cluster_{cluster}__rep_1",
            "route": "route_A",
            "parsed_output": {
                "dominant_process": "no coherent pathway identified",
                "pathway_confidence": "High",
                "established_genes": [],
                "novel_role_genes": [],
                "uncharacterized_genes": [],
            },
        }
        _write_jsonl(tmp_path / "parsed_outputs.jsonl", [record])
        result = compute_negative_abstention(tmp_path)
        assert result.iloc[0]["abstain_rate"] == 0.0

    def test_non_abstention_when_dominant_process_missing_keyword(self, tmp_path):
        """If dominant_process does not contain 'no coherent', it's not abstention."""
        screen, cluster, case_type = NEGATIVE_CLUSTERS[0]
        record = {
            "run_id": f"exp__route_A__{screen}__cluster_{cluster}__rep_1",
            "route": "route_A",
            "parsed_output": {
                "dominant_process": "cell cycle regulation",
                "pathway_confidence": "Low",
                "established_genes": [],
                "novel_role_genes": [],
                "uncharacterized_genes": [],
            },
        }
        _write_jsonl(tmp_path / "parsed_outputs.jsonl", [record])
        result = compute_negative_abstention(tmp_path)
        assert result.iloc[0]["abstain_rate"] == 0.0

    def test_ignores_non_negative_clusters(self, tmp_path):
        """Records for clusters not in NEGATIVE_CLUSTERS should be ignored."""
        record = {
            "run_id": "exp__route_A__aconcagua_interphase__cluster_29__rep_1",
            "route": "route_A",
            "parsed_output": {
                "dominant_process": "no coherent pathway",
                "pathway_confidence": "Low",
                "established_genes": [],
                "novel_role_genes": [],
                "uncharacterized_genes": [],
            },
        }
        _write_jsonl(tmp_path / "parsed_outputs.jsonl", [record])
        result = compute_negative_abstention(tmp_path)
        assert result.empty

    def test_multiple_routes_reported_separately(self, tmp_path):
        """Each route should get its own row in the output."""
        screen, cluster, case_type = NEGATIVE_CLUSTERS[0]
        records = [
            {
                "run_id": f"exp__route_A__{screen}__cluster_{cluster}__rep_1",
                "route": "route_A",
                "parsed_output": {
                    "dominant_process": "no coherent pathway",
                    "pathway_confidence": "Low",
                    "established_genes": [],
                    "novel_role_genes": [],
                    "uncharacterized_genes": [],
                },
            },
            {
                "run_id": f"exp__route_B__{screen}__cluster_{cluster}__rep_1",
                "route": "route_B",
                "parsed_output": {
                    "dominant_process": "cell cycle",
                    "pathway_confidence": "High",
                    "established_genes": ["TP53"],
                    "novel_role_genes": [],
                    "uncharacterized_genes": [],
                },
            },
        ]
        _write_jsonl(tmp_path / "parsed_outputs.jsonl", records)
        result = compute_negative_abstention(tmp_path)
        assert len(result) == 2
        route_a = result[result["route"] == "route_A"].iloc[0]
        route_b = result[result["route"] == "route_B"].iloc[0]
        assert route_a["abstain_rate"] == 1.0
        assert route_b["abstain_rate"] == 0.0


# ===========================================================================
# 3.7 compute_coverage
# ===========================================================================


class TestComputeCoverage:
    """Tests derived from docstring:

    Per-route coverage on unique gene symbols.
    Reports three rates so the "coverage drop" signal isn't confounded by the
    shuffled-controls case where lower coverage is correct:
    - coverage_rate_all: across every cluster in the benchmark
    - coverage_rate_real: excludes the shuffled negative-control clusters
    - coverage_rate_categorized: also excludes clusters with benchmark_case_type=NaN
    """

    def _make_preds(self, rows: list[dict]) -> pd.DataFrame:
        base = {
            "screen_name": "screen1",
            "cluster_id": "c1",
            "gene_symbol": "gene1",
            "route": "route_A",
            "replicate": 1,
            "run_id": "run1",
            "predicted_class": "ESTABLISHED",
            "predicted_subclass": "",
            "pathway": "some pathway",
            "pathway_confidence": "High",
        }
        records = [{**base, **r} for r in rows]
        return pd.DataFrame(records)

    def test_returns_empty_when_clusters_file_missing(self, tmp_path):
        preds = self._make_preds([{"gene_symbol": "A"}])
        result = compute_coverage(preds, tmp_path / "nonexistent.csv")
        assert result.empty

    def test_returns_empty_when_clusters_missing_required_columns(self, tmp_path):
        p = tmp_path / "clusters.csv"
        pd.DataFrame({"bad_col": ["x"]}).to_csv(p, index=False)
        preds = self._make_preds([{"gene_symbol": "A"}])
        result = compute_coverage(preds, p)
        assert result.empty

    def test_full_coverage_all_genes_predicted(self, tmp_path):
        """If predictions cover every gene in the clusters file, rate should be 1.0."""
        clusters_path = _clusters_csv(
            tmp_path,
            [
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gA"},
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gB"},
            ],
        )
        preds = self._make_preds(
            [
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gA", "replicate": 1},
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gB", "replicate": 1},
            ]
        )
        result = compute_coverage(preds, clusters_path)
        assert len(result) == 1
        assert result.iloc[0]["coverage_rate_all"] == 1.0

    def test_partial_coverage(self, tmp_path):
        """If only some genes are predicted, rate should reflect the fraction."""
        clusters_path = _clusters_csv(
            tmp_path,
            [
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gA"},
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gB"},
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gC"},
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gD"},
            ],
        )
        preds = self._make_preds(
            [
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gA", "replicate": 1},
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gB", "replicate": 1},
            ]
        )
        result = compute_coverage(preds, clusters_path)
        assert result.iloc[0]["coverage_rate_all"] == 0.5

    def test_shuffled_excluded_from_real_rate(self, tmp_path):
        """coverage_rate_real should exclude NEGATIVE_CLUSTERS clusters."""
        screen_neg, cluster_neg, _ = NEGATIVE_CLUSTERS[0]
        clusters_path = _clusters_csv(
            tmp_path,
            [
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gA"},
                {"screen_name": screen_neg, "cluster_id": cluster_neg, "gene_symbol": "gX"},
            ],
        )
        preds = self._make_preds(
            [
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gA", "replicate": 1},
            ]
        )
        result = compute_coverage(preds, clusters_path)
        assert result.iloc[0]["coverage_rate_all"] == 0.5
        assert result.iloc[0]["coverage_rate_real"] == 1.0

    def test_uncategorized_excluded_from_categorized_rate(self, tmp_path):
        """coverage_rate_categorized should also exclude clusters with NaN benchmark_case_type."""
        clusters_path = _clusters_csv(
            tmp_path,
            [
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gA"},
                {"screen_name": "s1", "cluster_id": "c2", "gene_symbol": "gB"},
            ],
        )
        gt = pd.DataFrame(
            [
                {
                    "screen_name": "s1",
                    "cluster_id": "c1",
                    "gene_symbol": "gA",
                    "benchmark_case_type": "type_A",
                },
                {
                    "screen_name": "s1",
                    "cluster_id": "c2",
                    "gene_symbol": "gB",
                    "benchmark_case_type": None,
                },
            ]
        )
        preds = self._make_preds(
            [
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gA", "replicate": 1},
            ]
        )
        result = compute_coverage(preds, clusters_path, ground_truth=gt)
        assert result.iloc[0]["coverage_rate_all"] == 0.5
        assert result.iloc[0]["coverage_rate_categorized"] == 1.0

    def test_renames_sheet_to_screen_name(self, tmp_path):
        """Clusters CSV with 'sheet' column should be auto-renamed to 'screen_name'."""
        p = tmp_path / "clusters.csv"
        pd.DataFrame(
            [
                {"sheet": "s1", "cluster_id": "c1", "gene_symbol": "gA"},
            ]
        ).to_csv(p, index=False)
        preds = self._make_preds(
            [
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gA", "replicate": 1},
            ]
        )
        result = compute_coverage(preds, p)
        assert not result.empty
        assert result.iloc[0]["coverage_rate_all"] == 1.0

    def test_multiple_routes_reported(self, tmp_path):
        """Each route should produce its own coverage row."""
        clusters_path = _clusters_csv(
            tmp_path,
            [
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gA"},
            ],
        )
        preds = pd.DataFrame(
            [
                {
                    "screen_name": "s1",
                    "cluster_id": "c1",
                    "gene_symbol": "gA",
                    "route": "route_A",
                    "replicate": 1,
                    "run_id": "r1",
                    "predicted_class": "E",
                    "predicted_subclass": "",
                    "pathway": "p",
                    "pathway_confidence": "H",
                },
                {
                    "screen_name": "s1",
                    "cluster_id": "c1",
                    "gene_symbol": "gA",
                    "route": "route_B",
                    "replicate": 1,
                    "run_id": "r2",
                    "predicted_class": "E",
                    "predicted_subclass": "",
                    "pathway": "p",
                    "pathway_confidence": "H",
                },
            ]
        )
        result = compute_coverage(preds, clusters_path)
        assert len(result) == 2
        assert set(result["route"]) == {"route_A", "route_B"}


# ===========================================================================
# 3.8 compute_output_fragility
# ===========================================================================


class TestComputeOutputFragility:
    """Tests derived from docstring:

    Per-route diagnostic on the output_fragility cluster (jebel/0, 147 genes).
    No expert annotations exist for this cluster, so it is not scored on accuracy
    or abstention. Reports per-route coverage and pathway-consistency to surface
    output-structure failures on a large coherent input.
    """

    def _make_preds(self, rows: list[dict]) -> pd.DataFrame:
        screen, cid = OUTPUT_FRAGILITY_CLUSTER
        base = {
            "screen_name": screen,
            "cluster_id": cid,
            "gene_symbol": "gene1",
            "route": "route_A",
            "replicate": 1,
            "run_id": "run1",
            "predicted_class": "ESTABLISHED",
            "predicted_subclass": "",
            "pathway": "ribosome biogenesis",
            "pathway_confidence": "High",
        }
        records = [{**base, **r} for r in rows]
        return pd.DataFrame(records)

    def test_returns_empty_when_cluster_absent(self, tmp_path):
        """If the fragility cluster isn't in predictions, return empty."""
        preds = pd.DataFrame(
            [
                {
                    "screen_name": "other_screen",
                    "cluster_id": "99",
                    "gene_symbol": "gA",
                    "route": "route_A",
                    "replicate": 1,
                    "run_id": "r1",
                    "predicted_class": "E",
                    "predicted_subclass": "",
                    "pathway": "p",
                    "pathway_confidence": "H",
                }
            ]
        )
        result = compute_output_fragility(preds, tmp_path / "clusters.csv")
        assert result.empty

    def test_coverage_computed_against_clusters_file(self, tmp_path):
        """Coverage should be predictions / expected genes from the clusters CSV."""
        screen, cid = OUTPUT_FRAGILITY_CLUSTER
        clusters_path = _clusters_csv(
            tmp_path,
            [{"screen_name": screen, "cluster_id": cid, "gene_symbol": f"g{i}"} for i in range(10)],
        )
        preds = self._make_preds([{"gene_symbol": f"g{i}", "replicate": 1} for i in range(5)])
        result = compute_output_fragility(preds, clusters_path)
        assert len(result) == 1
        assert result.iloc[0]["coverage_rate"] == 0.5
        assert result.iloc[0]["n_expected_per_cell"] == 10

    def test_pathway_consistency_ribosome(self, tmp_path):
        """Pathways mentioning 'ribosom' or 'translation' should count as consistent."""
        screen, cid = OUTPUT_FRAGILITY_CLUSTER
        clusters_path = _clusters_csv(
            tmp_path,
            [{"screen_name": screen, "cluster_id": cid, "gene_symbol": f"g{i}"} for i in range(4)],
        )
        preds = self._make_preds(
            [
                {"gene_symbol": "g0", "replicate": 1, "pathway": "Ribosome biogenesis"},
                {"gene_symbol": "g1", "replicate": 2, "pathway": "mRNA translation"},
                {"gene_symbol": "g2", "replicate": 3, "pathway": "cell cycle"},
            ]
        )
        result = compute_output_fragility(preds, clusters_path)
        assert result.iloc[0]["pathway_consistency_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_pathway_consistency_all_unrelated(self, tmp_path):
        """If no replicate mentions ribosome/translation, consistency should be 0."""
        screen, cid = OUTPUT_FRAGILITY_CLUSTER
        clusters_path = _clusters_csv(
            tmp_path,
            [
                {"screen_name": screen, "cluster_id": cid, "gene_symbol": "g0"},
            ],
        )
        preds = self._make_preds(
            [
                {"gene_symbol": "g0", "replicate": 1, "pathway": "cell cycle regulation"},
                {"gene_symbol": "g0", "replicate": 2, "pathway": "DNA damage response"},
            ]
        )
        result = compute_output_fragility(preds, clusters_path)
        assert result.iloc[0]["pathway_consistency_rate"] == 0.0

    def test_multiple_routes(self, tmp_path):
        """Each route should get a separate diagnostic row."""
        screen, cid = OUTPUT_FRAGILITY_CLUSTER
        clusters_path = _clusters_csv(
            tmp_path,
            [
                {"screen_name": screen, "cluster_id": cid, "gene_symbol": "g0"},
            ],
        )
        preds = self._make_preds(
            [
                {"gene_symbol": "g0", "replicate": 1, "route": "route_A", "pathway": "ribosome"},
                {"gene_symbol": "g0", "replicate": 1, "route": "route_B", "pathway": "cell cycle"},
            ]
        )
        result = compute_output_fragility(preds, clusters_path)
        assert len(result) == 2
        route_a = result[result["route"] == "route_A"].iloc[0]
        route_b = result[result["route"] == "route_B"].iloc[0]
        assert route_a["pathway_consistency_rate"] == 1.0
        assert route_b["pathway_consistency_rate"] == 0.0


# ===========================================================================
# 3.9 aggregate_per_route -- pathway_semantic_match_loose_rate
# ===========================================================================


class TestAggregatePerRouteLooseSemanticMatch:
    """Tests derived from docstring of compute_semantic_scores + aggregate_per_route:

    compute_semantic_scores adds pathway_semantic_match_loose: score >= 0.60 (hardcoded).
    aggregate_per_route should expose pathway_semantic_match_loose_rate when the
    column is present.
    """

    def _make_joined(self, rows: list[dict]) -> pd.DataFrame:
        base = {
            "screen_name": "screen1",
            "cluster_id": "c1",
            "gene_symbol": "geneA",
            "route": "route_1",
            "replicate": "rep1",
            "run_id": "run1",
            "predicted_class": "ESTABLISHED",
            "predicted_subclass": "",
            "pathway": "some pathway",
            "pathway_confidence": "High",
            "benchmark_case_type": "type_A",
            # Match metrics expected by _route_metric_cols
            "classification_match_consensus": True,
            "classification_match_either": True,
            "pathway_match_substring": True,
            "pathway_match_loose": True,
            "confidence_consensus_match": True,
            "subclass_match": True,
            # Semantic columns
            "pathway_semantic_score": 0.85,
            "pathway_semantic_match": True,
            "pathway_semantic_match_loose": True,
        }
        for r in REVIEWERS:
            base[f"classification_match_{r}"] = True
        records = [{**base, **r} for r in rows]
        return pd.DataFrame(records)

    def test_loose_rate_included_when_column_present(self):
        """When pathway_semantic_match_loose column exists, the rate should appear."""
        df = self._make_joined(
            [
                {"gene_symbol": "g1", "pathway_semantic_match_loose": True},
                {"gene_symbol": "g2", "pathway_semantic_match_loose": False},
            ]
        )
        result = aggregate_per_route(df, REVIEWERS)
        assert "pathway_semantic_match_loose_rate" in result.columns
        assert result.iloc[0]["pathway_semantic_match_loose_rate"] == 0.5

    def test_loose_rate_absent_when_column_missing(self):
        """When pathway_semantic_match_loose column is absent, no loose_rate column."""
        df = self._make_joined(
            [
                {"gene_symbol": "g1"},
                {"gene_symbol": "g2"},
            ]
        )
        df = df.drop(columns=["pathway_semantic_match_loose"])
        result = aggregate_per_route(df, REVIEWERS)
        assert "pathway_semantic_match_loose_rate" not in result.columns

    def test_loose_rate_is_mean_of_boolean_column(self):
        """Rate should be the arithmetic mean of the boolean loose-match column."""
        df = self._make_joined(
            [{"gene_symbol": f"g{i}", "pathway_semantic_match_loose": (i < 3)} for i in range(5)]
        )
        result = aggregate_per_route(df, REVIEWERS)
        assert result.iloc[0]["pathway_semantic_match_loose_rate"] == 0.6


# ===========================================================================
# 3.10 Multi-replicate stress tests
# ===========================================================================


class TestCoverageMultiReplicate:
    """Stress tests for compute_coverage with multiple replicates.

    The docstring says "Per-route coverage on unique gene symbols." The
    implementation counts unique genes per (screen, cluster, replicate) cell,
    then sums across all cells vs expected*n_replicates. These tests verify
    that multi-replicate scenarios produce coherent numbers.
    """

    def _make_preds(self, rows: list[dict]) -> pd.DataFrame:
        base = {
            "screen_name": "s1",
            "cluster_id": "c1",
            "gene_symbol": "gene1",
            "route": "route_A",
            "replicate": 1,
            "run_id": "run1",
            "predicted_class": "ESTABLISHED",
            "predicted_subclass": "",
            "pathway": "some pathway",
            "pathway_confidence": "High",
        }
        records = [{**base, **r} for r in rows]
        return pd.DataFrame(records)

    def test_full_coverage_across_two_replicates(self, tmp_path):
        """All genes predicted in both replicates -> 1.0 coverage."""
        clusters_path = _clusters_csv(
            tmp_path,
            [
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gA"},
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gB"},
            ],
        )
        preds = self._make_preds(
            [
                {"gene_symbol": "gA", "replicate": 1},
                {"gene_symbol": "gB", "replicate": 1},
                {"gene_symbol": "gA", "replicate": 2},
                {"gene_symbol": "gB", "replicate": 2},
            ]
        )
        result = compute_coverage(preds, clusters_path)
        assert result.iloc[0]["coverage_rate_all"] == 1.0

    def test_one_replicate_missing_all_genes(self, tmp_path):
        """If predictions only contain rep 1, the function can only see rep 1.
        Coverage is computed over replicates present in the data, not over some
        externally-known replicate count."""
        clusters_path = _clusters_csv(
            tmp_path,
            [
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gA"},
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gB"},
            ],
        )
        preds = self._make_preds(
            [
                {"gene_symbol": "gA", "replicate": 1},
                {"gene_symbol": "gB", "replicate": 1},
                # no replicate 2 in predictions at all
            ]
        )
        result = compute_coverage(preds, clusters_path)
        # Only 1 replicate visible -> grid is just [1], coverage = 2/2 = 1.0
        assert result.iloc[0]["coverage_rate_all"] == 1.0
        assert result.iloc[0]["n_replicates"] == 1

    def test_partial_genes_each_replicate(self, tmp_path):
        """Different genes predicted in each replicate."""
        clusters_path = _clusters_csv(
            tmp_path,
            [
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gA"},
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gB"},
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gC"},
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gD"},
            ],
        )
        preds = self._make_preds(
            [
                {"gene_symbol": "gA", "replicate": 1},
                {"gene_symbol": "gB", "replicate": 1},
                # rep 2 predicts different genes
                {"gene_symbol": "gC", "replicate": 2},
            ]
        )
        result = compute_coverage(preds, clusters_path)
        # expected: (2 + 1) / (4 + 4) = 3/8 = 0.375
        assert result.iloc[0]["coverage_rate_all"] == 0.375

    def test_duplicate_gene_in_same_replicate_counted_once(self, tmp_path):
        """Same gene predicted twice in one replicate should count as 1 unique."""
        clusters_path = _clusters_csv(
            tmp_path,
            [
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gA"},
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gB"},
            ],
        )
        preds = self._make_preds(
            [
                {"gene_symbol": "gA", "replicate": 1},
                {"gene_symbol": "gA", "replicate": 1},  # duplicate
                {"gene_symbol": "gB", "replicate": 1},
            ]
        )
        result = compute_coverage(preds, clusters_path)
        # 2 unique genes / 2 expected = 1.0 (not 1.5)
        assert result.iloc[0]["coverage_rate_all"] == 1.0

    def test_non_contiguous_replicate_ids(self, tmp_path):
        """Replicates numbered [1, 3] (gap at 2) -- predictions from rep 3 must not be dropped."""
        clusters_path = _clusters_csv(
            tmp_path,
            [
                {"screen_name": "s1", "cluster_id": "c1", "gene_symbol": "gA"},
            ],
        )
        preds = self._make_preds(
            [
                {"gene_symbol": "gA", "replicate": 1},
                {"gene_symbol": "gA", "replicate": 3},
            ]
        )
        result = compute_coverage(preds, clusters_path)
        assert result.iloc[0]["n_replicates"] == 2
        # Both replicates predicted gA, so coverage should be 1.0 (not 0.5).
        assert result.iloc[0]["coverage_rate_all"] == 1.0


class TestOutputFragilityMultiGenePerReplicate:
    """Stress test: multiple genes per replicate and pathway consistency."""

    def test_consistency_uses_first_row_per_replicate(self, tmp_path):
        """With multiple genes in one replicate, drop_duplicates('replicate') keeps
        only the first row's pathway. This test documents that behavior."""
        screen, cid = OUTPUT_FRAGILITY_CLUSTER
        clusters_path = _clusters_csv(
            tmp_path,
            [{"screen_name": screen, "cluster_id": cid, "gene_symbol": f"g{i}"} for i in range(10)],
        )
        preds = pd.DataFrame(
            [
                {
                    "screen_name": screen,
                    "cluster_id": cid,
                    "gene_symbol": "g0",
                    "route": "route_A",
                    "replicate": 1,
                    "run_id": "r1",
                    "predicted_class": "E",
                    "predicted_subclass": "",
                    "pathway": "ribosome biogenesis",
                    "pathway_confidence": "H",
                },
                {
                    "screen_name": screen,
                    "cluster_id": cid,
                    "gene_symbol": "g1",
                    "route": "route_A",
                    "replicate": 1,
                    "run_id": "r1",
                    "predicted_class": "E",
                    "predicted_subclass": "",
                    "pathway": "cell cycle",
                    "pathway_confidence": "H",
                },
                {
                    "screen_name": screen,
                    "cluster_id": cid,
                    "gene_symbol": "g2",
                    "route": "route_A",
                    "replicate": 2,
                    "run_id": "r2",
                    "predicted_class": "E",
                    "predicted_subclass": "",
                    "pathway": "DNA damage",
                    "pathway_confidence": "H",
                },
            ]
        )
        result = compute_output_fragility(preds, clusters_path)
        # Rep 1 has "ribosome biogenesis" as first row -> consistent
        # Rep 2 has "DNA damage" -> not consistent
        # Consistency = 1/2 = 0.5
        # NOTE: rep 1 also has "cell cycle" for g1, but that's ignored because
        # drop_duplicates keeps only the first row. This is questionable behavior
        # (the cluster-level pathway should be the same for all genes in a
        # replicate, but if it isn't, only the first is checked).
        assert result.iloc[0]["pathway_consistency_rate"] == 0.5


# ===========================================================================
# Consensus ground truth (from raw per-reviewer annotations)
# ===========================================================================


def test_consensus_matches_known(tmp_path):
    gt = _gt(tmp_path)
    real = {k: v for k, v in gt.items() if v["cluster_role"] == "real"}
    assert len(real) == 133
    # >=2-of-3 majority resolves every gene (no three-way splits on this dataset).
    assert sum(1 for v in real.values() if not v["consensus_class"]) == 0
    assert sum(1 for v in real.values() if str(v["unanimous"]) in ("True", "1")) == 98
    # De-blinded reviewer source preference over the real genes.
    pref = {
        s: sum(int(v[f"pref_{s}"]) for v in real.values())
        for s in ("affinage", "uniprot", "both", "neither")
    }
    assert pref == {"affinage": 156, "uniprot": 96, "both": 135, "neither": 12}


def test_consensus_subclass_ordinal_median():
    novel = "NOVEL_ROLE"
    # All-different resolves to the true ordinal middle (not insertion order).
    assert _consensus_subclass(["NO_EVIDENCE", "PARTIAL_EVIDENCE", "INDIRECT_EVIDENCE"], novel) == (
        "INDIRECT_EVIDENCE"
    )
    # A >=2 majority is respected by the median.
    assert _consensus_subclass(["NO_EVIDENCE", "PARTIAL_EVIDENCE", "PARTIAL_EVIDENCE"], novel) == (
        "PARTIAL_EVIDENCE"
    )
    # CONTRADICTORY sits at the bottom of the ladder (most skeptical), so a
    # scattered NO/PARTIAL/CONTRADICTORY vote resolves to the conservative middle.
    assert _consensus_subclass(["NO_EVIDENCE", "PARTIAL_EVIDENCE", "CONTRADICTORY_EVIDENCE"], novel) == (
        "NO_EVIDENCE"
    )
    # A CONTRADICTORY majority is respected, not dropped.
    assert _consensus_subclass(
        ["CONTRADICTORY_EVIDENCE", "CONTRADICTORY_EVIDENCE", "NO_EVIDENCE"], novel
    ) == "CONTRADICTORY_EVIDENCE"
    # Off-ladder votes (an UNCHARACTERIZED subclass on a NOVEL_ROLE gene) are dropped.
    assert _consensus_subclass(["NO_EVIDENCE", "INDIRECT_EVIDENCE", "ANNOTATED_ONLY"], novel) == (
        "NO_EVIDENCE"
    )
    # UNCHARACTERIZED ladder; the lone off-ladder NOVEL vote is dropped.
    assert _consensus_subclass(
        ["NO_EVIDENCE", "ANNOTATED_ONLY", "ANNOTATED_ONLY"], "UNCHARACTERIZED"
    ) == "ANNOTATED_ONLY"
    # ESTABLISHED carries no subclass.
    assert _consensus_subclass(["NO_EVIDENCE"], "ESTABLISHED") == ""


def test_consensus_subclass_on_real_examples(tmp_path):
    gt = _gt(tmp_path)
    by_gene = {k[2]: v for k, v in gt.items() if v["cluster_role"] == "real"}
    expected = {
        "CHP1": "INDIRECT_EVIDENCE",
        "ATL2": "PARTIAL_EVIDENCE",
        "CSDE1": "NO_EVIDENCE",
        "TCEAL4": "ANNOTATED_ONLY",
    }
    for gene, sub in expected.items():
        assert by_gene[gene]["consensus_subclass"] == sub, gene


def test_consensus_coherence_majority_and_ties():
    coh = consensus_coherence(_reviewers())
    # Four clusters have a High/Medium/Low majority; whitney/9 is a 3-way tie
    # (high/low/medium) and is therefore excluded.
    assert coh == {
        ("aconcagua_interphase", "41"): "Medium",
        ("denali", "24"): "Low",
        ("denali", "43"): "Medium",
        ("whitney", "6"): "Medium",
    }


def test_decoy_rows(tmp_path):
    out = tmp_path / "gt.csv"
    build_consensus_gt(
        _reviewers(),
        GT / "survey_key.csv",
        [{"screen": "x", "cluster": "1", "decoy_type": "nonsense", "genes": ["G1", "G2"]}],
        out,
    )
    gt = load_consensus_gt(out)
    decoys = {k: v for k, v in gt.items() if v["cluster_role"] == "decoy"}
    assert len(decoys) == 2
