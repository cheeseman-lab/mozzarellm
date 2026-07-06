"""Unit tests for bench_evaluator.py — 35 tests across 5 test classes."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import pandas as pd
from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_evaluator import (
    consensus_of,
    majority_confidence,
    pathway_substring_match,
    pathway_loose_match,
    _detect_phase_prefix,
    annotate_matches,
    _agg_rate,
    aggregate_per_route,
    aggregate_per_route_per_case_type,
    inter_reviewer_concordance,
    load_predictions,
    load_ground_truth,
    NULLABLE_METRICS,
    PRED_COLS_REQUIRED,
    _EVAL_OUTPUT_SUFFIXES,
)

REVIEWERS = ("eric", "iain", "liz")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_joined_df():
    """Build a minimal joined DataFrame suitable for annotate_matches.

    Returns a single-row frame; tests can modify/extend rows as needed.
    """

    def _make(rows: list[dict]) -> pd.DataFrame:
        records = []
        for overrides in rows:
            row = {
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
            }
            for r in REVIEWERS:
                row[f"expert_classification_{r}"] = "ESTABLISHED"
                row[f"nominated_pathway_{r}"] = "some pathway"
                row[f"pathway_confidence_{r}"] = "High"
                row[f"expert_subclass_{r}"] = ""
            row.update(overrides)
            records.append(row)
        return pd.DataFrame(records)

    return _make


# ===========================================================================
# 3.1 Utility functions
# ===========================================================================


class TestUtilityFunctions:
    def test_consensus_all_agree(self):
        assert consensus_of(["A", "A", "A"]) == "A"

    def test_consensus_disagreement(self):
        assert consensus_of(["A", "B", "A"]) == ""

    def test_consensus_blank_ignored(self):
        assert consensus_of(["A", "", "A"]) == "A"

    def test_consensus_all_empty(self):
        assert consensus_of(["", "", ""]) == ""

    def test_majority_confidence_clear(self):
        row = pd.Series(
            {
                "pathway_confidence_eric": "High",
                "pathway_confidence_iain": "High",
                "pathway_confidence_liz": "Low",
            }
        )
        assert majority_confidence(row, REVIEWERS) == "High"

    def test_majority_confidence_three_way_tie(self):
        row = pd.Series(
            {
                "pathway_confidence_eric": "High",
                "pathway_confidence_iain": "Medium",
                "pathway_confidence_liz": "Low",
            }
        )
        assert majority_confidence(row, REVIEWERS) == "High"

    def test_majority_confidence_two_way_tie(self):
        row = pd.Series(
            {
                "pathway_confidence_eric": "Medium",
                "pathway_confidence_iain": "Low",
                "pathway_confidence_liz": "",
            }
        )
        assert majority_confidence(row, REVIEWERS) == "Medium"

    def test_majority_confidence_all_empty(self):
        row = pd.Series(
            {
                "pathway_confidence_eric": "",
                "pathway_confidence_iain": "",
                "pathway_confidence_liz": "",
            }
        )
        assert majority_confidence(row, REVIEWERS) == ""

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

    def test_detect_phase_prefix(self):
        assert _detect_phase_prefix(Path("/project/1.arch/experiment_id")) == "arch"
        assert _detect_phase_prefix(Path("/project/2.order/experiment_id")) == "ord"
        assert _detect_phase_prefix(Path("/project/3.wording/experiment_id")) == "word"
        assert _detect_phase_prefix(Path("/project/4.comp/experiment_id")) == "comp"
        assert _detect_phase_prefix(Path("/project/other/experiment_id")) == ""


# ===========================================================================
# 3.2 Match annotation
# ===========================================================================


class TestAnnotateMatches:
    def test_all_agree_pred_matches(self, base_joined_df):
        df = base_joined_df(
            [
                {
                    "predicted_class": "ESTABLISHED",
                    "expert_classification_eric": "ESTABLISHED",
                    "expert_classification_iain": "ESTABLISHED",
                    "expert_classification_liz": "ESTABLISHED",
                }
            ]
        )
        result = annotate_matches(df, REVIEWERS)
        row = result.iloc[0]
        assert bool(row["classification_match_consensus"]) is True
        assert bool(row["classification_match_either"]) is True
        assert bool(row["experts_agree"]) is True

    def test_split_2_1_pred_matches_majority(self, base_joined_df):
        df = base_joined_df(
            [
                {
                    "predicted_class": "ESTABLISHED",
                    "expert_classification_eric": "ESTABLISHED",
                    "expert_classification_iain": "ESTABLISHED",
                    "expert_classification_liz": "NOVEL_ROLE",
                }
            ]
        )
        result = annotate_matches(df, REVIEWERS)
        row = result.iloc[0]
        assert bool(row["classification_match_either"]) is True
        assert row["consensus_class"] == ""
        assert row["classification_match_consensus"] is None

    def test_prediction_wrong(self, base_joined_df):
        df = base_joined_df(
            [
                {
                    "predicted_class": "NOVEL_ROLE",
                    "expert_classification_eric": "ESTABLISHED",
                    "expert_classification_iain": "ESTABLISHED",
                    "expert_classification_liz": "ESTABLISHED",
                }
            ]
        )
        result = annotate_matches(df, REVIEWERS)
        row = result.iloc[0]
        assert bool(row["classification_match_either"]) is False
        assert bool(row["classification_match_consensus"]) is False

    def test_prediction_empty(self, base_joined_df):
        df = base_joined_df(
            [
                {
                    "predicted_class": "",
                    "expert_classification_eric": "ESTABLISHED",
                    "expert_classification_iain": "ESTABLISHED",
                    "expert_classification_liz": "ESTABLISHED",
                }
            ]
        )
        result = annotate_matches(df, REVIEWERS)
        row = result.iloc[0]
        assert bool(row["classification_match_either"]) is False

    def test_expert_fields_all_blank(self, base_joined_df):
        df = base_joined_df(
            [
                {
                    "predicted_class": "ESTABLISHED",
                    "expert_classification_eric": "",
                    "expert_classification_iain": "",
                    "expert_classification_liz": "",
                }
            ]
        )
        result = annotate_matches(df, REVIEWERS)
        row = result.iloc[0]
        assert bool(row["classification_match_either"]) is False
        assert bool(row["experts_agree"]) is False

    def test_confidence_match(self, base_joined_df):
        df = base_joined_df(
            [
                {
                    "pathway_confidence": "High",
                    "pathway_confidence_eric": "High",
                    "pathway_confidence_iain": "High",
                    "pathway_confidence_liz": "Low",
                }
            ]
        )
        result = annotate_matches(df, REVIEWERS)
        row = result.iloc[0]
        assert bool(row["confidence_consensus_match"]) is True

    def test_confidence_mismatch(self, base_joined_df):
        df = base_joined_df(
            [
                {
                    "pathway_confidence": "Low",
                    "pathway_confidence_eric": "High",
                    "pathway_confidence_iain": "High",
                    "pathway_confidence_liz": "High",
                }
            ]
        )
        result = annotate_matches(df, REVIEWERS)
        row = result.iloc[0]
        assert bool(row["confidence_consensus_match"]) is False

    def test_subclass_match_novel_role(self, base_joined_df):
        df = base_joined_df(
            [
                {
                    "predicted_class": "NOVEL_ROLE",
                    "predicted_subclass": "NO_EVIDENCE",
                    "expert_classification_eric": "NOVEL_ROLE",
                    "expert_classification_iain": "NOVEL_ROLE",
                    "expert_classification_liz": "NOVEL_ROLE",
                    "expert_subclass_eric": "NO_EVIDENCE",
                    "expert_subclass_iain": "INDIRECT_EVIDENCE",
                    "expert_subclass_liz": "",
                }
            ]
        )
        result = annotate_matches(df, REVIEWERS)
        row = result.iloc[0]
        assert bool(row["subclass_match"]) is True

    def test_subclass_not_applicable(self, base_joined_df):
        df = base_joined_df(
            [
                {
                    "predicted_class": "ESTABLISHED",
                    "predicted_subclass": "NO_EVIDENCE",
                    "expert_classification_eric": "ESTABLISHED",
                    "expert_classification_iain": "ESTABLISHED",
                    "expert_classification_liz": "ESTABLISHED",
                    "expert_subclass_eric": "NO_EVIDENCE",
                    "expert_subclass_iain": "NO_EVIDENCE",
                    "expert_subclass_liz": "NO_EVIDENCE",
                }
            ]
        )
        result = annotate_matches(df, REVIEWERS)
        row = result.iloc[0]
        assert row["subclass_match"] is None

    def test_subclass_no_expert_labels(self, base_joined_df):
        df = base_joined_df(
            [
                {
                    "predicted_class": "NOVEL_ROLE",
                    "predicted_subclass": "NO_EVIDENCE",
                    "expert_classification_eric": "NOVEL_ROLE",
                    "expert_classification_iain": "NOVEL_ROLE",
                    "expert_classification_liz": "NOVEL_ROLE",
                    "expert_subclass_eric": "",
                    "expert_subclass_iain": "",
                    "expert_subclass_liz": "",
                }
            ]
        )
        result = annotate_matches(df, REVIEWERS)
        row = result.iloc[0]
        assert row["subclass_match"] is None


# ===========================================================================
# 3.3 Applicability / zero-score problem
# ===========================================================================


class TestApplicabilityBug:
    def test_all_blank_experts_produce_false_not_none(self, base_joined_df):
        """Current behavior: when all expert fields are blank, classification_match_either
        is False (not None). This documents the existing bug — after the applicability fix,
        these rows should become None (not scored) instead of False (wrong)."""
        rows = [
            {
                "gene_symbol": f"gene{i}",
                "predicted_class": "ESTABLISHED",
                "expert_classification_eric": "",
                "expert_classification_iain": "",
                "expert_classification_liz": "",
            }
            for i in range(4)
        ]
        df = base_joined_df(rows)
        result = annotate_matches(df, REVIEWERS)
        # Current buggy behavior: blank experts → False (counted as wrong)
        for val in result["classification_match_either"]:
            assert val is False

    def test_mixed_blank_dilutes_score(self, base_joined_df):
        """3 genes with labels + 2 blank. The blank rows are counted in the denominator,
        diluting the rate. This documents the current behavior."""
        labeled_rows = [
            {
                "gene_symbol": f"gene{i}",
                "predicted_class": "ESTABLISHED",
                "expert_classification_eric": "ESTABLISHED",
                "expert_classification_iain": "ESTABLISHED",
                "expert_classification_liz": "ESTABLISHED",
            }
            for i in range(3)
        ]
        blank_rows = [
            {
                "gene_symbol": f"gene{i + 3}",
                "predicted_class": "ESTABLISHED",
                "expert_classification_eric": "",
                "expert_classification_iain": "",
                "expert_classification_liz": "",
            }
            for i in range(2)
        ]
        df = base_joined_df(labeled_rows + blank_rows)
        result = annotate_matches(df, REVIEWERS)
        # Rate denominator includes blank rows: 3 correct / 5 total = 0.6
        rate = result["classification_match_either"].mean()
        assert rate == pytest.approx(0.6)


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

    def test_aggregate_per_route_two_routes(self, base_joined_df):
        rows = []
        for route in ("route_A", "route_B"):
            for i in range(3):
                rows.append(
                    {
                        "gene_symbol": f"gene{i}",
                        "route": route,
                        "predicted_class": "ESTABLISHED",
                        "expert_classification_eric": "ESTABLISHED",
                        "expert_classification_iain": "ESTABLISHED",
                        "expert_classification_liz": "ESTABLISHED",
                    }
                )
        df = base_joined_df(rows)
        annotated = annotate_matches(df, REVIEWERS)
        result = aggregate_per_route(annotated, REVIEWERS)
        assert len(result) == 2
        assert list(result["n_genes"]) == [3, 3]
        # Nullable metrics produce _rate and _n columns
        for m in NULLABLE_METRICS:
            assert f"{m}_rate" in result.columns
            assert f"{m}_n" in result.columns

    def test_aggregate_per_route_per_case_type(self, base_joined_df):
        rows = []
        idx = 0
        for route in ("route_A", "route_B"):
            for case_type in ("type_X", "type_Y"):
                for i in range(2):
                    rows.append(
                        {
                            "gene_symbol": f"gene{idx}",
                            "route": route,
                            "benchmark_case_type": case_type,
                            "predicted_class": "ESTABLISHED",
                            "expert_classification_eric": "ESTABLISHED",
                            "expert_classification_iain": "ESTABLISHED",
                            "expert_classification_liz": "ESTABLISHED",
                        }
                    )
                    idx += 1
        df = base_joined_df(rows)
        annotated = annotate_matches(df, REVIEWERS)
        result = aggregate_per_route_per_case_type(annotated, REVIEWERS)
        assert len(result) == 4
        assert "route" in result.columns
        assert "case_type" in result.columns

    def test_inter_reviewer_concordance(self, base_joined_df):
        rows = [
            {
                "gene_symbol": "gene0",
                "route": "route_A",
                "expert_classification_eric": "ESTABLISHED",
                "expert_classification_iain": "ESTABLISHED",
                "expert_classification_liz": "ESTABLISHED",
            },
            {
                "gene_symbol": "gene1",
                "route": "route_A",
                "expert_classification_eric": "ESTABLISHED",
                "expert_classification_iain": "ESTABLISHED",
                "expert_classification_liz": "NOVEL_ROLE",
            },
            {
                "gene_symbol": "gene2",
                "route": "route_A",
                "expert_classification_eric": "NOVEL_ROLE",
                "expert_classification_iain": "NOVEL_ROLE",
                "expert_classification_liz": "NOVEL_ROLE",
            },
            {
                "gene_symbol": "gene3",
                "route": "route_A",
                "expert_classification_eric": "ESTABLISHED",
                "expert_classification_iain": "NOVEL_ROLE",
                "expert_classification_liz": "ESTABLISHED",
            },
        ]
        df = base_joined_df(rows)
        annotated = annotate_matches(df, REVIEWERS)
        result = inter_reviewer_concordance(annotated, REVIEWERS)
        # eric vs iain: agree on gene0, gene2 → 2/4 = 0.5 (but gene1 also agree!)
        # gene0: E==E ✓, gene1: E==E ✓, gene2: N==N ✓, gene3: E!=N ✗ → 3/4
        assert result["eric_vs_iain"] == pytest.approx(0.75)
        # eric vs liz: gene0 E==E ✓, gene1 E!=N ✗, gene2 N==N ✓, gene3 E==E ✓ → 3/4
        assert result["eric_vs_liz"] == pytest.approx(0.75)
        # iain vs liz: gene0 E==E ✓, gene1 E!=N ✗, gene2 N==N ✓, gene3 N!=E ✗ → 2/4
        assert result["iain_vs_liz"] == pytest.approx(0.5)
        # unanimous: gene0 ✓, gene2 ✓ → 2/4 = 0.5
        assert result["unanimous"] == pytest.approx(0.5)


# ===========================================================================
# 3.5 Loading — use tmp_path
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

    def test_load_ground_truth_renames_sheet(self, tmp_path):
        df = pd.DataFrame(
            {
                "sheet": ["screen1"],
                "cluster_id": ["c1"],
                "gene_symbol": ["gene1"],
                "expert_classification_eric": ["ESTABLISHED"],
            }
        )
        csv_path = tmp_path / "gt.csv"
        df.to_csv(csv_path, index=False)
        result = load_ground_truth(csv_path)
        assert "screen_name" in result.columns
        assert "sheet" not in result.columns

    def test_load_ground_truth_screen_name_exists(self, tmp_path):
        df = pd.DataFrame(
            {
                "screen_name": ["screen1"],
                "cluster_id": ["c1"],
                "gene_symbol": ["gene1"],
                "expert_classification_eric": ["ESTABLISHED"],
            }
        )
        csv_path = tmp_path / "gt.csv"
        df.to_csv(csv_path, index=False)
        result = load_ground_truth(csv_path)
        assert "screen_name" in result.columns
        assert result.iloc[0]["screen_name"] == "screen1"
