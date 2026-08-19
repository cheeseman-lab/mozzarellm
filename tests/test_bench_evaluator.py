"""Unit tests for bench_evaluator.py"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_evaluator import (  # noqa: E402
    _consensus_subclass,
    build_consensus_gt,
    consensus_coherence,
    consensus_of,
    load_consensus_gt,
    pathway_loose_match,
    pathway_substring_match,
    score_decoys,
)

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


# ===========================================================================
# Per-run decoy validation (score_decoys)
# ===========================================================================


def _decoy_cell(route, screen, cluster, rep, confidence, valid=True):
    parsed = None if not valid else {"dominant_process": "x", "pathway_confidence": confidence}
    return {
        "run_id": f"exp__{route}__{screen}__cluster_{cluster}__rep_{rep}",
        "route": route,
        "parsed_output": parsed,
    }


def test_score_decoys_abstain_and_functional(tmp_path):
    rows = []
    # nonsense cluster -> correctly abstains (Low) across reps -> abstain PASS
    for r in (1, 2, 3):
        rows.append(_decoy_cell("single_call", "aconcagua_interphase_shuffled", "17", r, "Low"))
    # control-heavy cluster -> confidently annotates (High) -> abstain FAIL
    for r in (1, 2, 3):
        rows.append(_decoy_cell("single_call", "whitney", "49", r, "High"))
    # large coherent cluster -> valid High output every rep -> functional PASS
    for r in (1, 2, 3):
        rows.append(_decoy_cell("single_call", "jebel", "0", r, "High"))

    d = tmp_path / "run"
    d.mkdir()
    (d / "parsed_outputs.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    specs = {
        ("aconcagua_interphase_shuffled", "17"): "abstain",
        ("whitney", "49"): "abstain",
        ("jebel", "0"): "functional",
    }
    results = {(r.screen, r.cluster): r for r in score_decoys(d, specs)}
    assert results[("aconcagua_interphase_shuffled", "17")].passed is True
    assert results[("whitney", "49")].passed is False  # did not abstain
    assert results[("jebel", "0")].passed is True


def test_score_decoys_functional_fails_on_error(tmp_path):
    # jebel/0 errors out on one rep (truncation) -> functional FAIL
    rows = [
        _decoy_cell("single_call", "jebel", "0", 1, "High"),
        _decoy_cell("single_call", "jebel", "0", 2, None, valid=False),
        _decoy_cell("single_call", "jebel", "0", 3, "High"),
    ]
    d = tmp_path / "run"
    d.mkdir()
    (d / "parsed_outputs.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (res,) = score_decoys(d, {("jebel", "0"): "functional"})
    assert res.failures == 1
    assert res.passed is False


def test_score_decoys_abstain_fails_on_crash(tmp_path):
    # A crash is not abstention: 2 valid "Low" + 1 crash must NOT pass.
    rows = [
        _decoy_cell("single_call", "whitney", "49", 1, "Low"),
        _decoy_cell("single_call", "whitney", "49", 2, None, valid=False),
        _decoy_cell("single_call", "whitney", "49", 3, "Low"),
    ]
    d = tmp_path / "run"
    d.mkdir()
    (d / "parsed_outputs.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (res,) = score_decoys(d, {("whitney", "49"): "abstain"})
    assert res.modal_confidence == "Low"
    assert res.failures == 1
    assert res.passed is False  # crash masks abstention


def test_score_decoys_reports_completion(tmp_path):
    # jebel/0 (functional): the model classifies 8, 10, 10 genes across reps for a
    # 10-gene cluster -> median 10, completion 1.0 (over/under would show as != 1.0).
    rows = []
    for rep, k in ((1, 8), (2, 10), (3, 10)):
        rows.append(
            {
                "run_id": f"exp__single_call__jebel__cluster_0__rep_{rep}",
                "route": "single_call",
                "parsed_output": {
                    "dominant_process": "translation",
                    "pathway_confidence": "High",
                    "established_genes": [f"G{i}" for i in range(k)],
                    "novel_role_genes": [],
                    "uncharacterized_genes": [],
                },
            }
        )
    d = tmp_path / "run"
    d.mkdir()
    (d / "parsed_outputs.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (res,) = score_decoys(
        d, {("jebel", "0"): "functional"}, expected_counts={("jebel", "0"): 10}
    )
    assert res.genes_per_rep == [8, 10, 10]
    assert res.median_genes == 10.0
    assert res.expected_genes == 10 and res.completion == 1.0
    assert res.passed is True
