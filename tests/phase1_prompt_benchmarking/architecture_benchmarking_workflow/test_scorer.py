"""Tests for scoring a model run against reviewer-consensus ground truth."""

import json
from pathlib import Path

import pytest

from architecture_benchmarking_workflow.ground_truth import build_consensus_gt, load_consensus_gt
from architecture_benchmarking_workflow.scorer import score_decoys, score_run

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "tests/phase1_prompt_benchmarking"
GT = P / "ground_truth"
# Cached runs live in the benchmarking_outputs git submodule.
B = P / "benchmarking_outputs/8.source_mcp_3x2"
needs_cached_runs = pytest.mark.skipif(
    not B.exists(), reason="cached runs require the benchmarking_outputs submodule"
)
COH = {
    ("denali", "43"): "Medium",
    ("whitney", "6"): "Medium",
    ("aconcagua_interphase", "41"): "Medium",
    ("denali", "24"): "Low",
}


def _gt(tmp_path):
    out = tmp_path / "gt.csv"
    build_consensus_gt(
        {
            "eric": GT / "annotation_eric.csv",
            "liz": GT / "annotation_liz.csv",
            "iain": GT / "annotation_iain.csv",
        },
        GT / "survey_v3_key.csv",
        [],
        out,
    )
    return load_consensus_gt(out)


@needs_cached_runs
def test_reproduces_3x2_panel(tmp_path):
    gt = _gt(tmp_path)
    aff = score_run(B / "phase4_3x2_affinage_uniprot_20260716_115019", gt, cluster_coherence=COH)
    uni = score_run(B / "phase4_3x2_uniprot_uniprot_20260716_114118", gt, cluster_coherence=COH)
    both = score_run(B / "phase4_3x2_both_uniprot_20260716_115614", gt, cluster_coherence=COH)
    assert aff.category == 115 / 133
    assert uni.category == 107 / 133
    assert round(both.category, 4) == round(116 / 133, 4)  # both 87%
    assert aff.novel_subclass == (29, 54)
    assert uni.novel_subclass == (29, 50)
    assert aff.unchar_subclass == (5, 7)
    assert aff.coherence == (2, 4)
    assert uni.coherence == (1, 4)
    assert both.coherence == (3, 4)


@needs_cached_runs
def test_route_equals_exact_match(tmp_path):
    gt = _gt(tmp_path)
    d = B / "phase4_3x2_affinage_uniprot_20260716_115019"
    # Exact match on the non-mcp cell's route reproduces the default-exclude panel;
    # exact match on the mcp route isolates the mcp cell (2 over-call failures).
    exact_non_mcp = score_run(d, gt, cluster_coherence=COH, route_equals="3a")
    exact_mcp = score_run(d, gt, cluster_coherence=COH, route_equals="3a_mcp_gap")
    assert exact_non_mcp.category == 115 / 133
    assert exact_non_mcp.failures == 0
    assert exact_mcp.failures == 2


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
