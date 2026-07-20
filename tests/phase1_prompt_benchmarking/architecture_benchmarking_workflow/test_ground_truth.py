"""Tests for consensus ground-truth construction from reviewer annotation CSVs."""

from pathlib import Path

from architecture_benchmarking_workflow.ground_truth import (
    build_consensus_gt,
    consensus_coherence,
    load_consensus_gt,
)

ROOT = Path("/lab/barcheese01/mdiberna/mozzarellm")
P = ROOT / "tests/phase1_prompt_benchmarking"
GT = P / "ground_truth"


def test_consensus_matches_known(tmp_path):
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
    gt = load_consensus_gt(out)
    real = {k: v for k, v in gt.items() if v["cluster_role"] == "real"}
    assert len(real) == 133
    assert sum(1 for v in real.values() if str(v["unanimous"]) in ("True", "1")) == 98
    fl = [v for v in real.values() if str(v["flagged"]) == "1"]
    assert sum(int(v["pref_affinage"]) for v in fl) == 51
    assert sum(int(v["pref_uniprot"]) for v in fl) == 29


def test_consensus_coherence_majority_and_ties():
    coh = consensus_coherence(
        {
            "eric": GT / "annotation_eric.csv",
            "liz": GT / "annotation_liz.csv",
            "iain": GT / "annotation_iain.csv",
        }
    )
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
        {
            "eric": GT / "annotation_eric.csv",
            "liz": GT / "annotation_liz.csv",
            "iain": GT / "annotation_iain.csv",
        },
        GT / "survey_v3_key.csv",
        [{"screen": "x", "cluster": "1", "decoy_type": "nonsense", "genes": ["G1", "G2"]}],
        out,
    )
    gt = load_consensus_gt(out)
    decoys = {k: v for k, v in gt.items() if v["cluster_role"] == "decoy"}
    assert len(decoys) == 2
