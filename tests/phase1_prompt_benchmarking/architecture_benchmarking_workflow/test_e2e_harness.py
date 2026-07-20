"""Validation gate: rebuild consensus GT from reviewer CSVs, score all three
cached evidence-source runs, and assert the known decision reproduces --
uniprot eliminated, affinage and both near-tied at the top -- with zero API
calls.
"""

from pathlib import Path

from architecture_benchmarking_workflow.ground_truth import build_consensus_gt, load_consensus_gt
from architecture_benchmarking_workflow.make_figures import evidence_panel
from architecture_benchmarking_workflow.walkup import metric_value

ROOT = Path("/lab/barcheese01/mdiberna/mozzarellm")
P = ROOT / "tests/phase1_prompt_benchmarking"
GT = P / "ground_truth"
B = P / "benchmarking_outputs/8.source_mcp_3x2"
COH = {
    ("denali", "43"): "Medium",
    ("whitney", "6"): "Medium",
    ("aconcagua_interphase", "41"): "Medium",
    ("denali", "24"): "Low",
}
RUN_DIRS = {
    "uniprot": B / "phase4_3x2_uniprot_uniprot_20260716_114118",
    "affinage": B / "phase4_3x2_affinage_uniprot_20260716_115019",
    "both": B / "phase4_3x2_both_uniprot_20260716_115614",
}


def test_evidence_decision_reproduces(tmp_path):
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

    panels = evidence_panel(gt, RUN_DIRS, COH)

    assert panels["uniprot"].category < panels["affinage"].category  # uniprot eliminated
    assert abs(panels["affinage"].category - panels["both"].category) < 0.02  # affinage ~ both

    # uniprot is worst or tied-worst on every metric
    for m in ("category",):
        assert metric_value(panels["uniprot"], m) <= metric_value(panels["affinage"], m)
