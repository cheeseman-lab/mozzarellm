"""Unit tests for the unified benchmark evaluator.

Covers the three responsibilities folded into bench_evaluator: consensus
ground-truth construction, per-run scoring against that ground truth, and the
source-diagnostic metrics.
"""

import json
import sys
from pathlib import Path

import pytest

P = Path(__file__).resolve().parent / "phase1_prompt_benchmarking"
if str(P) not in sys.path:
    sys.path.insert(0, str(P))

from architecture_benchmarking_workflow.bench_evaluator import (  # noqa: E402
    _consensus_class,
    _consensus_subclass,
    build_consensus_gt,
    consensus_coherence,
    load_consensus_gt,
    reviewer_concordance,
    reviewer_label_sets,
    score_decoys,
    score_run,
    source_diagnostics,
    source_preference_tally,
)

GT = P / "benchmark_inputs" / "ground_truth"
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


def _reviewers() -> dict[str, Path]:
    return {r: GT / f"annotation_{r}.csv" for r in ("eric", "liz", "iain")}


def _gt(tmp_path):
    out = tmp_path / "gt.csv"
    build_consensus_gt(_reviewers(), GT / "survey_key.csv", [], out)
    return load_consensus_gt(out)


# ---------------------------------------------------------------------------
# Consensus ground truth
# ---------------------------------------------------------------------------


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


def test_consensus_class_majority_rule():
    # >=2-of-3 resolves; a 1-1-1 three-way split is unresolved ("").
    assert _consensus_class(["ESTABLISHED", "ESTABLISHED", "ESTABLISHED"]) == ("ESTABLISHED", True, 3)
    assert _consensus_class(["NOVEL_ROLE", "NOVEL_ROLE", "ESTABLISHED"]) == ("NOVEL_ROLE", False, 2)
    assert _consensus_class(["ESTABLISHED", "NOVEL_ROLE", "UNCHARACTERIZED"]) == ("", False, 1)


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


# ---------------------------------------------------------------------------
# Per-run scoring
# ---------------------------------------------------------------------------


@needs_cached_runs
def test_reproduces_3x2_panel(tmp_path):
    gt = _gt(tmp_path)
    aff = score_run(B / "phase4_3x2_affinage_uniprot_20260716_115019", gt, cluster_coherence=COH)
    uni = score_run(B / "phase4_3x2_uniprot_uniprot_20260716_114118", gt, cluster_coherence=COH)
    both = score_run(B / "phase4_3x2_both_uniprot_20260716_115614", gt, cluster_coherence=COH)
    # Category under the less-high hedge-resolution policy (UNCHAR > NOVEL > EST);
    # subclass under the ordinal-median consensus rule. Category/coherence are
    # unchanged from plurality consensus (no three-way category splits); the
    # median only shifts subclass on all-different votes (aff novel 29 -> 30).
    assert aff.category == 115 / 133
    assert uni.category == 106 / 133
    assert round(both.category, 4) == round(115 / 133, 4)  # both 86%
    assert aff.novel_subclass == (30, 54)
    assert uni.novel_subclass == (28, 49)
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


# ---------------------------------------------------------------------------
# Source diagnostics
# ---------------------------------------------------------------------------


def _write_reviewer_csv(path, rows):
    header = "annotator,screen,cluster,pathway,coherence,cluster_notes,gene,classification,subclass,preferred_source,used_web\n"
    with open(path, "w") as fh:
        fh.write(header)
        for r in rows:
            fh.write(f"{r['annotator']},s1,1,p,High,,{r['gene']},{r['classification']},,a,no\n")


def test_reviewer_label_sets_unions_across_reviewers(tmp_path):
    a = tmp_path / "annotation_a.csv"
    b = tmp_path / "annotation_b.csv"
    _write_reviewer_csv(a, [{"annotator": "a", "gene": "G1", "classification": "NOVEL_ROLE"},
                            {"annotator": "a", "gene": "G2", "classification": "ESTABLISHED"}])
    _write_reviewer_csv(b, [{"annotator": "b", "gene": "G1", "classification": "ESTABLISHED"},
                            {"annotator": "b", "gene": "G2", "classification": "ESTABLISHED"}])
    labels = reviewer_label_sets({"a": a, "b": b})
    assert labels[("s1", "1", "G1")] == {"NOVEL_ROLE", "ESTABLISHED"}  # reviewers disagree
    assert labels[("s1", "1", "G2")] == {"ESTABLISHED"}  # reviewers agree


def test_source_preference_tally_overall_and_by_class():
    gt = {
        ("s1", "1", "G1"): {"cluster_role": "real", "consensus_class": "NOVEL_ROLE",
                            "pref_affinage": "2", "pref_uniprot": "1", "pref_both": "0", "pref_neither": "0"},
        ("s1", "1", "G2"): {"cluster_role": "real", "consensus_class": "ESTABLISHED",
                            "pref_affinage": "1", "pref_uniprot": "0", "pref_both": "1", "pref_neither": "1"},
        ("s1", "1", "D"): {"cluster_role": "decoy", "consensus_class": "", "pref_affinage": "9"},
    }
    tally = source_preference_tally(gt)
    assert tally["overall"] == {"affinage": 3, "uniprot": 1, "both": 1, "neither": 1}  # decoy excluded
    assert tally["by_class"]["NOVEL_ROLE"]["affinage"] == 2


def test_reviewer_concordance_pairwise_and_unanimity(tmp_path):
    a = tmp_path / "annotation_a.csv"
    b = tmp_path / "annotation_b.csv"
    c = tmp_path / "annotation_c.csv"
    # G1: all agree (unanimous, ESTABLISHED). G2: a,b agree / c differs (split, NOVEL).
    _write_reviewer_csv(a, [{"annotator": "a", "gene": "G1", "classification": "ESTABLISHED"},
                            {"annotator": "a", "gene": "G2", "classification": "NOVEL_ROLE"}])
    _write_reviewer_csv(b, [{"annotator": "b", "gene": "G1", "classification": "ESTABLISHED"},
                            {"annotator": "b", "gene": "G2", "classification": "NOVEL_ROLE"}])
    _write_reviewer_csv(c, [{"annotator": "c", "gene": "G1", "classification": "ESTABLISHED"},
                            {"annotator": "c", "gene": "G2", "classification": "ESTABLISHED"}])
    gt = {
        ("s1", "1", "G1"): {"cluster_role": "real", "consensus_class": "ESTABLISHED"},
        ("s1", "1", "G2"): {"cluster_role": "real", "consensus_class": "NOVEL_ROLE"},
    }
    conc = reviewer_concordance({"a": a, "b": b, "c": c}, gt)
    assert conc["n_genes"] == 2
    assert conc["pairwise"]["a-b"] == 1.0  # a,b agree on both genes
    assert conc["pairwise"]["a-c"] == 0.5  # agree on G1 only
    assert conc["unanimous"] == 1 and conc["unanimous_frac"] == 0.5
    assert conc["by_class"]["ESTABLISHED"]["frac"] == 1.0  # G1 unanimous
    assert conc["by_class"]["NOVEL_ROLE"]["frac"] == 0.0  # G2 split


def test_source_diagnostics_consensus_vs_lenient(tmp_path):
    # One rep: model calls A=ESTABLISHED, B=ESTABLISHED, C=UNCHARACTERIZED.
    rec = {
        "run_id": "exp__affinage__single_call__s1__cluster_1__rep_1",
        "route": "affinage__single_call",
        "parsed_output": {
            "dominant_process": "x",
            "established_genes": ["A", "B"],
            "novel_role_genes": [],
            "uncharacterized_genes": [{"gene": "C", "class": "DARK_GENE"}],
            "pathway_confidence": "Low",
        },
    }
    (tmp_path / "parsed_outputs.jsonl").write_text(json.dumps(rec) + "\n")
    gt = {
        ("s1", "1", "A"): {"cluster_role": "real", "consensus_class": "ESTABLISHED"},
        ("s1", "1", "B"): {"cluster_role": "real", "consensus_class": "NOVEL_ROLE"},
        ("s1", "1", "C"): {"cluster_role": "real", "consensus_class": "NOVEL_ROLE"},
    }
    reviewer_labels = {
        ("s1", "1", "A"): {"ESTABLISHED"},
        ("s1", "1", "B"): {"NOVEL_ROLE", "ESTABLISHED"},  # a reviewer endorsed the ESTABLISHED call
        ("s1", "1", "C"): {"NOVEL_ROLE"},
    }
    d = source_diagnostics(tmp_path, gt, reviewer_labels, "affinage__single_call", n_real=3)
    assert d["n_scored"] == 3
    assert d["consensus_recall"] == 1 / 3  # only A matches consensus
    assert d["any_recall"] == 2 / 3  # A and B land within human envelope
    assert d["forgiven"] == 1  # B: right for a reviewer, wrong for consensus
    assert d["consensus_cw"] == 1 / 3 and d["any_cw"] == 2 / 3
    nov = d["per_class"]["NOVEL_ROLE"]
    assert nov["n"] == 2 and nov["recall_consensus"] == 0.0 and nov["recall_any"] == 0.5
    assert nov["confusion"] == {"ESTABLISHED": 1, "UNCHARACTERIZED": 1}
    # precision/recall/F1 emitted per class: no novel calls -> all zero
    assert nov["tp"] == 0 and nov["fn"] == 2 and nov["precision"] == 0.0 and nov["f1"] == 0.0
    # ESTABLISHED: A correct (TP) + B (novel) mislabeled established (FP) -> precision 0.5, recall 1.0
    est = d["per_class"]["ESTABLISHED"]
    assert est["tp"] == 1 and est["fp"] == 1 and est["precision"] == 0.5 and est["recall"] == 1.0
    # per-cluster recall emitted: 1 of 3 genes matched consensus
    assert d["per_cluster"]["s1/1"] == {"correct": 1, "n": 3, "recall": 0.333}


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


# ---------------------------------------------------------------------------
# Pathway agreement
# ---------------------------------------------------------------------------


def test_pathway_substring_and_loose_matching():
    from architecture_benchmarking_workflow.bench_evaluator import (
        pathway_loose_match,
        pathway_substring_match,
    )

    # expert term is a substring of the verbose prediction -> both match
    assert pathway_substring_match("nuclear pore complex assembly", ["nuclear pore"]) is True
    assert pathway_loose_match("nuclear pore complex assembly", ["nuclear pore"]) is True
    # prediction is a substring of the expert term -> only loose matches
    assert pathway_substring_match("splicing", ["mRNA splicing regulation"]) is False
    assert pathway_loose_match("splicing", ["mRNA splicing regulation"]) is True
    # empty prediction never matches
    assert pathway_substring_match("", ["x"]) is False
    assert pathway_loose_match("", ["x"]) is False


def test_pathway_diagnostics_no_semantic(tmp_path):
    from architecture_benchmarking_workflow.bench_evaluator import pathway_diagnostics

    rec = {
        "run_id": "exp__aff__single_call__s1__cluster_1__rep_1",
        "route": "aff__single_call",
        "parsed_output": {
            "dominant_process": "nuclear pore complex",
            "pathway_confidence": "High",
            "established_genes": ["A"],
            "novel_role_genes": [],
            "uncharacterized_genes": [],
        },
    }
    (tmp_path / "parsed_outputs.jsonl").write_text(json.dumps(rec) + "\n")
    gt = {("s1", "1", "A"): {"cluster_role": "real", "consensus_class": "ESTABLISHED"}}
    revs = {}
    for name, pw in [("a", "nuclear pore"), ("b", "nuclear envelope transport")]:
        p = tmp_path / f"annotation_{name}.csv"
        _write_reviewer_csv_pathway(p, name, pw)
        revs[name] = p
    r = pathway_diagnostics(tmp_path, gt, revs, "aff__single_call", use_semantic=False)
    assert r["n_clusters"] == 1
    assert r["substring_rate"] == 1.0  # "nuclear pore" is a substring of the prediction
    assert r["loose_rate"] == 1.0
    assert r["semantic_mean"] is None and r["semantic_available"] is False


def _write_reviewer_csv_pathway(path, annotator, pathway):
    path.write_text(
        "annotator,screen,cluster,pathway,coherence,cluster_notes,gene,classification,subclass,preferred_source,used_web\n"
        f"{annotator},s1,1,{pathway},High,,A,ESTABLISHED,,a,no\n"
    )
