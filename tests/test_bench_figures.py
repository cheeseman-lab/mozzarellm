"""Figure smoke tests: synthetic states render to PNG+PDF+CSV (headless)."""

import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.workflow import bench_figures as bf  # noqa: E402

_PANEL = {
    "category": 0.9,
    "novel_subclass": [20, 30],
    "unchar_subclass": [3, 5],
    "coherence": [1, 4],
    "coverage": 1.0,
    "n": 103,
    "failures": 0,
}
_CLASSES = ("ESTABLISHED", "NOVEL_ROLE", "UNCHARACTERIZED")
_DIAG = {
    "n_scored": 103,
    "per_class": {c: {"recall_consensus": 0.9, "tp": 9, "n": 10} for c in _CLASSES},
    "per_cluster": {"whitney/6": {"correct": 25, "n": 25, "recall": 1.0}},
}
_DECOYS = [
    {"screen": "whitney", "cluster": "49", "expectation": "abstain", "passed": True},
    {"screen": "jebel", "cluster": "0", "expectation": "functional", "passed": False},
]
_CONC = {
    "reviewers": ["a", "b", "c"],
    "pairwise": {"a-b": 0.9, "a-c": 0.9, "b-c": 0.9},
    "unanimous_frac": 0.85,
    "fleiss_kappa": 0.8,
    "by_class": {c: {"frac": 0.8} for c in _CLASSES},
}


def _cells_state(conds, winner=None, **extra):
    return {
        "cells": {f"{c}__x": dict(_PANEL) for c in conds},
        "diagnostics": dict.fromkeys(conds, _DIAG),
        "decoys": dict.fromkeys(conds, _DECOYS),
        "winner_condition": winner or conds[0],
        "dominated": [f"{conds[-1]}__x"],
        "reviewer_concordance": _CONC,
        **extra,
    }


def _walkup_state():
    return {
        "order": ["CAT", "GCR"],
        "stages": [
            {
                "stage": "CAT",
                "goal": "category",
                "prior": dict(_PANEL),
                "prior_abstain": "4/4",
                "candidates": {"a": dict(_PANEL), "b": dict(_PANEL)},
                "candidate_abstain": {"a": "4/4", "b": "3/4"},
                "selected": "a",
            },
            {
                "stage": "GCR",
                "goal": "novel_subclass",
                "prior": dict(_PANEL),
                "prior_abstain": "4/4",
                "candidates": {"c": dict(_PANEL)},
                "candidate_abstain": {"c": "4/4"},
                "selected": None,
            },
        ],
    }


def test_criterion_value_shapes():
    assert bf.criterion_value(_PANEL, "category") == (0.9, 93, 103)
    assert bf.criterion_value(_PANEL, "novel_subclass") == (20 / 30, 20, 30)
    assert bf.criterion_value(_PANEL, "controls", "3/4") == (0.75, 3, 4)
    assert bf.criterion_value(_PANEL, "controls", _DECOYS) == (0.5, 1, 2)
    value, num, den = bf.criterion_value(_PANEL, "controls", None)
    assert value != value and (num, den) == (0, 0)


def test_walkup_rows_mark_goal_and_roles():
    rows = bf.rows_from_walkup(_walkup_state())
    cat = [r for r in rows if r["panel"] == "CAT"]
    assert {r["role"] for r in cat} == {"prior", "candidate", "selected"}
    assert all(r["goal"] == (r["metric"] == "category") for r in cat)
    gcr = [r for r in rows if r["panel"] == "GCR"]
    assert all(r["role"] != "selected" for r in gcr)
    assert len(rows) == (3 + 2) * len(bf.CRITERIA)


def test_cells_rows_roles():
    rows = bf.rows_from_cells(
        "mode", _cells_state(["cot_litb", "cot", "stepwise"], winner="cot_litb")
    )
    roles = {r["condition"]: r["role"] for r in rows}
    assert roles == {"cot_litb": "winner", "cot": "candidate", "stepwise": "dominated"}
    assert {r["metric"] for r in rows if r["panel"] == "per_class"} == {
        "recall_established",
        "recall_novel_role",
        "recall_uncharacterized",
    }


_TIDY_FIELDS = {
    "figure",
    "panel",
    "condition",
    "role",
    "metric",
    "value",
    "numerator",
    "denominator",
    "goal",
}


def _write_states(root, **states):
    for name, state in states.items():
        (root / name).mkdir(parents=True, exist_ok=True)
        (root / name / f"{name}_state.json").write_text(json.dumps(state))


def _chain_states():
    walkup = _walkup_state()
    walkup["stages"][1]["selected"] = "c"
    mode = _cells_state(
        [f"{d}{s}" for d, _ in bf.DELIVERIES for s, _ in bf.LIT_SETTINGS],
        winner="single_call_litb",
        human_gate={"overrode": "single_call_litb", "to": "cot_litb", "reason": ""},
    )
    return {
        "source": _cells_state(["uniprot", "affinage"], winner="affinage"),
        "walkup": walkup,
        "mode": mode,
        "order": _cells_state(["O", "O1"], winner="O"),
        "features": _cells_state(["baseline", "features_strength"], winner="features_strength"),
    }


def test_render_all_writes_tables_and_supplement(tmp_path):
    _write_states(tmp_path, **_chain_states())
    written = bf.render_all(tmp_path / "figs", outputs=tmp_path)
    data = tmp_path / "figs" / "data"
    for stem in ("source", "walkup", "mode", "order", "features"):
        with (data / f"{stem}.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        assert rows and set(rows[0]) == _TIDY_FIELDS
    bench = tmp_path / "figs" / "benchmark"
    for ext in ("png", "pdf", "svg"):
        assert bench / f"benchmark_composite.{ext}" in written
        assert bench / f"benchmark_prompt_scorecard.{ext}" in written
    assert not (bench / "benchmark_confidence_by_rubric.pdf").exists()  # no screen collected


def test_walkup_csv_round_trips_plotted_values(tmp_path):
    state = _walkup_state()
    bf.write_tables({"walkup": state}, tmp_path)
    with (tmp_path / "walkup.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    sel = next(r for r in rows if r["role"] == "selected" and r["metric"] == "category")
    assert float(sel["value"]) == state["stages"][0]["candidates"]["a"]["category"]
    assert (int(sel["numerator"]), int(sel["denominator"])) == (93, 103)


def test_render_all_skips_missing_states(tmp_path):
    _write_states(tmp_path, source=_cells_state(["uniprot", "affinage"], winner="affinage"))
    written = bf.render_all(tmp_path / "figs", outputs=tmp_path)
    assert written == [tmp_path / "figs" / "data" / "source.csv"]


def test_step_table_reads_human_gates_and_features(tmp_path):
    states = _chain_states()
    bf.write_tables(states, tmp_path)
    with (tmp_path / "mode.csv").open() as fh:
        roles = {(r["condition"], r["role"]) for r in csv.DictReader(fh)}
    assert ("cot_litb", "kept") in roles and ("single_call_litb", "overridden") in roles

    steps = bf.step_table(tmp_path)
    assert [s["label"] for s in steps] == [
        "source", "+ task", "+ category rules", "delivery", "order", "+ features"
    ]  # fmt: skip
    assert [s["kept"] for s in steps] == [
        "affinage",
        "a",
        "c",
        "cot_litb",
        "O",
        "features_strength",
    ]
    assert steps[2]["goal"] == "novel_subclass"
    assert steps[3]["after"]["category"] == (0.9, 93, 103)


def test_mann_whitney_matches_reference():
    # scipy.stats.mannwhitneyu(a, b, alternative="greater").pvalue
    a, b = [3, 4, 5, 5, 6, 7], [1, 2, 2, 3, 4]
    assert abs(bf.mann_whitney_greater(a, b) - 0.010633468552799844) < 1e-12


def _screen_run(root, arm, levels, genes):
    run = root / f"run_{arm}"
    run.mkdir(parents=True)
    with (run / "aconcagua_r12_clusters.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cluster_id", "dominant_process", "pathway_confidence", "n_genes"])
        for i, (level, n) in enumerate(levels):
            process = "No coherent biological pathway" if level == "none" else "DNA repair"
            w.writerow([i, process, "Low" if level == "none" else level, n])
    with (run / "aconcagua_r12_genes.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gene", "cluster_id", "category"])
        w.writerows(genes)


def test_collect_screen_tables_and_render_both_figures(tmp_path, monkeypatch):
    k = 0
    genes = [
        ["FANCD2", k, "ESTABLISHED"],
        ["CCNA2", k, "NOVEL_ROLE"],
        ["LRCH2", k, "UNCHARACTERIZED"],
    ]
    monkeypatch.setattr(bf, "EXAMPLES", {k: "DNA repair"})
    for arm, _ in bf.SPREAD_ARMS[1:]:
        _screen_run(tmp_path, arm, [("High", 3), ("none", 5)], genes)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    for i, (process, level) in enumerate([("DNA repair", "Medium"), ("No coherent x", "Low")]):
        (legacy / f"cluster_{i}.json").write_text(
            json.dumps({"dominant_process": process, "pathway_confidence": level})
        )
    phate = tmp_path / "phate.tsv"
    rows = [(g, k, -0.03 + 0.002 * i, -0.02 + 0.001 * i, 0.6) for i, (g, _, _) in enumerate(genes)]
    rows += [(f"G{i}", k + 1, -0.05 + 0.001 * i, -0.01, 0.55) for i in range(5)]
    phate.write_text(
        "gene_symbol_0\tcluster\tPHATE_0\tPHATE_1\tperturbation_auc\n"
        + "".join("\t".join(map(str, r)) + "\n" for r in rows)
    )
    data = tmp_path / "figs" / "data"
    bf.collect_aconcagua(tmp_path, legacy, phate, data)
    with (data / "screen_confidence.csv").open() as fh:
        conf = {(r["arm"], r["level"]): int(r["n"]) for r in csv.DictReader(fh)}
    assert conf[("legacy", "Medium")] == 1 and conf[("legacy", "none")] == 1
    assert conf[(bf.SHIPPED_ARM, "High")] == 1 and conf[(bf.SHIPPED_ARM, "none")] == 1

    _write_states(tmp_path, **_chain_states())
    written = bf.render_all(tmp_path / "figs", outputs=tmp_path)
    main = tmp_path / "figs" / f"mozzarellm_cluster_{k}"
    for name in ("labeling_schematic", "labeled_screen", "cluster_zoom", "composite"):
        assert main / f"mozzarellm_{name}.svg" in written
    assert tmp_path / "figs" / "benchmark" / "benchmark_confidence_vs_phenotype.pdf" in written
