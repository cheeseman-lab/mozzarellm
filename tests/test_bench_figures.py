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


def test_all_figures_render_png_pdf_csv(tmp_path):
    bf.setup_plot_style()
    mode = _cells_state(
        [f"{d}{s}" for d in bf.DELIVERIES for s, _ in bf.LIT_SETTINGS], winner="cot_litb"
    )
    order = _cells_state(
        ["O", "O1", "O2"], winner="O", human_gate={"overrode": "O1", "to": "O", "reason": ""}
    )
    bf.fig_pipeline(tmp_path)
    bf.fig_benchmark_design(mode, tmp_path)
    bf.fig_source(_cells_state(["uniprot", "affinage"], winner="affinage"), tmp_path)
    bf.fig_walkup(_walkup_state(), tmp_path)
    bf.fig_mode_order(mode, order, tmp_path)
    bf.fig_features(_cells_state(["baseline", "features"], winner="features"), tmp_path)
    for stem in (
        "fig1_pipeline",
        "fig2_benchmark_design",
        "fig3_source",
        "fig4_walkup",
        "fig5_mode_order",
        "figS1_features",
    ):
        assert (tmp_path / f"{stem}.png").exists()
        assert (tmp_path / f"{stem}.pdf").exists()
    for stem in ("fig3_source", "fig4_walkup", "fig5_mode", "fig5_order", "figS1_features"):
        with (tmp_path / f"{stem}.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        assert rows and set(rows[0]) == {
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


def test_walkup_csv_round_trips_plotted_values(tmp_path):
    bf.setup_plot_style()
    state = _walkup_state()
    bf.fig_walkup(state, tmp_path)
    with (tmp_path / "fig4_walkup.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    sel = next(r for r in rows if r["role"] == "selected" and r["metric"] == "category")
    assert float(sel["value"]) == state["stages"][0]["candidates"]["a"]["category"]
    assert (int(sel["numerator"]), int(sel["denominator"])) == (93, 103)


def test_render_all_skips_missing_states(tmp_path, monkeypatch):
    monkeypatch.setattr(bf, "OUTPUTS", tmp_path)
    (tmp_path / "source").mkdir()
    (tmp_path / "source" / "source_state.json").write_text(
        json.dumps(_cells_state(["uniprot", "affinage"], winner="affinage"))
    )
    written = bf.render_all(tmp_path / "figs")
    assert len(written) == 3  # pipeline schematic, benchmark design, source
