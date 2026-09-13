"""Figure smoke tests: synthetic states render to PNG+PDF (headless)."""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.workflow import bench_figures as bf  # noqa: E402

_PANEL = {
    "category": 0.9, "novel_subclass": [20, 30], "unchar_subclass": [3, 5],
    "coherence": [1, 4], "coverage": 1.0, "n": 103, "failures": 0,
}
_DIAG = {"per_class": {c: {"recall_consensus": 0.9} for c in
                       ("ESTABLISHED", "NOVEL_ROLE", "UNCHARACTERIZED")}}


def test_all_figures_render_png_and_pdf(tmp_path):
    bf.setup_plot_style()
    source = {
        "cells": {"uniprot__single_call": dict(_PANEL), "affinage__single_call": dict(_PANEL)},
        "diagnostics": {"uniprot": _DIAG, "affinage": _DIAG},
    }
    walkup = {
        "order": ["CAT", "GCR"],
        "stages": [
            {"stage": "CAT", "prior": dict(_PANEL),
             "candidates": {"a": dict(_PANEL), "b": dict(_PANEL)}, "selected": "a"},
            {"stage": "GCR", "prior": dict(_PANEL),
             "candidates": {"c": dict(_PANEL)}, "selected": None},
        ],
    }
    mode = {"cells": {f"{d}{s}__x": dict(_PANEL) for d in ("single_call", "cot", "stepwise")
                      for s in ("", "_lit", "_litb")}}
    order = {"cells": {f"O{i}__single_call": dict(_PANEL) for i in range(5)},
             "winner_condition": "O2"}

    for fn, state, stem in [
        (bf.fig_source, source, "source_comparison"),
        (bf.fig_walkup, walkup, "walkup_trajectory"),
        (bf.fig_mode, mode, "mode_matrix"),
        (bf.fig_order, order, "order_spread"),
    ]:
        fn(state, tmp_path)
        assert (tmp_path / f"{stem}.png").exists()
        assert (tmp_path / f"{stem}.pdf").exists()


def test_render_all_skips_missing_states(tmp_path, monkeypatch):
    monkeypatch.setattr(bf, "OUTPUTS", tmp_path)
    (tmp_path / "source").mkdir()
    (tmp_path / "source" / "source_state.json").write_text(json.dumps({
        "cells": {"uniprot__single_call": dict(_PANEL), "affinage__single_call": dict(_PANEL)},
        "diagnostics": {"uniprot": _DIAG, "affinage": _DIAG},
    }))
    written = bf.render_all(tmp_path / "figs")
    assert len(written) == 1
