"""Tests for the source/mode winner-selection primitives (pure logic, no API)."""

import sys
from pathlib import Path

P = Path(__file__).resolve().parent / "phase1_prompt_benchmarking"
if str(P) not in sys.path:
    sys.path.insert(0, str(P))

from architecture_benchmarking_workflow.bench_evaluator import MetricPanel  # noqa: E402
from architecture_benchmarking_workflow.bench_pipeline_common import (  # noqa: E402
    HOLISTIC_METRICS,
    N_REAL_GENES,
    PRIMARY,
    metric_value,
    select_holistic,
)


def test_coverage_weighted_category_is_recall_over_all_genes():
    # category is correct/n (over scored genes); coverage-weighted is correct/N_REAL.
    p = MetricPanel(
        category=0.80, novel_subclass=(0, 1), unchar_subclass=(0, 1), coherence=(0, 1),
        n=100, failures=0,
    )
    assert metric_value(p, "category") == 0.80
    assert metric_value(p, "coverage") == 100 / N_REAL_GENES
    assert metric_value(p, "coverage_weighted_category") == 0.80 * 100 / N_REAL_GENES


def _cell(cat, n, nov, unc):
    return MetricPanel(
        category=cat, novel_subclass=nov, unchar_subclass=unc, coherence=(1, 4), n=n, failures=0
    )


def test_source_selection_is_coverage_honest_not_fooled_by_gene_dropping():
    # The real Step-1 blank-W0 source x MCP panels. uniprot::single_call has the
    # top RAW category (0.824) but scores only 102/133 genes; affinage leads on
    # coverage-weighted recall. The baked-in coverage-weighted primary must pick
    # affinage, while the old raw-category rule picks the coverage-collapsed uniprot.
    cells = {
        "affinage::single_call_mcp": _cell(0.767, 133, (19, 39), (2, 6)),
        "both::single_call_mcp": _cell(0.767, 133, (14, 39), (2, 6)),
        "uniprot::single_call_mcp": _cell(0.714, 133, (9, 23), (3, 6)),
        "affinage::single_call": _cell(0.791, 115, (14, 31), (4, 7)),
        "uniprot::single_call": _cell(0.824, 102, (12, 26), (4, 6)),
        "both::single_call": _cell(0.759, 108, (15, 22), (4, 7)),
    }
    winner, _dominated = select_holistic(cells, PRIMARY, HOLISTIC_METRICS)
    assert winner.split("::")[0] == "affinage"
    # the old coverage-blind rule rewards uniprot for dropping 31 hard genes.
    raw_winner, _ = select_holistic(cells, "category", HOLISTIC_METRICS)
    assert raw_winner == "uniprot::single_call"


_HOLISTIC_METRICS = ["category", "novel_subclass", "unchar_subclass", "coherence"]


def _panel(cat, nov, unc, coh):
    return MetricPanel(
        category=cat,
        novel_subclass=(round(nov * 100), 100),
        unchar_subclass=(round(unc * 100), 100),
        coherence=(round(coh * 100), 100),
        n=133,
        failures=0,
    )


def test_holistic_picks_argmax_primary_among_survivors():
    # The real Stage-1 source panel: uniprot::single_call is the category floor
    # but is not dominated; the winner is the top-category (0.872) cell, not a
    # guarded do-no-harm fallback to the baseline.
    cells = {
        "uniprot::single_call": _panel(0.752, 0.543, 0.667, 0.500),
        "uniprot::single_call_mcp": _panel(0.857, 0.538, 0.500, 0.750),
        "affinage::single_call": _panel(0.850, 0.574, 0.857, 0.250),
        "affinage::single_call_mcp": _panel(0.872, 0.593, 0.500, 0.750),
        "both::single_call": _panel(0.857, 0.667, 0.571, 0.500),
        "both::single_call_mcp": _panel(0.872, 0.519, 0.667, 0.750),
    }
    winner, dominated = select_holistic(cells, "category", _HOLISTIC_METRICS)
    assert cells[winner].category == 0.872
    assert winner in ("affinage::single_call_mcp", "both::single_call_mcp")


def test_holistic_excludes_dominated_even_with_high_primary():
    cells = {
        "dominant": _panel(0.86, 0.60, 0.60, 0.60),
        "dominated_high_cat": _panel(0.90, 0.50, 0.50, 0.50),  # top cat but beaten on nothing it wins
    }
    # 'dominated_high_cat' is NOT dominated (it wins on category), so it should win.
    winner, dominated = select_holistic(cells, "category", _HOLISTIC_METRICS)
    assert winner == "dominated_high_cat"
    assert dominated == []
    # Now make it genuinely dominated on every axis:
    cells["dominated_high_cat"] = _panel(0.80, 0.50, 0.50, 0.50)
    winner, dominated = select_holistic(cells, "category", _HOLISTIC_METRICS)
    assert winner == "dominant"
    assert "dominated_high_cat" in dominated
