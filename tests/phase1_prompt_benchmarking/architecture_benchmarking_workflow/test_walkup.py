"""Tests for the prompt walkup driver (pure logic, no API calls)."""

from architecture_benchmarking_workflow.scorer import MetricPanel
from architecture_benchmarking_workflow.walkup import (
    StageSpec,
    run_walkup,
    select_holistic,
    select_winner,
)


def P(cat, nov=(29, 50), unc=(3, 6), coh=(1, 4)):  # noqa: N802
    return MetricPanel(
        category=cat, novel_subclass=nov, unchar_subclass=unc, coherence=coh, n=133, failures=0
    )


def test_greedy_primary_with_guard():
    base = P(0.80)
    variants = {"W_better": P(0.86, nov=(30, 50)), "W_regress": P(0.90, nov=(20, 50))}
    assert (
        select_winner(base, variants, primary="category", guards=["novel_subclass"], eps=0.02)
        == "W_better"
    )


def test_keep_baseline_if_nothing_beats():
    base = P(0.86)
    assert select_winner(base, {"W1": P(0.85)}, primary="category", guards=[]) == "baseline"


def test_tie_breaks_lexicographically():
    base = P(0.80)
    assert (
        select_winner(base, {"Wb": P(0.86), "Wa": P(0.86)}, primary="category", guards=[]) == "Wa"
    )


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


def test_run_walkup_carries_winners_forward():
    # two stages; stage1 has a winner, stage2 has none (baseline kept)
    stages = (
        StageSpec("GCR", "category", ("novel_subclass",)),
        StageSpec("NPR", "novel_subclass", ("category",)),
    )
    panels = {
        ("GCR", "W1"): P(0.90),
        ("NPR", "W2"): P(0.90, nov=(20, 50)),
    }  # NPR variant regresses its own primary

    def variants_for(stage):
        return {"GCR": ["W1"], "NPR": ["W2"]}[stage.name]

    def generate_fn(stage, variant, carried):
        return (stage.name, variant)

    def score_fn(run_dir):
        return panels[run_dir]

    res = run_walkup(stages, P(0.80), variants_for, generate_fn, score_fn)
    assert res.carried == ["W1"]  # GCR winner carried, NPR kept baseline
    assert res.final_panel.category == 0.90
    assert len(res.stage_rows) == 2
