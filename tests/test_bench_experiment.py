"""Unit tests for bench_experiment.py -- yaml loading, selection, state plumbing."""

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow import (  # noqa: E402
    bench_experiment,
)
from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_evaluator import (  # noqa: E402
    N_REAL_GENES,
    MetricPanel,
)
from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_experiment import (  # noqa: E402
    latest_run_dir,
    load_experiment,
    metric_value,
    run_experiment,
    select_holistic,
)

SOURCE_YAML = (
    Path(__file__).resolve().parent / "phase1_prompt_benchmarking" / "experiments" / "source.yaml"
)

# ---------------------------------------------------------------------------
# Experiment yaml loading
# ---------------------------------------------------------------------------

_MINIMAL_YAML = """\
experiment: t
model: {model_name: claude-sonnet-5}
run: {replicates: 1, route: single_call}
conditions:
  - {name: a, bundle_source: uniprot}
  - {name: b, bundle_source: affinage}
selection: {primary: coverage_weighted_category, metrics: [category, coverage]}
carry: [source]
"""


def _write_yaml(tmp_path, text):
    p = tmp_path / "exp.yaml"
    p.write_text(text)
    return p


class TestLoadExperiment:
    def test_minimal_yaml_parses(self, tmp_path):
        exp = load_experiment(_write_yaml(tmp_path, _MINIMAL_YAML))
        assert exp["experiment"] == "t"
        assert [c["name"] for c in exp["conditions"]] == ["a", "b"]

    def test_source_yaml_parses(self):
        exp = load_experiment(SOURCE_YAML)
        assert exp["experiment"] == "source"
        assert [c["name"] for c in exp["conditions"]] == ["uniprot", "affinage"]
        assert exp["run"]["route"] == "single_call"
        assert exp["carry"] == ["source"]

    @pytest.mark.parametrize("reserved", ["stages", "uses"])
    def test_reserved_keys_rejected_until_walkup_pr(self, tmp_path, reserved):
        with pytest.raises(ValueError, match="reserved for staged experiments"):
            load_experiment(_write_yaml(tmp_path, _MINIMAL_YAML + f"{reserved}: []\n"))

    def test_missing_required_key_rejected(self, tmp_path):
        text = _MINIMAL_YAML.replace(
            "selection: {primary: coverage_weighted_category, metrics: [category, coverage]}\n", ""
        )
        with pytest.raises(ValueError, match="missing required key 'selection:'"):
            load_experiment(_write_yaml(tmp_path, text))

    def test_unknown_route_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="not in registry"):
            load_experiment(_write_yaml(tmp_path, _MINIMAL_YAML.replace("single_call", "3a")))

    def test_duplicate_condition_names_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="duplicate condition names"):
            load_experiment(_write_yaml(tmp_path, _MINIMAL_YAML.replace("name: b", "name: a")))

    def test_unknown_condition_key_rejected(self, tmp_path):
        text = _MINIMAL_YAML.replace(
            "{name: a, bundle_source: uniprot}", "{name: a, bundle_source: uniprot, source: x}"
        )
        with pytest.raises(ValueError, match="unknown key"):
            load_experiment(_write_yaml(tmp_path, text))

    def test_unknown_selection_metric_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unknown selection metric"):
            load_experiment(
                _write_yaml(tmp_path, _MINIMAL_YAML.replace("[category, coverage]", "[accuracy]"))
            )

    def test_stage_and_select_raise_on_stageless_experiment(self, tmp_path):
        path = _write_yaml(tmp_path, _MINIMAL_YAML)
        with pytest.raises(ValueError, match="declares no stages"):
            run_experiment(path, stage="CAT")
        with pytest.raises(ValueError, match="declares no stages"):
            run_experiment(path, select=("CAT", "prior"))


# ---------------------------------------------------------------------------
# Selection primitives
# ---------------------------------------------------------------------------


def _cell(cat, n, nov, unc):
    return MetricPanel(
        category=cat, novel_subclass=nov, unchar_subclass=unc, coherence=(1, 4), n=n, failures=0
    )


def test_coverage_weighted_category_is_recall_over_all_genes():
    # category is correct/n (over scored genes); coverage-weighted is correct/N_REAL.
    p = _cell(0.80, 100, (0, 1), (0, 1))
    assert metric_value(p, "category") == 0.80
    assert metric_value(p, "coverage") == 100 / N_REAL_GENES
    assert metric_value(p, "coverage_weighted_category") == 0.80 * 100 / N_REAL_GENES


def test_selection_is_coverage_honest_not_fooled_by_gene_dropping():
    # uniprot has the top RAW category but scores 31 fewer genes; the
    # coverage-weighted primary must pick affinage, while the coverage-blind
    # raw-category rule rewards uniprot for dropping hard genes.
    metrics = ["category", "novel_subclass", "unchar_subclass", "coherence", "coverage"]
    cells = {
        "affinage__single_call": _cell(0.791, 103, (14, 31), (4, 7)),
        "uniprot__single_call": _cell(0.824, 72, (12, 26), (4, 6)),
    }
    winner, _dominated = select_holistic(cells, "coverage_weighted_category", metrics)
    assert winner == "affinage__single_call"
    raw_winner, _ = select_holistic(cells, "category", metrics)
    assert raw_winner == "uniprot__single_call"


def test_holistic_excludes_dominated_even_with_high_primary():
    def _panel(cat, nov, unc, coh):
        return MetricPanel(
            category=cat,
            novel_subclass=(round(nov * 100), 100),
            unchar_subclass=(round(unc * 100), 100),
            coherence=(round(coh * 100), 100),
            n=103,
            failures=0,
        )

    metrics = ["category", "novel_subclass", "unchar_subclass", "coherence"]
    cells = {
        "dominant": _panel(0.86, 0.60, 0.60, 0.60),
        "challenger": _panel(0.90, 0.50, 0.50, 0.50),  # top cat, beaten on nothing it wins
    }
    # 'challenger' is NOT dominated (it wins on category), so it should win.
    winner, dominated = select_holistic(cells, "category", metrics)
    assert winner == "challenger"
    assert dominated == []
    # Now make it genuinely dominated on every axis:
    cells["challenger"] = _panel(0.80, 0.50, 0.50, 0.50)
    winner, dominated = select_holistic(cells, "category", metrics)
    assert winner == "dominant"
    assert dominated == ["challenger"]


# ---------------------------------------------------------------------------
# Archived run dirs
# ---------------------------------------------------------------------------


def test_latest_run_dir_ignores_prefix_sibling_conditions(tmp_path, monkeypatch):
    # 'uniprot_backfill_<stamp>' sorts after 'uniprot_<stamp>' but must never be
    # picked up as a 'uniprot' run.
    monkeypatch.setattr(bench_experiment, "OUTPUTS", tmp_path)
    base = tmp_path / "source"
    for d in (
        "uniprot_20260101_000000",
        "uniprot_20260102_000000",
        "uniprot_backfill_20260103_000000",
        "uniprot_notes",
    ):
        (base / d).mkdir(parents=True)
    assert latest_run_dir("source", "uniprot").name == "uniprot_20260102_000000"
    assert latest_run_dir("source", "uniprot_backfill").name == "uniprot_backfill_20260103_000000"
    assert latest_run_dir("source", "affinage") is None
    assert latest_run_dir("nonexistent", "uniprot") is None


# ---------------------------------------------------------------------------
# Dry-run plumbing (the full path: yaml -> engine -> scoring -> state)
# ---------------------------------------------------------------------------


def test_dry_run_of_source_experiment_writes_state(tmp_path, monkeypatch):
    monkeypatch.setattr(bench_experiment, "OUTPUTS", tmp_path)
    monkeypatch.setattr(bench_experiment, "GT_PATH", tmp_path / "consensus_gt.csv")

    state = run_experiment(SOURCE_YAML, dry_run=True)

    # Two stamped, per-condition run dirs archive under OUTPUTS/source/.
    run_dirs = {c: tmp_path / "source" / d for c, d in state["runs"].items()}
    assert set(run_dirs) == {"uniprot", "affinage"}
    for cond, d in run_dirs.items():
        assert d.is_dir() and d.name.startswith(f"{cond}_")

    # Per-arm prompt purity: prompt construction is real even in dry-run.
    uniprot_prompts = (run_dirs["uniprot"] / "prompts.jsonl").read_text()
    affinage_prompts = (run_dirs["affinage"] / "prompts.jsonl").read_text()
    assert "affinage_functional_annotation" not in uniprot_prompts
    assert "UniProt_functional_annotation" not in affinage_prompts
    assert "affinage_functional_annotation" in affinage_prompts

    # The state file carries every family, with the same content as the return.
    on_disk = json.loads((tmp_path / "source" / "source_state.json").read_text())
    assert on_disk == json.loads(json.dumps(state))
    assert set(state["cells"]) == {"uniprot__single_call", "affinage__single_call"}
    for panel in state["cells"].values():
        assert panel["n"] > 0
    assert state["winner_condition"] in ("uniprot", "affinage")
    assert state["carry"] == {"source": state["winner_condition"]}
    for cond in ("uniprot", "affinage"):
        controls = {(d["screen"], d["cluster"]) for d in state["decoys"][cond]}
        assert {("aconcagua_interphase_shuffled", "17"), ("whitney", "49"), ("jebel", "0")} \
            <= controls
        assert state["diagnostics"][cond]["condition"] == f"{cond}__single_call"
        assert cond in state["audit_flags"]
        assert cond in state["pathway"]
    assert "reviewer_concordance" in state
    assert "source_preference" in state
