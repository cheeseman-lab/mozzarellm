"""Unit tests for bench_experiment.py -- yaml loading, selection, state plumbing."""

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.workflow import (  # noqa: E402
    bench_experiment,
)
from benchmarks.workflow.bench_evaluator import (  # noqa: E402
    N_REAL_GENES,
    MetricPanel,
)
from benchmarks.workflow.bench_experiment import (  # noqa: E402
    latest_run_dir,
    load_experiment,
    metric_value,
    run_experiment,
    select_holistic,
)

SOURCE_YAML = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "experiments" / "source.yaml"
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


# ---------------------------------------------------------------------------
# Staged experiments
# ---------------------------------------------------------------------------

_STAGED_YAML = """\
experiment: w
model: {model_name: claude-sonnet-5}
run: {replicates: 1, route: single_call}
uses: source.carry.source
stages:
  - component: CAT
    goal: category
    candidates:
      - {id: c1, rationale: test framing, text: CAT TEXT ONE}
  - component: GCR
    goal: category
    candidates:
      - {id: g1, rationale: test framing, text: GCR TEXT ONE}
"""


class TestStagedSchema:
    def test_staged_yaml_parses(self, tmp_path):
        exp = load_experiment(_write_yaml(tmp_path, _STAGED_YAML))
        assert [s["component"] for s in exp["stages"]] == ["CAT", "GCR"]
        assert exp["uses"] == "source.carry.source"

    def test_stages_and_conditions_are_exclusive(self, tmp_path):
        text = _STAGED_YAML + "conditions:\n  - {name: a, bundle_source: uniprot}\n"
        with pytest.raises(ValueError, match="never both"):
            load_experiment(_write_yaml(tmp_path, text))

    def test_malformed_uses_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="must be '<experiment>.carry.<key>'"):
            load_experiment(
                _write_yaml(tmp_path, _STAGED_YAML.replace("source.carry.source", "source"))
            )

    def test_stage_component_must_be_a_prompt_slot(self, tmp_path):
        with pytest.raises(ValueError, match="not a prompt slot"):
            load_experiment(_write_yaml(tmp_path, _STAGED_YAML.replace("component: GCR", "component: XXX")))

    def test_stage_goal_must_be_a_selection_metric(self, tmp_path):
        text = _STAGED_YAML.replace("goal: category", "goal: accuracy", 1)
        with pytest.raises(ValueError, match="goal 'accuracy'"):
            load_experiment(_write_yaml(tmp_path, text))

    def test_candidate_id_prior_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="not 'prior'"):
            load_experiment(_write_yaml(tmp_path, _STAGED_YAML.replace("id: c1", "id: prior")))

    def test_candidate_requires_rationale(self, tmp_path):
        text = _STAGED_YAML.replace("rationale: test framing, ", "", 1)
        with pytest.raises(ValueError, match="missing 'rationale'"):
            load_experiment(_write_yaml(tmp_path, text))

    def test_staged_invocation_requires_stage_or_select(self, tmp_path):
        with pytest.raises(ValueError, match="is staged"):
            run_experiment(_write_yaml(tmp_path, _STAGED_YAML))

    def test_source_override_needs_a_uses_source(self, tmp_path):
        with pytest.raises(ValueError, match="declares no uses.source"):
            run_experiment(_write_yaml(tmp_path, _MINIMAL_YAML), source="affinage")


class TestStagedInvocation:
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bench_experiment, "OUTPUTS", tmp_path)
        monkeypatch.setattr(bench_experiment, "GT_PATH", tmp_path / "consensus_gt.csv")
        (tmp_path / "source").mkdir()
        (tmp_path / "source" / "source_state.json").write_text(
            json.dumps({"carry": {"source": "affinage"}})
        )
        return _write_yaml(tmp_path, _STAGED_YAML)

    def test_stage_runs_scores_and_stops_at_the_gate(self, tmp_path, monkeypatch):
        path = self._setup(tmp_path, monkeypatch)
        state = run_experiment(path, stage="CAT", dry_run=True)

        assert state["source"] == "affinage"  # resolved through uses:
        assert [s["stage"] for s in state["stages"]] == ["CAT"]  # GCR did NOT run
        rec = state["stages"][0]
        assert rec["resolved"] == {"source": "affinage", "carried_components": []}
        assert rec["selected"] is None
        assert rec["prior"]["n"] > 0 and rec["candidates"]["c1"]["n"] > 0
        run_dir = tmp_path / "w" / rec["run_dir"]
        assert run_dir.is_dir() and rec["run_dir"].startswith("CAT_")
        on_disk = json.loads((tmp_path / "w" / "w_state.json").read_text())
        assert on_disk == json.loads(json.dumps(state))

    def test_select_carries_text_and_finalizes_after_last_stage(self, tmp_path, monkeypatch):
        path = self._setup(tmp_path, monkeypatch)
        run_experiment(path, stage="CAT", dry_run=True)

        with pytest.raises(ValueError, match="has not been run yet"):
            run_experiment(path, select=("GCR", "g1"))
        with pytest.raises(ValueError, match="unknown candidate"):
            run_experiment(path, select=("CAT", "nope"))

        state = run_experiment(path, select=("CAT", "c1"))
        assert state["carried"]["CAT"] == "CAT TEXT ONE"
        assert state["components_filled"] == ["CAT"]
        assert "carry" not in state  # GCR still pending

        state = run_experiment(path, stage="GCR", dry_run=True)
        rec = next(s for s in state["stages"] if s["stage"] == "GCR")
        assert rec["resolved"]["carried_components"] == ["CAT"]
        # The carried CAT text reaches the next stage's prompts.
        prompts = (tmp_path / "w" / rec["run_dir"] / "prompts.jsonl").read_text()
        assert "CAT TEXT ONE" in prompts

        state = run_experiment(path, select=("GCR", "prior"))
        assert state["carried"]["GCR"] == ""
        assert state["winner"] == "CAT"
        assert state["carry"] == {
            "source": "affinage",
            "components_filled": ["CAT"],
            "final_component_texts": {"CAT": "CAT TEXT ONE"},
        }

    def test_mid_experiment_source_switch_refused(self, tmp_path, monkeypatch):
        path = self._setup(tmp_path, monkeypatch)
        run_experiment(path, stage="CAT", dry_run=True)
        with pytest.raises(ValueError, match="refusing"):
            run_experiment(path, stage="GCR", dry_run=True, source="uniprot")

    def test_source_override_beats_uses(self, tmp_path, monkeypatch):
        path = self._setup(tmp_path, monkeypatch)
        state = run_experiment(path, stage="CAT", dry_run=True, source="uniprot")
        assert state["source"] == "uniprot"
        assert state["stages"][0]["resolved"]["source"] == "uniprot"


WALKUP_YAML = SOURCE_YAML.parent / "walkup.yaml"


def test_walkup_yaml_parses_with_the_full_candidate_bank():
    exp = load_experiment(WALKUP_YAML)
    assert exp["uses"] == "source.carry.source"
    stages = {s["component"]: s for s in exp["stages"]}
    assert list(stages) == ["CAT", "GCR", "NPR", "UPR", "PCC"]
    goals = {c: s["goal"] for c, s in stages.items()}
    assert goals == {
        "CAT": "category",
        "GCR": "category",
        "NPR": "novel_subclass",
        "UPR": "unchar_subclass",
        "PCC": "coherence",
    }
    assert {c["id"] for c in stages["CAT"]["candidates"]} == {
        "concise",
        "discovery_first",
        "pathway_anchored",
        "process_relative",
        "process_guarded",
    }
    for stage in stages.values():
        for cand in stage["candidates"]:
            assert cand["rationale"] and cand["text"]


# ---------------------------------------------------------------------------
# Cross-experiment inputs on stage-less experiments (uses: mapping)
# ---------------------------------------------------------------------------

_DOWNSTREAM_YAML = """\
experiment: m
model: {model_name: claude-sonnet-5}
run: {replicates: 1, route: single_call}
uses:
  source: walkup.carry.source
  component_overrides: walkup.carry.final_component_texts
conditions:
  - {name: single_call, route: single_call}
  - {name: cot, route: cot}
selection: {primary: coverage_weighted_category, metrics: [category, coverage]}
carry: [source, mode]
"""


class TestStagelessUses:
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bench_experiment, "OUTPUTS", tmp_path)
        monkeypatch.setattr(bench_experiment, "GT_PATH", tmp_path / "consensus_gt.csv")
        (tmp_path / "walkup").mkdir()
        (tmp_path / "walkup" / "walkup_state.json").write_text(
            json.dumps(
                {
                    "carry": {
                        "source": "affinage",
                        "final_component_texts": {"CAT": "TUNED CAT TEXT"},
                    }
                }
            )
        )
        return _write_yaml(tmp_path, _DOWNSTREAM_YAML)

    def test_uses_mapping_parses(self, tmp_path):
        exp = load_experiment(_write_yaml(tmp_path, _DOWNSTREAM_YAML))
        assert exp["uses"]["source"] == "walkup.carry.source"

    def test_string_uses_rejected_on_stageless(self, tmp_path):
        text = _DOWNSTREAM_YAML.replace(
            "uses:\n  source: walkup.carry.source\n"
            "  component_overrides: walkup.carry.final_component_texts\n",
            "uses: walkup.carry.source\n",
        )
        with pytest.raises(ValueError, match="is a mapping"):
            load_experiment(_write_yaml(tmp_path, text))

    def test_unknown_uses_slot_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unknown uses slot"):
            load_experiment(
                _write_yaml(tmp_path, _DOWNSTREAM_YAML.replace("source:", "src:", 1))
            )

    def test_bundle_source_conflicts_with_uses_source(self, tmp_path):
        text = _DOWNSTREAM_YAML.replace(
            "{name: cot, route: cot}", "{name: cot, route: cot, bundle_source: uniprot}"
        )
        with pytest.raises(ValueError, match="exactly one place"):
            load_experiment(_write_yaml(tmp_path, text))

    def test_condition_without_any_source_rejected(self, tmp_path):
        text = _MINIMAL_YAML.replace("{name: b, bundle_source: affinage}", "{name: b}")
        with pytest.raises(ValueError, match="exactly one place"):
            load_experiment(_write_yaml(tmp_path, text))

    def test_resolved_inputs_reach_every_condition(self, tmp_path, monkeypatch):
        path = self._setup(tmp_path, monkeypatch)
        state = run_experiment(path, dry_run=True)

        assert state["resolved"]["source"] == "affinage"
        assert state["resolved"]["component_overrides"] == {"CAT": "TUNED CAT TEXT"}
        # carry.source is the resolved input, not the winning condition.
        assert state["carry"]["source"] == "affinage"
        assert state["carry"]["mode"] == state["winner_condition"]
        # The carried CAT text reaches both routes' prompts (CAT is a shared slot).
        for d in state["runs"].values():
            prompts = (tmp_path / "m" / d / "prompts.jsonl").read_text()
            assert "TUNED CAT TEXT" in prompts
        # Both arms ran on the resolved source's evidence view.
        snap = json.loads(
            (tmp_path / "m" / state["runs"]["cot"] / "config_snapshot.yaml").read_text()
        )
        assert snap["experiment"]["bundle_source"] == "affinage"

    def test_source_kwarg_overrides_uses_resolution(self, tmp_path, monkeypatch):
        path = self._setup(tmp_path, monkeypatch)
        state = run_experiment(path, dry_run=True, source="uniprot")
        assert state["resolved"]["source"] == "uniprot"
        assert state["carry"]["source"] == "uniprot"


MODE_YAML = SOURCE_YAML.parent / "mode.yaml"


def test_mode_yaml_parses_as_the_full_delivery_x_mcp_matrix():
    exp = load_experiment(MODE_YAML)
    assert exp["uses"] == {
        "source": "walkup.carry.source",
        "component_overrides": "walkup.carry.final_component_texts",
    }
    conds = {c["name"]: c for c in exp["conditions"]}
    assert list(conds) == [
        "single_call", "cot", "stepwise",
        "single_call_lit", "cot_lit", "stepwise_lit",
        "single_call_litb", "cot_litb", "stepwise_litb",
    ]
    for delivery in ("single_call", "cot", "stepwise"):
        assert conds[delivery]["route"] == delivery
        assert conds[f"{delivery}_lit"]["route"] == f"{delivery}_mcp"
        litb = conds[f"{delivery}_litb"]
        assert litb["route"] == f"{delivery}_mcp"
        assert litb["component_overrides"]["LIT"].startswith("LITERATURE GAP-FILL")
    # One LITB text, anchored -- identical across the three deliveries.
    texts = {
        conds[f"{d}_litb"]["component_overrides"]["LIT"]
        for d in ("single_call", "cot", "stepwise")
    }
    assert len(texts) == 1
    assert exp["carry"] == ["source", "mode"]


# ---------------------------------------------------------------------------
# Order-variant conditions
# ---------------------------------------------------------------------------


def test_order_variant_condition_reorders_the_route():
    from benchmarks.workflow.bench_orderings import (  # noqa: E501
        apply_order_variant,
    )
    from benchmarks.workflow.bench_routes import (
        ROUTE_REGISTRY,
    )

    plain = bench_experiment._condition_route({"name": "O"}, "single_call")
    assert plain is ROUTE_REGISTRY["single_call"]
    reordered = bench_experiment._condition_route(
        {"name": "O1", "order_variant": "O1"}, "single_call"
    )
    expected = apply_order_variant(ROUTE_REGISTRY["single_call"], "O1")
    assert reordered.component_order == expected.component_order
    assert reordered.component_order != plain.component_order


def test_unknown_order_variant_rejected(tmp_path):
    text = _MINIMAL_YAML.replace(
        "{name: a, bundle_source: uniprot}",
        "{name: a, bundle_source: uniprot, order_variant: O9}",
    )
    with pytest.raises(ValueError, match="order_variant 'O9'"):
        load_experiment(_write_yaml(tmp_path, text))


ORDER_YAML = SOURCE_YAML.parent / "order.yaml"


def test_order_yaml_parses_with_the_variant_catalog():
    exp = load_experiment(ORDER_YAML)
    assert exp["uses"]["source"] == "mode.carry.source"
    assert [c["name"] for c in exp["conditions"]] == ["O", "O1", "O2", "O3", "O4"]
    assert all(c["order_variant"] == c["name"] for c in exp["conditions"])
    assert exp["carry"] == ["source", "order"]


def test_condition_route_extends_cot_with_phenotype_steps():
    from benchmarks.workflow.bench_experiment import _condition_route

    r = _condition_route({"name": "fs", "features": True, "strength": True}, "cot_mcp")
    assert r.features and r.strength
    assert r.component_order[-4:] == ("cFC", "cPC", "cPS", "cO")
    assert "LIT" in r.component_order
    plain = _condition_route({"name": "b"}, "cot_mcp")
    assert not plain.features and not plain.strength
    import pytest

    with pytest.raises(ValueError, match="cot route"):
        _condition_route({"name": "x", "features": True}, "single_call")


def test_feature_coherence_splits_on_either_separator():
    import pandas as pd

    from mozzarellm.utils.cluster_utils import compute_feature_coherence

    df = pd.DataFrame({"g": ["a", "b"], "up_features": ["f1; f2", "f1,f3"], "down_features": ["", ""]})
    block = compute_feature_coherence(df, ["up_features", "down_features"], gene_column="g")
    by_name = {r["feature"]: r["up_genes"] for r in block["features"]}
    assert by_name == {"f1": ["a", "b"], "f2": ["a"], "f3": ["b"]}
