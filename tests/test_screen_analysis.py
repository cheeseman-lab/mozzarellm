"""Unit tests for analyze_screen (stub client, no API)."""

import json
import shutil
from pathlib import Path

import pandas as pd

from mozzarellm.pipeline.screen_analysis import analyze_screen

_CONTEXT = Path(__file__).resolve().parents[1] / "examples" / "ops" / "screen_context.json"


def _context(tmp_path):
    ctx = tmp_path / "screen_context.json"
    shutil.copy(_CONTEXT, ctx)
    return ctx


_PARSED = {
    "dominant_process": "ribosome biogenesis",
    "pathway_confidence": "High",
    "established_genes": ["RPL3"],
    "novel_role_genes": [{"gene": "C1orf131", "class": "PARTIAL_EVIDENCE", "rationale": "r"}],
    "uncharacterized_genes": [],
}


class _StubClient:
    model = "stub-model"

    def __init__(self, fail_on=None):
        self.fail_on = fail_on or set()
        self.calls = []

    def analyze(self, *, system_prompt, user_prompt, mode, mcp):
        self.calls.append({"system": system_prompt, "user": user_prompt})
        cluster_id = next(c for c in ("21", "37") if f"cluster {c}" in user_prompt)
        if cluster_id in self.fail_on:
            raise RuntimeError("boom")
        return dict(_PARSED), {"response_text": "{}", "cost_usd": 0.01, "elapsed_s": 1.0}


def _bundles(tmp_path):
    bundles = {}
    for cid in ("21", "37"):
        p = tmp_path / f"cluster_{cid}__bundle.json"
        p.write_text(
            json.dumps(
                {
                    "feature_coherence": {
                        "n_genes_in_cluster": 1,
                        "features": [{"feature": "nucleolar area", "n_up": 1, "frac_up": 1.0}],
                    },
                    "cluster_genes": [
                        {
                            "gene_symbol": "RPL3",
                            "up_features": "nucleolar area",
                            "UniProt_functional_annotation": "ribosomal protein",
                        }
                    ],
                }
            )
        )
        bundles[cid] = p
    return bundles


def test_analyze_screen_writes_traces_json_and_tables(tmp_path):
    client = _StubClient()
    out = analyze_screen(
        screen_name="s1",
        cluster_to_bundle_map=_bundles(tmp_path),
        client=client,
        run_dir=tmp_path / "run",
        screen_context_path=_context(tmp_path),
        mode="cot",
    )
    assert set(out["results"]) == {"21", "37"}
    assert (tmp_path / "run" / "traces" / "cluster_21.json").exists()
    assert (tmp_path / "run" / "s1_genes.csv").exists()
    assert (tmp_path / "run" / "s1_clusters.csv").exists()
    assert len(out["gene_df"]) == 4  # 2 genes x 2 clusters
    assert out["total_cost_usd"] == 0.02
    assert out["errors"] == {}
    # Auto mode: bundles carry the coherence table, so it reaches the prompt;
    # an explicit False strips it. Per-gene lists never reach the model.
    assert "feature_coherence" in client.calls[0]["user"]
    assert "up_features" not in client.calls[0]["user"]
    off = _StubClient()
    analyze_screen(
        screen_name="s1",
        cluster_to_bundle_map=_bundles(tmp_path),
        client=off,
        run_dir=tmp_path / "run_off",
        screen_context_path=_context(tmp_path),
        mode="cot",
        include_features=False,
    )
    assert "feature_coherence" not in off.calls[0]["user"]


def test_analyze_screen_feature_mode_feeds_features_and_cot_steps(tmp_path):
    client = _StubClient()
    analyze_screen(
        screen_name="s1",
        cluster_to_bundle_map=_bundles(tmp_path),
        client=client,
        run_dir=tmp_path / "run",
        screen_context_path=_context(tmp_path),
        mode="cot",
        include_features=True,
    )
    call = client.calls[0]
    assert "feature_coherence" in call["user"]  # the bounded table reaches the model
    assert "nucleolar area" in call["user"]
    assert "up_features" not in call["user"]


def test_feature_mode_rejected_outside_cot():
    import pytest

    with pytest.raises(ValueError, match="mode='cot'"):
        analyze_screen(
            screen_name="s1",
            cluster_to_bundle_map={},
            client=_StubClient(),
            run_dir="unused",
            screen_context_path=None,
            mode="standard",
            include_features=True,
        )


def test_one_failing_cluster_does_not_kill_the_run(tmp_path):
    client = _StubClient(fail_on={"21"})
    out = analyze_screen(
        screen_name="s1",
        cluster_to_bundle_map=_bundles(tmp_path),
        client=client,
        run_dir=tmp_path / "run",
        screen_context_path=_context(tmp_path),
    )
    assert list(out["errors"]) == ["21"]
    assert set(out["results"]) == {"37"}
    trace = json.loads((tmp_path / "run" / "traces" / "cluster_21.json").read_text())
    assert trace["error"] == "boom"


def test_original_df_metadata_merges_into_tables(tmp_path):
    original = pd.DataFrame({"cluster_id": ["21", "37"], "n_input_genes": [10, 12]})
    out = analyze_screen(
        screen_name="s1",
        cluster_to_bundle_map=_bundles(tmp_path),
        client=_StubClient(),
        run_dir=tmp_path / "run",
        screen_context_path=_context(tmp_path),
        original_df=original,
    )
    assert "n_input_genes" in out["cluster_df"].columns


def test_prepare_screen_bundles_reuses_cache(tmp_path):
    from mozzarellm.pipeline.screen_analysis import prepare_screen_bundles

    bundle_dir = tmp_path / "s1_analysis" / "s1_evidence_bundles"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "s1__cluster_21__bundle.json").write_text("{}")

    table = pd.DataFrame({"cluster": ["21"], "gene_symbol": ["RPL3"]})
    bundles = prepare_screen_bundles(
        screen_name="s1", cluster_table=table, output_dir=tmp_path
    )  # cache hit: no network
    assert "21" in bundles


def test_component_overrides_reach_the_system_prompt(tmp_path):
    client = _StubClient()
    analyze_screen(
        screen_name="s1",
        cluster_to_bundle_map=_bundles(tmp_path),
        client=client,
        run_dir=tmp_path / "run",
        screen_context_path=_context(tmp_path),
        mode="cot",
        component_overrides={"cGCR": "MY CUSTOM CATEGORIZATION RULES"},
    )
    assert "MY CUSTOM CATEGORIZATION RULES" in client.calls[0]["system"]


def test_coverage_is_bundle_grounded_not_self_reported(tmp_path):
    out = analyze_screen(
        screen_name="s1",
        cluster_to_bundle_map=_bundles(tmp_path),
        client=_StubClient(),
        run_dir=tmp_path / "run",
        screen_context_path=_context(tmp_path),
    )
    row = out["cluster_df"].iloc[0]
    # Bundle has 1 gene (RPL3); the stub classifies RPL3 + 1 hallucinated gene.
    assert row["n_genes"] == 1
    assert row["classification_completeness"] == 1.0


class _EmptyClient(_StubClient):
    def analyze(self, *, system_prompt, user_prompt, mode, mcp):
        self.calls.append({"system": system_prompt, "user": user_prompt})
        parsed = {
            "dominant_process": "ribosome biogenesis",  # a pathway call, yet no genes
            "pathway_confidence": "High",
            "established_genes": [],
            "novel_role_genes": [],
            "uncharacterized_genes": [],
        }
        return parsed, {"response_text": "{}", "cost_usd": 0.0}


def test_empty_classification_with_pathway_call_is_an_error(tmp_path):
    out = analyze_screen(
        screen_name="s1",
        cluster_to_bundle_map=_bundles(tmp_path),
        client=_EmptyClient(),
        run_dir=tmp_path / "run",
        screen_context_path=_context(tmp_path),
    )
    assert set(out["errors"]) == {"21", "37"}
    assert out["cluster_df"]["classification_completeness"].eq(0.0).all()
    assert (out["cluster_df"]["missed_genes"] == "RPL3").all()


class _AbstainClient(_StubClient):
    def analyze(self, *, system_prompt, user_prompt, mode, mcp):
        self.calls.append({})
        parsed = {
            "dominant_process": "No coherent biological pathway",
            "pathway_confidence": "Low",
            "established_genes": [],
            "novel_role_genes": [],
            "uncharacterized_genes": [],
        }
        return parsed, {"response_text": "{}", "cost_usd": 0.0}


def test_abstention_is_not_flagged_as_an_error(tmp_path):
    out = analyze_screen(
        screen_name="s1",
        cluster_to_bundle_map=_bundles(tmp_path),
        client=_AbstainClient(),
        run_dir=tmp_path / "run",
        screen_context_path=_context(tmp_path),
    )
    assert out["errors"] == {}


def test_screen_context_dict_replaces_path(tmp_path):
    """An in-memory context dict works without any JSON file on disk."""
    ctx = json.loads(_CONTEXT.read_text())
    out = analyze_screen(
        screen_name="s1",
        cluster_to_bundle_map=_bundles(tmp_path),
        client=_StubClient(),
        run_dir=tmp_path / "run",
        screen_context=ctx,
        mode="cot",
    )
    assert set(out["results"]) == {"21", "37"}


def test_latest_pointer_and_versioned_outputs(tmp_path):
    """A successful run writes latest.json; clusters.json carries schema_version;
    the cluster table carries the display summary."""
    out = analyze_screen(
        screen_name="s1",
        cluster_to_bundle_map=_bundles(tmp_path),
        client=_StubClient(),
        run_dir=tmp_path / "runs" / "run_01",
        screen_context_path=_context(tmp_path),
        mode="cot",
    )
    latest = json.loads((tmp_path / "runs" / "latest.json").read_text())
    assert latest["run_dir"] == "run_01"
    data = json.loads((tmp_path / "runs" / "run_01" / "s1_clusters.json").read_text())
    assert data["metadata"]["schema_version"] == "1"
    assert "summary" in out["cluster_df"].columns


def test_control_prefix_is_configurable():
    from mozzarellm.pipeline.bundle_builder import _lookup_accession

    assert (
        _lookup_accession("myctrl_g1_g1", 9606, False, None, control_prefix="myctrl_")
        == "NON_TARGETING_CONTROL"
    )
    assert _lookup_accession("nontargeting_g1_g1", 9606, False, None) == "NON_TARGETING_CONTROL"


def test_attach_strength_ranks_numeric_and_passthrough():
    import pandas as pd

    from mozzarellm.utils.cluster_utils import attach_strength_ranks

    df = pd.DataFrame({"gene_symbol": list("abcd"), "auc": [0.9, 0.5, None, 0.7]})
    out = attach_strength_ranks(df, "auc")  # the user's column keeps its name
    assert out["auc"].tolist()[:2] == ["1/3", "3/3"]
    assert pd.isna(out["auc"].iloc[2])  # unscored gene carries no rank
    assert out["auc"].iloc[3] == "2/3"  # raw values never enter bundles

    lower_is_stronger = attach_strength_ranks(
        pd.DataFrame({"g": ["x", "y"], "dist": [0.1, 0.9]}), "dist", higher_is_stronger=False
    )
    assert lower_is_stronger["dist"].tolist() == ["1/2", "2/2"]

    prerank = attach_strength_ranks(pd.DataFrame({"g": ["x"], "s": ["669/5299"]}), "s")
    assert prerank["s"].tolist() == ["669/5299"]


def test_strength_step_enters_prompt_iff_ranks_present(tmp_path):
    """cPS + ranks appear together (auto), and only then."""
    bundles = {}
    for cid, rank in (("21", "12/5299"), ("37", None)):
        p = tmp_path / f"cluster_{cid}__bundle.json"
        gene = {"gene_symbol": "RPL3", "UniProt_functional_annotation": "ribosomal protein"}
        bundle = {"cluster_genes": [gene]}
        if rank:
            gene["auc"] = rank
            bundle["phenotype_strength"] = {"column": "auc", "n_ranked": 1, "screen_size": 5299}
        p.write_text(json.dumps(bundle))
        bundles[cid] = p
    client = _StubClient()
    analyze_screen(
        screen_name="s1",
        cluster_to_bundle_map=bundles,
        client=client,
        run_dir=tmp_path / "run",
        screen_context_path=_context(tmp_path),
        mode="cot",
    )
    assert "PHENOTYPE STRENGTH" in client.calls[0]["system"]
    assert any("12/5299" in c["user"] for c in client.calls)
    # No feature data anywhere -> feature steps stay out even in auto mode.
    assert "FEATURE COHERENCE" not in client.calls[0]["system"]

    # Bundles without ranks -> no strength step, no dangling reference.
    plain = _StubClient()
    analyze_screen(
        screen_name="s1",
        cluster_to_bundle_map=_bundles(tmp_path),
        client=plain,
        run_dir=tmp_path / "run2",
        screen_context_path=_context(tmp_path),
        mode="cot",
        include_features=False,
    )
    assert "PHENOTYPE STRENGTH" not in plain.calls[0]["system"]


def test_include_strength_true_requires_data(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="include_strength=True"):
        analyze_screen(
            screen_name="s1",
            cluster_to_bundle_map=_bundles(tmp_path),
            client=_StubClient(),
            run_dir=tmp_path / "run",
            screen_context_path=_context(tmp_path),
            mode="cot",
            include_strength=True,
        )


def test_phenotype_strength_block_is_recall_table():
    import pandas as pd

    from mozzarellm.utils.cluster_utils import compute_phenotype_strength

    df = pd.DataFrame(
        {"gene_symbol": ["a", "b", "c", "d"], "auc": ["10/100", "90/100", None, "30/100"]}
    )
    block = compute_phenotype_strength(df, gene_column="gene_symbol", strength_column="auc")
    assert block["column"] == "auc"  # the table names the user's column
    assert block["n_ranked"] == 3 and block["screen_size"] == 100
    assert block["median_rank"] == "30/100"
    assert block["strongest_quartile_frac"] == round(1 / 3, 3)
    assert block["weakest_quartile_frac"] == round(1 / 3, 3)
    assert [g["gene"] for g in block["ranked_genes"]] == ["a", "d", "b"]
    empty = compute_phenotype_strength(
        df.iloc[[2]], gene_column="gene_symbol", strength_column="auc"
    )
    assert empty["n_ranked"] == 0


def test_strip_selects_each_phenotype_signal():
    from mozzarellm.prompts import strip_feature_fields

    # The user's column names come from the aggregates themselves.
    def bundle():
        return {
            "feature_coherence": {"columns": ["de_up", "de_down"]},
            "phenotype_strength": {"column": "edist"},
            "cluster_genes": [{"gene_symbol": "x", "de_up": "f", "de_down": "", "edist": "1/9"}],
        }

    b = bundle()
    strip_feature_fields(b, features=False, strength=True)
    assert "feature_coherence" in b and "phenotype_strength" not in b
    assert b["cluster_genes"][0].keys() == {"gene_symbol"}  # per-gene lists never survive
    b = bundle()
    strip_feature_fields(b, features=True, strength=False)
    assert "phenotype_strength" in b and "feature_coherence" not in b
    assert b["cluster_genes"][0].keys() == {"gene_symbol", "edist"}


def test_custom_component_order_is_used_verbatim(tmp_path):
    """A user's own chain replaces the default; phenotype steps enter iff included."""
    short = ["CAT", "SC", "cPH", "cGCR", "cO"]
    client = _StubClient()
    analyze_screen(
        screen_name="s1",
        cluster_to_bundle_map=_bundles(tmp_path),
        client=client,
        run_dir=tmp_path / "run",
        screen_context_path=_context(tmp_path),
        mode="cot",
        component_order=short,
    )
    system = client.calls[0]["system"]
    assert "STEP 5 - FINAL JSON OUTPUT" in system and "STEP 6" not in system
    assert "FEATURE COHERENCE" not in system  # bundles carry features, chain does not ask
    assert "feature_coherence" not in client.calls[0]["user"]

    class _TurnStub(_StubClient):
        def analyze(self, *, stepwise_turns, **kw):
            self.calls.append({"turns": stepwise_turns})
            return dict(_PARSED), {"response_text": "{}", "cost_usd": 0.01, "elapsed_s": 1.0}

    stepwise = _TurnStub()
    analyze_screen(
        screen_name="s1",
        cluster_to_bundle_map=_bundles(tmp_path),
        client=stepwise,
        run_dir=tmp_path / "run_sw",
        screen_context_path=_context(tmp_path),
        mode="stepwise",
        component_order=short,
    )
    turns = stepwise.calls[0]["turns"]
    assert [t["content"].split(" - ")[1].split(":")[0] for t in turns] == [
        "PATHWAY HYPOTHESIS",
        "GENE CATEGORIZATION (cite evidence)",
        "FINAL JSON OUTPUT",
    ]

    import pytest

    with pytest.raises(ValueError, match="include_strength=True"):
        analyze_screen(
            screen_name="s1",
            cluster_to_bundle_map=_bundles(tmp_path),
            client=_StubClient(),
            run_dir=tmp_path / "run_x",
            screen_context_path=_context(tmp_path),
            mode="cot",
            component_order=short[:-1] + ["cPS", "cO"],  # asks for strength the bundles lack
        )


def test_resume_reuses_recorded_responses(tmp_path):
    class _Recorded(_StubClient):  # a trace carries the real response text
        def analyze(self, **kw):
            parsed, raw = super().analyze(**kw)
            return parsed, {**raw, "response_text": json.dumps(parsed)}

    first = _Recorded()
    run_dir = tmp_path / "run"
    out = analyze_screen(
        screen_name="s1",
        cluster_to_bundle_map=_bundles(tmp_path),
        client=first,
        run_dir=run_dir,
        screen_context_path=_context(tmp_path),
        mode="cot",
    )
    assert len(first.calls) == 2 and out["resumed"] == []

    class _Never(_StubClient):
        def analyze(self, **kw):
            raise AssertionError("resume must not call the model")

    again = analyze_screen(
        screen_name="s1",
        cluster_to_bundle_map=_bundles(tmp_path),
        client=_Never(),
        run_dir=run_dir,
        screen_context_path=_context(tmp_path),
        mode="cot",
        resume=True,
    )
    assert sorted(again["resumed"]) == ["21", "37"]
    assert again["total_cost_usd"] == 0.0
    assert set(again["results"]) == {"21", "37"} and len(again["gene_df"]) == 4


def test_dry_run_writes_prompts_and_estimates_without_calls(tmp_path):
    class _Never(_StubClient):
        def analyze(self, **kw):
            raise AssertionError("dry_run must not call the model")

    out = analyze_screen(
        screen_name="s1",
        cluster_to_bundle_map=_bundles(tmp_path),
        client=_Never(),
        run_dir=tmp_path / "dry",
        screen_context_path=_context(tmp_path),
        mode="cot",
        dry_run=True,
    )
    est = out["estimates"]
    assert list(est["cluster_id"]) == ["21", "37"] and (est["est_input_tokens"] > 0).all()
    assert (tmp_path / "dry" / "prompts_used" / "system_prompt.txt").exists()
    assert (tmp_path / "dry" / "prompts_used" / "user_prompt_cluster_21.txt").exists()
    assert not (tmp_path / "latest.json").exists() and out["results"] == {}


def test_gene_table_carries_the_strength_rank(tmp_path):
    p = tmp_path / "cluster_21__bundle.json"
    p.write_text(
        json.dumps(
            {
                "phenotype_strength": {"column": "auc", "n_ranked": 1, "screen_size": 9},
                "cluster_genes": [
                    {"gene_symbol": "RPL3", "auc": "2/9", "UniProt_functional_annotation": "r"}
                ],
            }
        )
    )
    out = analyze_screen(
        screen_name="s1",
        cluster_to_bundle_map={"21": p},
        client=_StubClient(),
        run_dir=tmp_path / "run",
        screen_context_path=_context(tmp_path),
        mode="cot",
    )
    row = out["gene_df"][out["gene_df"]["gene"] == "RPL3"].iloc[0]
    assert row["phenotype_strength_rank"] == "2/9"
