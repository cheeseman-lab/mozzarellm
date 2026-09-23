"""Unit tests for analyze_screen (stub client, no API)."""

import json
import shutil
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

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


def test_prepare_screen_bundles_rebuilds_a_partial_directory(tmp_path, monkeypatch):
    from mozzarellm.pipeline import screen_analysis

    bundle_dir = tmp_path / "s1_analysis" / "s1_evidence_bundles"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "s1__cluster_21__bundle.json").write_text("{}")

    built = []
    monkeypatch.setattr(
        screen_analysis, "get_or_append_stable_accession", lambda **kw: kw["cluster_df"]
    )

    def _build(**kw):
        built.append(sorted(kw["acc_cluster_df"]["cluster"].astype(str)))
        for c in built[-1]:
            (bundle_dir / f"s1__cluster_{c}__bundle.json").write_text("{}")

    monkeypatch.setattr(screen_analysis, "build_evidence_bundles", _build)

    table = pd.DataFrame({"cluster": ["21", "22"], "gene_symbol": ["RPL3", "RPL4"]})
    bundles = screen_analysis.prepare_screen_bundles(
        screen_name="s1", cluster_table=table, output_dir=tmp_path
    )
    assert built == [["21", "22"]]
    assert set(bundles) == {"21", "22"}


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


_CONCURRENT_IDS = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7"]


class _SlowStubClient:
    """Stub whose per-cluster latency is reversed, so completion order is not input order."""

    model = "stub-model"

    def __init__(self, cluster_ids, fail_on=None, delay=0.02):
        self.cluster_ids = list(cluster_ids)
        self.fail_on = fail_on or set()
        self.delay = delay
        self.calls = []
        self.completions = []
        self._lock = threading.Lock()

    def _cluster_of(self, user_prompt):
        return next(c for c in self.cluster_ids if f"cluster {c}" in user_prompt)

    def analyze(self, *, system_prompt, user_prompt, mode, mcp):
        cluster_id = self._cluster_of(user_prompt)
        with self._lock:
            self.calls.append(cluster_id)
        time.sleep(self.delay * (len(self.cluster_ids) - self.cluster_ids.index(cluster_id)))
        with self._lock:
            self.completions.append(cluster_id)
        if cluster_id in self.fail_on:
            raise RuntimeError(f"boom {cluster_id}")
        cost = 0.01 * (self.cluster_ids.index(cluster_id) + 1)
        parsed = dict(_PARSED, dominant_process=f"process {cluster_id}")
        return parsed, {"response_text": json.dumps(parsed), "cost_usd": cost, "elapsed_s": 1.0}


def _many_bundles(tmp_path, cluster_ids):
    bundles = {}
    for cid in cluster_ids:
        p = tmp_path / f"cluster_{cid}__bundle.json"
        p.write_text(
            json.dumps(
                {
                    "cluster_genes": [
                        {
                            "gene_symbol": "RPL3",
                            "UniProt_functional_annotation": "ribosomal protein",
                        },
                        {
                            "gene_symbol": "C1orf131",
                            "UniProt_functional_annotation": "",
                        },
                    ]
                }
            )
        )
        bundles[cid] = p
    return bundles


def _run(tmp_path, client, name, bundles, **kw):
    return analyze_screen(
        screen_name="s1",
        cluster_to_bundle_map=bundles,
        client=client,
        run_dir=tmp_path / name,
        screen_context_path=_context(tmp_path),
        mode="cot",
        **kw,
    )


def test_max_workers_one_is_the_sequential_path(tmp_path):
    bundles = _bundles(tmp_path)
    default = _run(tmp_path, _StubClient(), "default", bundles)
    explicit_client = _StubClient()
    explicit = _run(tmp_path, explicit_client, "explicit", bundles, max_workers=1)
    assert list(explicit["results"]) == list(default["results"]) == ["21", "37"]
    assert explicit["results"] == default["results"]
    assert explicit["errors"] == default["errors"] == {}
    assert explicit["resumed"] == default["resumed"] == []
    assert explicit["total_cost_usd"] == default["total_cost_usd"] == 0.02
    # max_workers=1 still calls the model once per cluster, in input order
    assert [c["user"].count("cluster") > 0 for c in explicit_client.calls] == [True, True]


def test_max_workers_rejects_zero(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="max_workers must be >= 1"):
        _run(tmp_path, _StubClient(), "bad", _bundles(tmp_path), max_workers=0)


def test_concurrent_run_matches_sequential_results_and_order(tmp_path):
    bundles = _many_bundles(tmp_path, _CONCURRENT_IDS)
    fail_on = {"c1", "c5"}
    serial_client = _SlowStubClient(_CONCURRENT_IDS, fail_on=fail_on, delay=0.0)
    serial = _run(tmp_path, serial_client, "serial", bundles, max_workers=1)
    pooled_client = _SlowStubClient(_CONCURRENT_IDS, fail_on=fail_on)
    pooled = _run(tmp_path, pooled_client, "pooled", bundles, max_workers=8)

    # The pool really did finish out of input order; the outputs do not show it
    assert pooled_client.completions != _CONCURRENT_IDS
    assert sorted(pooled_client.calls) == sorted(_CONCURRENT_IDS)
    expected = [c for c in _CONCURRENT_IDS if c not in fail_on]
    assert list(pooled["results"]) == list(serial["results"]) == expected
    assert pooled["results"] == serial["results"]
    assert list(pooled["errors"]) == list(serial["errors"]) == ["c1", "c5"]
    assert pooled["total_cost_usd"] == serial["total_cost_usd"]
    assert pooled["resumed"] == serial["resumed"] == []
    assert list(pooled["cluster_df"]["cluster_id"]) == list(serial["cluster_df"]["cluster_id"])
    assert pooled["gene_df"].equals(serial["gene_df"])
    for cid in _CONCURRENT_IDS:
        assert (tmp_path / "pooled" / "traces" / f"cluster_{cid}.json").exists()


def test_resume_under_concurrency(tmp_path):
    bundles = _many_bundles(tmp_path, _CONCURRENT_IDS)
    first = _SlowStubClient(_CONCURRENT_IDS)
    _run(tmp_path, first, "resumed", bundles, max_workers=4)
    assert sorted(first.calls) == sorted(_CONCURRENT_IDS)

    class _Never(_SlowStubClient):
        def analyze(self, **kw):
            raise AssertionError("resume must not call the model")

    again = _run(
        tmp_path,
        _Never(_CONCURRENT_IDS),
        "resumed",
        bundles,
        max_workers=4,
        resume=True,
    )
    assert again["resumed"] == _CONCURRENT_IDS
    assert list(again["results"]) == _CONCURRENT_IDS
    assert again["total_cost_usd"] == 0.0
    assert again["errors"] == {}


def test_rate_limited_cluster_is_retried_not_failed(tmp_path, monkeypatch):
    import mozzarellm.pipeline.screen_analysis as sa

    monkeypatch.setattr(sa, "_RATE_LIMIT_BACKOFF_S", 0.0)

    class _Throttled(_StubClient):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        def analyze(self, **kw):
            self.attempts += 1
            if self.attempts <= 2:
                raise RuntimeError("Error code: 429 - rate_limit_error")
            return super().analyze(**kw)

    client = _Throttled()
    out = _run(
        tmp_path,
        client,
        "throttled",
        {"21": _bundles(tmp_path)["21"]},
        max_workers=1,
    )
    assert client.attempts == 3
    assert out["errors"] == {} and list(out["results"]) == ["21"]


def test_rate_limit_retry_is_bounded_and_other_errors_are_not_retried(tmp_path, monkeypatch):
    import mozzarellm.pipeline.screen_analysis as sa

    monkeypatch.setattr(sa, "_RATE_LIMIT_BACKOFF_S", 0.0)
    bundles = {"21": _bundles(tmp_path)["21"]}

    class _AlwaysThrottled(_StubClient):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        def analyze(self, **kw):
            self.attempts += 1
            raise RuntimeError("overloaded_error")

    throttled = _AlwaysThrottled()
    out = _run(tmp_path, throttled, "always", bundles)
    assert throttled.attempts == sa._RATE_LIMIT_ATTEMPTS
    assert "overloaded_error" in out["errors"]["21"]

    class _Broken(_StubClient):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        def analyze(self, **kw):
            self.attempts += 1
            raise RuntimeError("bad request")

    broken = _Broken()
    out = _run(tmp_path, broken, "broken", bundles)
    assert broken.attempts == 1 and out["errors"] == {"21": "bad request"}


# The benchmark-selected build annotates from Affinage, so the entry point every
# caller uses has to be able to ask for it; "uniprot" stays the default.
class _StubUniProt:
    def get_accession_from_gene_symbol(self, *, gene_symbol, organism_id, warn_on_fallback):
        return f"ACC_{gene_symbol}"

    def fetch_functional_annotations(self, chunk, stable_accession_col):
        return pd.DataFrame(
            {
                stable_accession_col: chunk[stable_accession_col],
                "UniProt_functional_annotation": ["uniprot narrative"] * len(chunk),
            }
        )


class _StubAffinage:
    def fetch_functional_annotations(self, chunk, gene_column):
        return pd.DataFrame(
            {
                gene_column: chunk[gene_column],
                "affinage_functional_annotation": ["affinage narrative"] * len(chunk),
                "affinage_audit_note": [""] * len(chunk),
            }
        )


def _prepare_with_source(tmp_path, source):
    from mozzarellm.pipeline.screen_analysis import prepare_screen_bundles

    table = pd.DataFrame({"cluster": ["21"], "gene_symbol": ["RPL3"]})
    bundles = prepare_screen_bundles(
        screen_name="s1",
        cluster_table=table,
        output_dir=tmp_path,
        source=source,
        uniprot_client=_StubUniProt(),
        affinage_client=_StubAffinage(),
    )
    bundle = json.loads(Path(bundles["21"]).read_text(encoding="utf-8"))
    return bundle["cluster_genes"][0]


@pytest.mark.parametrize(
    "source,expected,absent",
    [
        ("uniprot", ["UniProt_functional_annotation"], ["affinage_functional_annotation"]),
        ("affinage", ["affinage_functional_annotation"], ["UniProt_functional_annotation"]),
        (
            "both",
            ["UniProt_functional_annotation", "affinage_functional_annotation"],
            [],
        ),
        (None, ["UniProt_functional_annotation"], ["affinage_functional_annotation"]),
    ],
)
def test_prepare_screen_bundles_selects_the_annotation_source(tmp_path, source, expected, absent):
    gene = _prepare_with_source(tmp_path, source) if source else _prepare_default(tmp_path)
    for key in expected:
        assert gene[key]
    for key in absent:
        assert key not in gene


def _prepare_default(tmp_path):
    """prepare_screen_bundles with no source argument -- the backward-compatible path."""
    from mozzarellm.pipeline.screen_analysis import prepare_screen_bundles

    table = pd.DataFrame({"cluster": ["21"], "gene_symbol": ["RPL3"]})
    bundles = prepare_screen_bundles(
        screen_name="s1",
        cluster_table=table,
        output_dir=tmp_path,
        uniprot_client=_StubUniProt(),
    )
    bundle = json.loads(Path(bundles["21"]).read_text(encoding="utf-8"))
    return bundle["cluster_genes"][0]


def test_prepare_screen_bundles_rejects_an_unknown_source(tmp_path):
    from mozzarellm.pipeline.screen_analysis import prepare_screen_bundles

    table = pd.DataFrame({"cluster": ["21"], "gene_symbol": ["RPL3"]})
    with pytest.raises(ValueError, match="Unknown annotation source 'uniport'"):
        prepare_screen_bundles(
            screen_name="s1", cluster_table=table, output_dir=tmp_path, source="uniport"
        )


def test_prepare_screen_bundles_does_not_backfill_across_sources(tmp_path):
    """A gene Affinage has nothing for stays empty; UniProt does not fill the gap."""
    from mozzarellm.pipeline.screen_analysis import prepare_screen_bundles

    class _EmptyAffinage:
        def fetch_functional_annotations(self, chunk, gene_column):
            raise ValueError("No usable Affinage narratives for 1 symbol(s).")

    table = pd.DataFrame({"cluster": ["21"], "gene_symbol": ["RPL3"]})
    bundles = prepare_screen_bundles(
        screen_name="s1",
        cluster_table=table,
        output_dir=tmp_path,
        source="affinage",
        uniprot_client=_StubUniProt(),
        affinage_client=_EmptyAffinage(),
    )
    gene = json.loads(Path(bundles["21"]).read_text(encoding="utf-8"))["cluster_genes"][0]
    assert "affinage_functional_annotation" not in gene
    assert "UniProt_functional_annotation" not in gene


def test_parse_repairs_a_missing_comma_and_a_trailing_comma():
    from mozzarellm.pipeline.literature_mcp import _parse_json_from_text

    text = '```json\n{\n  "a": [\n    "x"\n  ]\n  "b": {\n    "c": 1,\n  }\n}\n```'
    assert _parse_json_from_text(text) == {"a": ["x"], "b": {"c": 1}}
    stray = '{\n  "summary": "a ] in text",\n  ],\n  "b": 1\n}'
    assert _parse_json_from_text(stray) == {"summary": "a ] in text", "b": 1}


def test_call_timeout_grows_with_the_token_budget():
    from mozzarellm.pipeline.literature_mcp import PER_CALL_TIMEOUT_S, call_timeout_s

    assert call_timeout_s(4000) == PER_CALL_TIMEOUT_S
    assert call_timeout_s(64000) > 1500
