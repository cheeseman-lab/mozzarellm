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
                    "cluster_genes": [
                        {
                            "gene_symbol": "RPL3",
                            "up_features": "nucleolar area up",
                            "UniProt_functional_annotation": "ribosomal protein",
                        }
                    ]
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
    # Features stay out of the prompt unless asked for.
    assert "up_features" not in client.calls[0]["user"]


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
    assert "up_features" in call["user"]  # phenotypic features reach the model
    assert "nucleolar area up" in call["user"]


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
