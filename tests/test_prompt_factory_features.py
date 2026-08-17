"""Tests for the feature-interpretation gate on the evidence-bundle user prompt.

When no feature-interpretation component is active, screen-derived feature data
(per-gene up/down features + phenotypic strength, and any aggregate
feature_coherence) must not reach the model.
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mozzarellm.utils.prompt_factory import (  # noqa: E402
    make_single_cluster_analysis_user_prompt,
    strip_feature_fields,
)


def test_strip_feature_fields_removes_features_keeps_annotation():
    bundle = {
        "screen_name": "s1",
        "cluster_id": "1",
        "feature_coherence": {"features": []},
        "cluster_genes": [
            {
                "gene_symbol": "G1",
                "up_features": "cell_x; cell_y",
                "down_features": "nucleus_z",
                "phenotypic_strength": "4.2",
                "UniProt_functional_annotation": "does a thing",
                "accession": "P1",
            }
        ],
    }
    strip_feature_fields(bundle)
    assert "feature_coherence" not in bundle
    g = bundle["cluster_genes"][0]
    assert "up_features" not in g and "down_features" not in g and "phenotypic_strength" not in g
    # non-feature evidence is preserved
    assert g["UniProt_functional_annotation"] == "does a thing"
    assert g["gene_symbol"] == "G1" and g["accession"] == "P1"


def test_user_prompt_gate(tmp_path):
    bundle = {
        "screen_name": "s1",
        "cluster_id": "1",
        "cluster_genes": [
            {
                "gene_symbol": "G1",
                "up_features": "cell_x; cell_y",
                "down_features": "nucleus_z",
                "phenotypic_strength": "4.2",
                "UniProt_functional_annotation": "does a thing",
            }
        ],
    }
    bp = tmp_path / "bundle.json"
    bp.write_text(json.dumps(bundle))
    m = {"1": bp}

    off = make_single_cluster_analysis_user_prompt("1", "s1", m, include_features=False)
    on = make_single_cluster_analysis_user_prompt("1", "s1", m, include_features=True)

    for field in ("up_features", "down_features", "phenotypic_strength"):
        assert field not in off, field
        assert field in on, field
    # annotation always present; default (no arg) strips
    assert "does a thing" in off and "does a thing" in on
    default = make_single_cluster_analysis_user_prompt("1", "s1", m)
    assert "up_features" not in default


def test_batch_request_gate(tmp_path):
    # The gate must also hold on the client's batch-request path, where it is a
    # per-request parameter (default: strip) rather than client state.
    from mozzarellm.clients.llm_api_clients import AnthropicClient

    bundle = {
        "screen_name": "s1",
        "cluster_id": "1",
        "cluster_genes": [
            {
                "gene_symbol": "G1",
                "up_features": "cell_x; cell_y",
                "phenotypic_strength": "4.2",
                "UniProt_functional_annotation": "does a thing",
            }
        ],
    }
    bp = tmp_path / "bundle.json"
    bp.write_text(json.dumps(bundle))
    client = AnthropicClient(model="claude-sonnet-5", api_key="test-key")

    def _bundle_text(request) -> str:
        return request["params"]["messages"][0]["content"][0]["text"]

    off = client._make_single_cluster_message_request("1", str(bp), "sys")
    on = client._make_single_cluster_message_request("1", str(bp), "sys", include_features=True)
    for field in ("up_features", "phenotypic_strength"):
        assert field not in _bundle_text(off), field
        assert field in _bundle_text(on), field
    assert "does a thing" in _bundle_text(off)
