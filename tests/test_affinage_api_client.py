"""
Unit tests for mozzarellm.clients.affinage_api_client
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from mozzarellm.clients.affinage_api_client import (
    ANNOTATION_COL,
    AUDIT_NOTE_COL,
    AffinageClient,
)

GENE_COL = "gene_symbol"


def _mock_response(*, status_code: int = 200, json_data: dict | None = None):
    resp = Mock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    if status_code >= 400 and status_code != 404:
        resp.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status_code}")
    else:
        resp.raise_for_status.return_value = None
    return resp


@pytest.fixture
def client():
    return AffinageClient(max_retries=3, backoff_time=0.0)


# =============================================================================
# get_annotation: success + soft-failure paths
# =============================================================================


def test_get_annotation_success_returns_narrative(client):
    payload = {"gene": "TP53", "mechanistic_narrative": "real narrative", "audit_flag": None}
    with patch.object(client._session, "get", return_value=_mock_response(json_data=payload)):
        assert client.get_annotation("TP53") == "real narrative"


def test_get_annotation_404_returns_none_and_warns(client):
    with (
        patch.object(client._session, "get", return_value=_mock_response(status_code=404)),
        pytest.warns(UserWarning, match="gene not found"),
    ):
        assert client.get_annotation("NOTAREALGENE") is None


def test_get_annotation_audit_flagged_surfaces_narrative_and_warns(client):
    payload = {"gene": "X", "mechanistic_narrative": "anything", "audit_flag": True}
    with (
        patch.object(client._session, "get", return_value=_mock_response(json_data=payload)),
        pytest.warns(UserWarning, match="audit-flagged"),
    ):
        assert client.get_annotation("X") == "anything"


def test_audit_note_renders_api_human_readable_fields():
    from mozzarellm.clients.affinage_api_client import _audit_note

    # Real API payload shape: verdict + subtype are the human-facing fields;
    # the rule-coded `issue` string must NOT be the rendered note.
    flag = {
        "gene": "PSMA4",
        "tier": "GROUNDING",
        "verdict": "Evidence-grounding concern",
        "subtype": "uncited_synthesis",
        "uniprot_band": "medium",
        "rules_fired": "R6,R8",
        "issue": "R6: narrative-cited PMIDs vs gene2pubmed overlap = 0.00%; R8: 1/3 claims uncited",
    }
    assert _audit_note(flag) == "Evidence-grounding concern: some claims lack citations"
    assert _audit_note({"verdict": "Identity concern", "subtype": "paralog"}) == (
        "Identity concern: narrative may describe a different gene (paralog/alias)"
    )
    # The map is complete against the R1-R10 rulebook's subtype vocabulary.
    from mozzarellm.clients.affinage_api_client import _AUDIT_SUBTYPES

    assert len(_AUDIT_SUBTYPES) == 14
    assert _audit_note({"verdict": "Model-behavior concern", "subtype": "parse_failure"}) == (
        "Model-behavior concern: no usable narrative (output could not be parsed)"
    )
    # Unknown subtype falls back to the verbatim value; no subtype falls back to issue.
    assert _audit_note({"verdict": "V", "subtype": "new_kind"}) == "V: new_kind"
    assert _audit_note({"issue": "raw detail"}) == "raw detail"
    assert _audit_note(True) == "audit-flagged"
    assert _audit_note(None) == ""


def test_get_annotation_refusal_prefix_returns_none_and_warns(client):
    payload = {"gene": "X", "mechanistic_narrative": "Insufficient data to build a narrative."}
    with (
        patch.object(client._session, "get", return_value=_mock_response(json_data=payload)),
        pytest.warns(UserWarning, match="refusal narrative"),
    ):
        assert client.get_annotation("X") is None


def test_get_annotation_empty_narrative_returns_none_and_warns(client):
    payload = {"gene": "X", "mechanistic_narrative": None, "audit_flag": None}
    with (
        patch.object(client._session, "get", return_value=_mock_response(json_data=payload)),
        pytest.warns(UserWarning, match="empty narrative"),
    ):
        assert client.get_annotation("X") is None


# =============================================================================
# get_annotation: infra failure raises (does not silently degrade)
# =============================================================================


def test_get_annotation_5xx_raises_after_retries(client):
    with (
        patch.object(client._session, "get", return_value=_mock_response(status_code=503)),
        pytest.raises(requests.HTTPError),
    ):
        client.get_annotation("TP53")


def test_get_annotation_connection_error_raises_after_retries(client):
    with (
        patch.object(client._session, "get", side_effect=requests.ConnectionError("network down")),
        pytest.raises(requests.ConnectionError),
    ):
        client.get_annotation("TP53")


def test_404_does_not_retry(client):
    mock_get = Mock(return_value=_mock_response(status_code=404))
    with patch.object(client._session, "get", mock_get), pytest.warns(UserWarning):
        client.get_annotation("NOPE")
    assert mock_get.call_count == 1


def test_5xx_retries_then_raises(client):
    mock_get = Mock(return_value=_mock_response(status_code=503))
    with patch.object(client._session, "get", mock_get), pytest.raises(requests.HTTPError):
        client.get_annotation("TP53")
    assert mock_get.call_count == client.max_retries


# =============================================================================
# Cache behavior
# =============================================================================


def test_cache_hit_skips_request_and_warning(client):
    payload = {"gene": "TP53", "mechanistic_narrative": "narrative", "audit_flag": None}
    mock_get = Mock(return_value=_mock_response(json_data=payload))
    with patch.object(client._session, "get", mock_get):
        assert client.get_annotation("TP53") == "narrative"
        assert client.get_annotation("TP53") == "narrative"
    assert mock_get.call_count == 1


# =============================================================================
# fetch_functional_annotations: aggregate behavior
# =============================================================================


def test_fetch_functional_annotations_returns_only_usable(client):
    payloads = {
        "TP53": _mock_response(
            json_data={"mechanistic_narrative": "real narrative", "audit_flag": None}
        ),
        "BRCA1": _mock_response(status_code=404),
    }

    def side_effect(url, **_):
        symbol = url.rsplit("/", 1)[-1]
        return payloads[symbol]

    chunk = pd.DataFrame({GENE_COL: ["TP53", "BRCA1"]})
    with (
        patch.object(client._session, "get", side_effect=side_effect),
        pytest.warns(UserWarning),
    ):
        result = client.fetch_functional_annotations(chunk, GENE_COL)
    assert list(result.columns) == [GENE_COL, ANNOTATION_COL, AUDIT_NOTE_COL]
    assert result[GENE_COL].tolist() == ["TP53"]


def test_fetch_functional_annotations_raises_when_all_missing(client):
    chunk = pd.DataFrame({GENE_COL: ["NOPE1", "NOPE2"]})
    with (
        patch.object(client._session, "get", return_value=_mock_response(status_code=404)),
        pytest.warns(UserWarning),
        pytest.raises(ValueError, match="No usable Affinage narratives"),
    ):
        client.fetch_functional_annotations(chunk, GENE_COL)


def test_fetch_functional_annotations_filters_non_targeting(client):
    payload = {"mechanistic_narrative": "narrative", "audit_flag": None}
    mock_get = Mock(return_value=_mock_response(json_data=payload))
    chunk = pd.DataFrame({GENE_COL: ["TP53", "NON_TARGETING_CONTROL", ""]})
    with patch.object(client._session, "get", mock_get):
        result = client.fetch_functional_annotations(chunk, GENE_COL)
    assert mock_get.call_count == 1
    assert result[GENE_COL].tolist() == ["TP53"]
