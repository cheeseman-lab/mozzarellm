"""
Unit tests for mozzarellm.clients.affinage_api_client
"""

from __future__ import annotations

import os
import sqlite3
import time
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from mozzarellm.clients.affinage_api_client import (
    ANNOTATION_COL,
    AUDIT_NOTE_COL,
    CACHE_PATH_ENV,
    AffinageClient,
)
from mozzarellm.clients.sqlite_cache import CACHE_BUSY_TIMEOUT_MS

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
def client(tmp_path):
    """Client backed by a fresh temporary SQLite cache, so tests never touch the user's."""
    return AffinageClient(max_retries=3, backoff_time=0.0, cache_path=tmp_path / "cache.db")


# =============================================================================
# get_annotation_record: success + soft-failure paths
# =============================================================================


def test_get_annotation_record_success_returns_narrative(client):
    payload = {"gene": "TP53", "mechanistic_narrative": "real narrative", "audit_flag": None}
    with patch.object(client._session, "get", return_value=_mock_response(json_data=payload)):
        assert client.get_annotation_record("TP53")["narrative"] == "real narrative"


def test_get_annotation_record_404_returns_none_and_warns(client):
    with (
        patch.object(client._session, "get", return_value=_mock_response(status_code=404)),
        pytest.warns(UserWarning, match="gene not found"),
    ):
        assert client.get_annotation_record("NOTAREALGENE") is None


def test_get_annotation_record_audit_flagged_surfaces_narrative_and_warns(client):
    payload = {"gene": "X", "mechanistic_narrative": "anything", "audit_flag": True}
    with (
        patch.object(client._session, "get", return_value=_mock_response(json_data=payload)),
        pytest.warns(UserWarning, match="audit-flagged"),
    ):
        assert client.get_annotation_record("X")["narrative"] == "anything"


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
        assert client.get_annotation_record("X") is None


def test_get_annotation_empty_narrative_returns_none_and_warns(client):
    payload = {"gene": "X", "mechanistic_narrative": None, "audit_flag": None}
    with (
        patch.object(client._session, "get", return_value=_mock_response(json_data=payload)),
        pytest.warns(UserWarning, match="empty narrative"),
    ):
        assert client.get_annotation_record("X") is None


# =============================================================================
# get_annotation_record: infra failure raises (does not silently degrade)
# =============================================================================


def test_get_annotation_5xx_raises_after_retries(client):
    with (
        patch.object(client._session, "get", return_value=_mock_response(status_code=503)),
        pytest.raises(requests.HTTPError),
    ):
        client.get_annotation_record("TP53")


def test_get_annotation_connection_error_raises_after_retries(client):
    with (
        patch.object(client._session, "get", side_effect=requests.ConnectionError("network down")),
        pytest.raises(requests.ConnectionError),
    ):
        client.get_annotation_record("TP53")


def test_404_does_not_retry(client):
    mock_get = Mock(return_value=_mock_response(status_code=404))
    with patch.object(client._session, "get", mock_get), pytest.warns(UserWarning):
        client.get_annotation_record("NOPE")
    assert mock_get.call_count == 1


def test_5xx_retries_then_raises(client):
    mock_get = Mock(return_value=_mock_response(status_code=503))
    with patch.object(client._session, "get", mock_get), pytest.raises(requests.HTTPError):
        client.get_annotation_record("TP53")
    assert mock_get.call_count == client.max_retries


# =============================================================================
# Cache behavior
#
# NOTE: A screen's panels share one gene set and differ only in cluster assignment,
# so the same symbols were fetched once per panel. The disk cache makes the second
# and later panels -- and later processes -- free. As with the UniProt cache, an
# unusable cache must cost the cache, not the annotations.
# =============================================================================


class _BrokenConn:
    """Connection stand-in whose every statement fails the way a corrupt file does."""

    def execute(self, *args, **kwargs):
        raise sqlite3.DatabaseError("database disk image is malformed")

    def close(self):
        pass


def _new_client(tmp_path, **kwargs):
    return AffinageClient(
        max_retries=3, backoff_time=0.0, cache_path=tmp_path / "cache.db", **kwargs
    )


def test_cache_hit_skips_request_and_warning(client):
    payload = {"gene": "TP53", "mechanistic_narrative": "narrative", "audit_flag": None}
    mock_get = Mock(return_value=_mock_response(json_data=payload))
    with patch.object(client._session, "get", mock_get):
        assert client.get_annotation_record("TP53")["narrative"] == "narrative"
        assert client.get_annotation_record("TP53")["narrative"] == "narrative"
    assert mock_get.call_count == 1


def test_cache_is_nfs_safe(client, tmp_path):
    """Test the cache journals through a rollback journal and waits out other writers"""
    conn = sqlite3.connect(tmp_path / "cache.db")
    try:
        assert str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "delete"
    finally:
        conn.close()
    assert client._cache_conn.execute("PRAGMA busy_timeout").fetchone()[0] == CACHE_BUSY_TIMEOUT_MS


def test_disk_cache_serves_a_second_client_without_a_request(tmp_path):
    """Test a fresh process (fresh client, same cache file) makes no HTTP call"""
    payload = {
        "gene": "TP53",
        "mechanistic_narrative": "narrative",
        "audit_flag": {"verdict": "Identity concern", "subtype": "paralog"},
    }
    first = _new_client(tmp_path)
    with (
        patch.object(first._session, "get", return_value=_mock_response(json_data=payload)),
        pytest.warns(UserWarning, match="audit-flagged"),
    ):
        first.get_annotation_record("TP53")

    second = _new_client(tmp_path)
    mock_get = Mock(side_effect=AssertionError("cache miss: the API was called"))
    with patch.object(second._session, "get", mock_get):
        record = second.get_annotation_record("TP53")

    assert mock_get.call_count == 0
    assert record == {
        "narrative": "narrative",
        "audit_note": "Identity concern: narrative may describe a different gene (paralog/alias)",
    }


@pytest.mark.parametrize(
    "payload,status_code",
    [
        (None, 404),
        ({"gene": "X", "mechanistic_narrative": None, "audit_flag": None}, 200),
        (
            {"gene": "X", "mechanistic_narrative": "No mechanistic discoveries found."},
            200,
        ),
    ],
)
def test_negative_and_refusal_results_are_cached(tmp_path, payload, status_code):
    """Test a not-found, empty, or refusal answer is stored, not refetched next run"""
    first = _new_client(tmp_path)
    response = _mock_response(status_code=status_code, json_data=payload)
    with patch.object(first._session, "get", return_value=response), pytest.warns(UserWarning):
        assert first.get_annotation_record("X") is None

    second = _new_client(tmp_path)
    mock_get = Mock(side_effect=AssertionError("cache miss: the API was called"))
    with patch.object(second._session, "get", mock_get):
        assert second.get_annotation_record("X") is None
    assert mock_get.call_count == 0


def test_negative_results_expire_on_their_shorter_ttl(tmp_path):
    """Test a stale negative is refetched while a narrative under no TTL is not"""
    first = _new_client(tmp_path, negative_cache_ttl_seconds=3600)
    with (
        patch.object(first._session, "get", return_value=_mock_response(status_code=404)),
        pytest.warns(UserWarning, match="gene not found"),
    ):
        first.get_annotation_record("X")

    second = _new_client(tmp_path, negative_cache_ttl_seconds=3600)
    payload = {"gene": "X", "mechanistic_narrative": "now it has one", "audit_flag": None}
    mock_get = Mock(return_value=_mock_response(json_data=payload))
    with (
        patch.object(second._session, "get", mock_get),
        patch("time.time", return_value=time.time() + 7200),
    ):
        assert second.get_annotation_record("X")["narrative"] == "now it has one"
    assert mock_get.call_count == 1


def test_cache_key_covers_the_base_url(tmp_path):
    """Test a different API host is a different cache entry, not a stale hit"""
    payload = {"gene": "TP53", "mechanistic_narrative": "narrative", "audit_flag": None}
    first = _new_client(tmp_path)
    with patch.object(first._session, "get", return_value=_mock_response(json_data=payload)):
        first.get_annotation_record("TP53")

    other = AffinageClient(
        base_url="https://staging.example.org",
        max_retries=3,
        backoff_time=0.0,
        cache_path=tmp_path / "cache.db",
    )
    mock_get = Mock(return_value=_mock_response(json_data=payload))
    with patch.object(other._session, "get", mock_get):
        other.get_annotation_record("TP53")
    assert mock_get.call_count == 1


def test_corrupt_cache_file_degrades_to_no_cache(tmp_path):
    """Test a corrupt database is reported loudly and the client opens without a cache"""
    cache_path = tmp_path / "corrupt.db"
    cache_path.write_bytes(b"SQLite format 3\x00" + b"\x00" * 2048)

    with pytest.warns(UserWarning, match="unusable"):
        corrupt_client = AffinageClient(cache_path=cache_path)

    assert corrupt_client._cache_conn is None


def test_corrupt_cache_still_yields_annotations(client):
    """Test a corrupt cache falls through to the API instead of returning nothing"""
    client._cache_conn = _BrokenConn()
    payload = {"gene": "TP53", "mechanistic_narrative": "real narrative", "audit_flag": None}
    chunk = pd.DataFrame({GENE_COL: ["TP53"]})

    with (
        patch.object(client._session, "get", return_value=_mock_response(json_data=payload)),
        pytest.warns(UserWarning, match="unusable"),
    ):
        result = client.fetch_functional_annotations(chunk, GENE_COL)

    assert result.iloc[0][ANNOTATION_COL] == "real narrative"


def test_unwritable_cache_directory_degrades_to_no_cache(tmp_path):
    """Test a cache directory that cannot be created costs the cache, not the run"""
    with (
        patch("os.makedirs", side_effect=PermissionError("read-only filesystem")),
        pytest.warns(UserWarning, match="unusable"),
    ):
        unwritable_client = AffinageClient(cache_path=tmp_path / "nope" / "cache.db")

    assert unwritable_client._cache_path is None
    assert unwritable_client._cache_conn is None


def test_cache_path_comes_from_the_environment(tmp_path):
    """Test $MOZZARELLM_AFFINAGE_CACHE relocates the cache, e.g. onto node-local storage"""
    cache_path = tmp_path / "node_local" / "affinage.sqlite3"
    with patch.dict(os.environ, {CACHE_PATH_ENV: str(cache_path)}):
        env_client = AffinageClient()

    assert env_client._cache_path == str(cache_path)
    assert cache_path.exists()


@pytest.mark.parametrize("value", ["", "none", "OFF"])
def test_cache_can_be_switched_off_from_the_environment(value):
    """Test an empty/none/off env value runs with no cache at all"""
    with patch.dict(os.environ, {CACHE_PATH_ENV: value}):
        off_client = AffinageClient()

    assert off_client._cache_path is None
    assert off_client._cache_conn is None


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
