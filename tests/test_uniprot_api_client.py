"""
Unit tests for mozzarellm.clients.uniprot_api_client
"""

from __future__ import annotations

import os
import time
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from mozzarellm.clients.uniprot_api_client import (
    UniProtClient,
    DEFAULT_TIMEOUT,
    DEFAULT_MAX_RETRIES,
    DEFAULT_BACKOFF_TIME,
    BASE_URL,
)

####################### TEST CONSTANTS #######################
TEST_URL = "https://test.com/api"
CACHE_TTL = 3600
HUMAN_ORGANISM_ID = 9606
ACCESSION_COL = "accession"

####################### FIXTURES #######################


@pytest.fixture
def client(tmp_path):
    """UniProtClient backed by a fresh temporary SQLite cache, no TTL."""
    return UniProtClient(cache_path=tmp_path / "cache.db", cache_ttl_seconds=None)


@pytest.fixture
def client_with_ttl(tmp_path):
    """UniProtClient backed by a fresh temporary SQLite cache with a 1-hour TTL."""
    return UniProtClient(cache_path=tmp_path / "cache.db", cache_ttl_seconds=CACHE_TTL)


@pytest.fixture
def single_accession_chunk():
    """Single-row DataFrame for TP53 (P04637)."""
    return pd.DataFrame({ACCESSION_COL: ["P04637"], "gene_symbol": ["TP53"]})


@pytest.fixture
def two_accession_chunk():
    """Two-row DataFrame for TP53 (P04637) and BRCA1 (P38398)."""
    return pd.DataFrame({ACCESSION_COL: ["P04637", "P38398"], "gene_symbol": ["TP53", "BRCA1"]})


####################### HELPERS #######################


def _make_uniprot_entry(accession: str, function_text: str) -> dict:
    """Build a minimal UniProt API result entry for use in mocks."""
    return {
        "primaryAccession": accession,
        "comments": [{"commentType": "FUNCTION", "texts": [{"value": function_text}]}],
    }


# =============================================================================
# Test: Client Initialization
# =============================================================================


def test_client_initialization_defaults():
    """Test client initializes with default parameters"""
    with patch.object(UniProtClient, "_init_cache", return_value=None):
        client = UniProtClient()
        assert client.base_url == BASE_URL
        assert client.timeout == DEFAULT_TIMEOUT
        assert client.max_retries == DEFAULT_MAX_RETRIES
        assert client.backoff == DEFAULT_BACKOFF_TIME


def test_client_initialization_custom_params(tmp_path):
    """Test client accepts custom parameters"""
    cache_path = tmp_path / "custom_cache.db"
    client = UniProtClient(
        base_url=BASE_URL,
        timeout=60.0,
        max_retries=5,
        backoff_time=2.0,
        cache_path=cache_path,
        cache_ttl_seconds=CACHE_TTL,
    )
    assert client.base_url == BASE_URL
    assert client.timeout == 60.0
    assert client.max_retries == 5
    assert client.backoff == 2.0
    assert client._cache_ttl_seconds == CACHE_TTL


def test_client_strips_trailing_slash_from_base_url(tmp_path):
    """Test client removes trailing slash from base URL"""
    client = UniProtClient(base_url="https://rest.uniprot.org///", cache_path=tmp_path / "cache.db")
    assert client.base_url == "https://rest.uniprot.org"


# =============================================================================
# Test: Cache Directory Detection
#
# NOTE: These are best-effort unit tests. They mock platform.system() and environment variables to simulate each OS branch,
# but they do NOT verify that the returned path is valid or writable on the actual host OS. True cross-platform validation
# requires running the test suite on each target OS (Windows, macOS, Linux), which requires CI/CD (e.g. GitHub Actions matrix).
#
# What these tests DO guarantee:
#   - Each OS branch in _default_cache_dir() is reachable and returns the correct path structure given controlled env vars.
#   - The env var priority logic (LOCALAPPDATA > APPDATA, XDG > ~/.cache) is correctly implemented.
#   - os.path.join() is used throughout, so separators are OS-native at runtime even though the mocked base paths use forward
#     slashes here.
# =============================================================================


@pytest.mark.parametrize(
    "env_vars,expected_suffix",
    [
        (
            {"LOCALAPPDATA": os.path.join("C:", "Users", "Test", "AppData", "Local")},
            os.path.join("C:", "Users", "Test", "AppData", "Local", "mozzarellm"),
        ),
        (
            {"APPDATA": os.path.join("C:", "Users", "Test", "AppData", "Roaming")},
            os.path.join("C:", "Users", "Test", "AppData", "Roaming", "mozzarellm"),
        ),
    ],
)
@patch("platform.system", return_value="Windows")
def test_default_cache_dir_windows(mock_system, env_vars, expected_suffix):
    """Test Windows cache dir uses LOCALAPPDATA, falling back to APPDATA."""
    with patch.dict(os.environ, env_vars, clear=True):
        assert UniProtClient._default_cache_dir("mozzarellm") == expected_suffix


@pytest.mark.parametrize(
    "system,env_vars,expanduser_return,expected",
    [
        (
            "Darwin",
            {},
            "/Users/test",
            os.path.join("/Users/test", "Library", "Caches", "mozzarellm"),
        ),
        (
            "Linux",
            {"XDG_CACHE_HOME": "/home/test/.cache"},
            "/home/test",
            os.path.join("/home/test/.cache", "mozzarellm"),
        ),
        (
            "Linux",
            {},
            "/home/test",
            os.path.join("/home/test", ".cache", "mozzarellm"),
        ),
    ],
)
def test_default_cache_dir_unix(system, env_vars, expanduser_return, expected):
    """Test macOS and Linux cache dir detection, including XDG fallback."""
    with (
        patch("platform.system", return_value=system),
        patch.dict(os.environ, env_vars, clear=True),
        patch("os.path.expanduser", return_value=expanduser_return),
    ):
        assert UniProtClient._default_cache_dir("mozzarellm") == expected


# =============================================================================
# Test: SQLite Cache
# =============================================================================


def test_init_cache_creates_table(tmp_path):
    """Test cache initialization creates SQLite table"""
    cache_path = tmp_path / "test_cache.db"
    conn = UniProtClient._init_cache(str(cache_path))

    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='uniprot_http_cache'"
    )
    assert cursor.fetchone() is not None

    conn.close()


def test_cache_key_generation(client):
    """Test cache key is deterministic hash of URL and params"""
    key1 = client._make_cache_key("https://api.test.com/search", {"query": "TP53", "limit": "10"})
    key2 = client._make_cache_key("https://api.test.com/search", {"query": "TP53", "limit": "10"})
    key3 = client._make_cache_key("https://api.test.com/search", {"limit": "10", "query": "TP53"})

    assert key1 == key2  # Same params, same order
    assert key1 == key3  # Same params, different order (sorted)
    assert len(key1) == 64  # SHA256 hex digest


def test_cache_key_different_for_different_params(client):
    """Test cache key changes with different parameters"""
    key1 = client._make_cache_key("https://api.test.com/search", {"query": "P04637"})
    key2 = client._make_cache_key("https://api.test.com/search", {"query": "P38398"})

    assert key1 != key2


def test_cache_set_and_get(client):
    """Test caching stores and retrieves data"""
    params = {"query": "test"}
    data = {"results": [{"id": "123"}]}

    cache_key = client._make_cache_key(TEST_URL, params)
    client._cache_set(cache_key, TEST_URL, params, data)

    assert client._cache_get(cache_key) == data


def test_cache_get_returns_none_for_missing_key(client):
    """Test cache returns None for non-existent key"""
    assert client._cache_get("nonexistent_key") is None


def test_cache_respects_ttl(client_with_ttl):
    """Test cache expires entries after TTL using timestamp backdating."""
    params = {"query": "test"}
    data = {"results": []}

    cache_key = client_with_ttl._make_cache_key(TEST_URL, params)
    client_with_ttl._cache_set(cache_key, TEST_URL, params, data)

    assert client_with_ttl._cache_get(cache_key) == data

    # Backdate entry to 2 hours ago (past the 1-hour TTL)
    client_with_ttl._cache_conn.execute(
        "UPDATE uniprot_http_cache SET created_at = ? WHERE cache_key = ?",
        (int(time.time()) - 7200, cache_key),
    )

    assert client_with_ttl._cache_get(cache_key) is None


def test_cache_no_ttl_never_expires(client):
    """Test cache without TTL never expires (backdated entry still returned)."""
    data = {"results": []}
    cache_key = client._make_cache_key(TEST_URL, {})
    client._cache_set(cache_key, TEST_URL, {}, data)

    # Backdate to 1 year ago — should still be returned with no TTL
    client._cache_conn.execute(
        "UPDATE uniprot_http_cache SET created_at = ? WHERE cache_key = ?",
        (int(time.time()) - 365 * 24 * 3600, cache_key),
    )

    assert client._cache_get(cache_key) == data


# =============================================================================
# Test: Cache Management (clear_cache / evict_expired)
# =============================================================================


def test_clear_cache_removes_all_entries(client):
    """Test clear_cache deletes all rows and returns count"""
    for i in range(3):
        url = f"{TEST_URL}/{i}"
        key = client._make_cache_key(url, {})
        client._cache_set(key, url, {}, {"id": i})

    assert client.clear_cache() == 3

    total = client._cache_conn.execute("SELECT COUNT(*) FROM uniprot_http_cache").fetchone()[0]
    assert total == 0


def test_clear_cache_returns_zero_when_empty(client):
    """Test clear_cache returns 0 on an already-empty cache"""
    assert client.clear_cache() == 0


def test_clear_cache_no_op_without_connection(client):
    """Test clear_cache returns 0 when _cache_conn is None (patched out)"""
    client._cache_conn = None
    assert client.clear_cache() == 0


def test_evict_expired_removes_stale_entries(client_with_ttl):
    """Test evict_expired deletes only entries older than TTL.

    Uses direct timestamp manipulation instead of time.sleep for speed and determinism.
    """
    url_stale = f"{TEST_URL}/old"
    key_stale = client_with_ttl._make_cache_key(url_stale, {})
    client_with_ttl._cache_set(key_stale, url_stale, {}, {"stale": True})

    # Backdate the stale entry to 2 hours ago (past the 1-hour TTL)
    client_with_ttl._cache_conn.execute(
        "UPDATE uniprot_http_cache SET created_at = ? WHERE cache_key = ?",
        (int(time.time()) - 7200, key_stale),
    )

    url_fresh = f"{TEST_URL}/fresh"
    key_fresh = client_with_ttl._make_cache_key(url_fresh, {})
    client_with_ttl._cache_set(key_fresh, url_fresh, {}, {"fresh": True})

    assert client_with_ttl.evict_expired() == 1

    remaining = client_with_ttl._cache_conn.execute(
        "SELECT COUNT(*) FROM uniprot_http_cache"
    ).fetchone()[0]
    assert remaining == 1

    row = client_with_ttl._cache_conn.execute("SELECT cache_key FROM uniprot_http_cache").fetchone()
    assert row[0] == key_fresh


def test_evict_expired_no_op_without_ttl(client):
    """Test evict_expired is a no-op when no TTL is configured"""
    key = client._make_cache_key(TEST_URL, {})
    client._cache_set(key, TEST_URL, {}, {"data": True})

    assert client.evict_expired() == 0

    remaining = client._cache_conn.execute("SELECT COUNT(*) FROM uniprot_http_cache").fetchone()[0]
    assert remaining == 1


def test_evict_expired_no_op_without_connection(client):
    """Test evict_expired returns 0 when _cache_conn is None (patched out)"""
    client._cache_conn = None
    assert client.evict_expired() == 0


def test_evict_expired_returns_zero_when_nothing_stale(client_with_ttl):
    """Test evict_expired returns 0 when all entries are within TTL"""
    key = client_with_ttl._make_cache_key(TEST_URL, {})
    client_with_ttl._cache_set(key, TEST_URL, {}, {"data": True})

    assert client_with_ttl.evict_expired() == 0


# =============================================================================
# Test: HTTP GET with Retry
# =============================================================================


def test_get_success_on_first_attempt(client):
    """Test _get succeeds on first attempt and caches result"""
    expected = {"results": [{"id": "123"}]}
    mock_response = Mock()
    mock_response.json.return_value = expected
    mock_response.raise_for_status = Mock()

    with patch.object(client._session, "get", return_value=mock_response):
        result = client._get(path="/test", params={"query": "test"})

    assert result == expected

    cache_key = client._make_cache_key("https://rest.uniprot.org/test", {"query": "test"})
    assert client._cache_get(cache_key) == expected


def test_get_retries_on_failure(tmp_path):
    """Test _get retries with exponential backoff on failure"""
    client = UniProtClient(
        cache_path=tmp_path / "cache.db", max_retries=3, backoff_time=0.1
    )  # custom params — not using fixture

    call_count = 0

    def mock_get(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise requests.exceptions.RequestException("Temporary error")
        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status = Mock()
        return mock_response

    with patch.object(client._session, "get", side_effect=mock_get):
        start = time.time()
        result = client._get(path="/test")
        elapsed = time.time() - start

    assert result == {"success": True}
    assert call_count == 3
    # Should have backoff delays: 0.1 * 2^0 + 0.1 * 2^1 = 0.3s
    assert elapsed >= 0.3


def test_get_raises_after_max_retries(client):
    """Test _get raises exception after exhausting retries"""
    with patch.object(
        client._session, "get", side_effect=requests.exceptions.RequestException("Persistent error")
    ):
        with pytest.raises(requests.exceptions.RequestException, match="Persistent error"):
            client._get(path="/test")


def test_get_uses_cache_if_available(client):
    """Test _get returns cached result without making HTTP request"""
    params = {"query": "cached"}
    full_url = "https://rest.uniprot.org/test"
    cache_key = client._make_cache_key(full_url, params)
    client._cache_set(cache_key, full_url, params, {"cached": True})

    with patch.object(client._session, "get") as mock_get:
        result = client._get(path="/test", params=params)

    assert result == {"cached": True}
    mock_get.assert_not_called()


# =============================================================================
# Test: Accession Lookup
# =============================================================================


def test_get_accession_from_gene_symbol_reviewed(client):
    """Test accession lookup prefers reviewed entries"""
    mock_response = {
        "results": [
            {"primaryAccession": "Q9UNREVIEWED", "reviewed": False},
            {"primaryAccession": "P04637", "reviewed": True},
        ]
    }

    with patch.object(client, "_get", return_value=mock_response):
        accession = client.get_accession_from_gene_symbol("TP53", organism_id=HUMAN_ORGANISM_ID)

    assert accession == "P04637"


def test_get_accession_from_gene_symbol_unreviewed_fallback(client):
    """Test accession lookup falls back to unreviewed if no reviewed entries"""
    mock_response = {"results": [{"primaryAccession": "Q9UNREVIEWED", "reviewed": False}]}

    with patch.object(client, "_get", return_value=mock_response):
        with pytest.warns(UserWarning, match="No reviewed UniProt entries found"):
            accession = client.get_accession_from_gene_symbol(
                "NOVEL_GENE", organism_id=HUMAN_ORGANISM_ID
            )

    assert accession == "Q9UNREVIEWED"


def test_get_accession_from_gene_symbol_no_results(client):
    """Test accession lookup returns empty string when no results"""
    with patch.object(client, "_get", return_value={"results": []}):
        with pytest.warns(UserWarning, match="No UniProt entries found"):
            accession = client.get_accession_from_gene_symbol(
                "NONEXISTENT", organism_id=HUMAN_ORGANISM_ID
            )

    assert accession == ""


def test_get_accession_from_gene_symbol_empty_input_raises(client):
    """Test accession lookup raises on empty gene symbol"""
    with pytest.raises(ValueError, match="Gene symbol is required"):
        client.get_accession_from_gene_symbol("", organism_id=HUMAN_ORGANISM_ID)

    with pytest.raises(ValueError, match="Gene symbol is required"):
        client.get_accession_from_gene_symbol("nan", organism_id=HUMAN_ORGANISM_ID)


def test_get_accession_suppresses_warning_when_requested(client):
    """Test accession lookup can suppress fallback warning"""
    mock_response = {"results": [{"primaryAccession": "Q9UNREVIEWED", "reviewed": False}]}

    with patch.object(client, "_get", return_value=mock_response):
        accession = client.get_accession_from_gene_symbol(
            "NOVEL_GENE", organism_id=HUMAN_ORGANISM_ID, warn_on_fallback=False
        )

    assert accession == "Q9UNREVIEWED"


# =============================================================================
# Test: Functional Annotation Fetching
# =============================================================================


def test_fetch_functional_annotations_success(client, two_accession_chunk):
    """Test fetching functional annotations for gene cluster"""
    mock_response = {
        "results": [
            _make_uniprot_entry("P04637", "Acts as a tumor suppressor"),
            _make_uniprot_entry("P38398", "DNA repair protein"),
        ]
    }

    with patch.object(client, "_get", return_value=mock_response):
        result = client.fetch_functional_annotations(two_accession_chunk, ACCESSION_COL)

    assert len(result) == 2
    assert ACCESSION_COL in result.columns
    assert "UniProt_functional_annotation" in result.columns
    assert result.iloc[0][ACCESSION_COL] == "P04637"
    assert "tumor suppressor" in result.iloc[0]["UniProt_functional_annotation"]


def test_fetch_functional_annotations_no_results_raises(client):
    """Test fetching annotations raises when no results found"""
    chunk = pd.DataFrame({ACCESSION_COL: ["FAKE123"]})

    with patch.object(client, "_get", return_value={"results": []}):
        with pytest.raises(ValueError, match="No UniProt entries found"):
            client.fetch_functional_annotations(chunk, ACCESSION_COL)


def test_generate_cluster_search_query():
    """Test cluster search query generation"""
    chunk = pd.DataFrame({ACCESSION_COL: ["P04637", "P38398", "Q9Y6K9"]})

    query = UniProtClient._generate_cluster_search_query(chunk, ACCESSION_COL)

    assert query == "(P04637 OR P38398 OR Q9Y6K9) AND reviewed:true"


def test_fetch_functional_annotations_filters_non_function_comments(client, single_accession_chunk):
    """Test annotation fetching ignores non-FUNCTION comments"""
    mock_response = {
        "results": [
            {
                "primaryAccession": "P04637",
                "comments": [
                    {"commentType": "SUBCELLULAR LOCATION", "texts": [{"value": "Nucleus"}]},
                    {"commentType": "FUNCTION", "texts": [{"value": "Tumor suppressor"}]},
                ],
            }
        ]
    }

    with patch.object(client, "_get", return_value=mock_response):
        result = client.fetch_functional_annotations(single_accession_chunk, ACCESSION_COL)

    assert len(result) == 1
    annotation = result.iloc[0]["UniProt_functional_annotation"]
    assert "Tumor suppressor" in annotation
    assert "Nucleus" not in annotation


def test_fetch_functional_annotations_combines_multiple_function_texts(
    client, single_accession_chunk
):
    """Test annotation fetching combines multiple FUNCTION texts"""
    mock_response = {
        "results": [
            {
                "primaryAccession": "P04637",
                "comments": [
                    {
                        "commentType": "FUNCTION",
                        "texts": [{"value": "First function"}, {"value": "Second function"}],
                    }
                ],
            }
        ]
    }

    with patch.object(client, "_get", return_value=mock_response):
        result = client.fetch_functional_annotations(single_accession_chunk, ACCESSION_COL)

    annotation = result.iloc[0]["UniProt_functional_annotation"]
    assert "First function" in annotation
    assert "Second function" in annotation
    assert "\n" in annotation  # Should be joined with newline
