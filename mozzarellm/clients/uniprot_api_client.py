from __future__ import annotations

import hashlib
import json
import os
import platform
import sqlite3
import time
from typing import Any
import pandas as pd

import requests
import warnings

##### CONSTANTS ##### (configurable)
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_TIME = 1.0
BASE_URL = "https://rest.uniprot.org"


class UniProtClient:
    """Configurable UniProt REST client with in-memory caching and backoff."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,  # timeout in seconds
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_time: float = DEFAULT_BACKOFF_TIME,  # initial backoff time in seconds
        cache_path: str | os.PathLike[str] | None = None,
        cache_ttl_seconds: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")  # defense: remove trailing slash
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff_time
        self._session = requests.Session()

        self._cache_ttl_seconds = cache_ttl_seconds
        if cache_path is None:
            cache_dir = self._default_cache_dir("mozzarellm")
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, "uniprot_cache.sqlite3")
        self._cache_path = os.fspath(cache_path) if cache_path is not None else None
        self._cache_conn = self._init_cache(self._cache_path) if self._cache_path else None

    ### CACHE METHODS ###
    @staticmethod
    def _default_cache_dir(app_name: str) -> str:
        system = platform.system()

        if system == "Windows":
            base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
            if base:
                return os.path.join(base, app_name)

        if system == "Darwin":
            return os.path.join(os.path.expanduser("~"), "Library", "Caches", app_name)

        base = os.environ.get("XDG_CACHE_HOME")  # linux
        if base:
            return os.path.join(base, app_name)
        return os.path.join(os.path.expanduser("~"), ".cache", app_name)

    @staticmethod
    def _init_cache(cache_path: str) -> sqlite3.Connection:
        """Initialize SQLite cache with optimized settings.

        Note: Uses Python's built-in sqlite3 module (SQLite 3.x).
        Developed with SQLite 3.37+. Requires SQLite 3.7.0+ for WAL mode.
        PRAGMA statements may change in future SQLite releases.
        """
        conn = sqlite3.connect(cache_path, timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")  # Requires SQLite 3.7.0+
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS uniprot_http_cache (
                cache_key TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                params_json TEXT,
                response_json TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        return conn

    def _make_cache_key(self, url: str, params: dict[str, Any] | None) -> str:
        params_json = json.dumps(params or {}, sort_keys=True, separators=(",", ":"))
        payload = f"{url}|{params_json}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _cache_get(self, cache_key: str) -> dict[str, Any] | None:
        if self._cache_conn is None:
            return None
        row = self._cache_conn.execute(
            "SELECT response_json, created_at FROM uniprot_http_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if not row:
            return None

        response_json, created_at = row
        if self._cache_ttl_seconds is not None:
            age = int(time.time()) - int(created_at)
            if age > self._cache_ttl_seconds:
                return None

        try:
            return json.loads(response_json)
        except Exception:
            return None

    def _cache_set(
        self, cache_key: str, url: str, params: dict[str, Any] | None, data: dict[str, Any]
    ) -> None:
        if self._cache_conn is None:
            return
        params_json = json.dumps(params or {}, sort_keys=True)
        response_json = json.dumps(data, sort_keys=True)
        self._cache_conn.execute(
            """
            INSERT INTO uniprot_http_cache (cache_key, url, params_json, response_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                response_json = excluded.response_json,
                created_at = excluded.created_at
            """,
            (cache_key, url, params_json, response_json, int(time.time())),
        )

    def clear_cache(self) -> int:
        """Delete all entries from the cache. Returns number of rows deleted."""
        if self._cache_conn is None:
            return 0
        cursor = self._cache_conn.execute("DELETE FROM uniprot_http_cache")
        return cursor.rowcount

    def evict_expired(self) -> int:
        """Delete entries older than the current TTL. Returns number of rows deleted.

        No-op if cache_ttl_seconds was not set (entries never expire).
        """
        if self._cache_conn is None or self._cache_ttl_seconds is None:
            return 0
        cutoff = int(time.time()) - self._cache_ttl_seconds
        cursor = self._cache_conn.execute(
            "DELETE FROM uniprot_http_cache WHERE created_at < ?", (cutoff,)
        )
        return cursor.rowcount

    ### HTTP METHODS ###
    def _get(self, *, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"

        cache_key = self._make_cache_key(url, params)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        for attempt in range(self.max_retries):
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict):
                    self._cache_set(cache_key, url, params, data)
                return data
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff * (2**attempt))
                else:
                    raise last_error
        raise RuntimeError("UniProt request failed")

    ### QUERY GENERATION ###
    @staticmethod
    def _generate_cluster_search_query(chunk: pd.DataFrame, stable_accession_col: str) -> str:
        """Generate a search query for a chunk of gene-level data."""
        if stable_accession_col in chunk.columns:
            chunk_genes = chunk[stable_accession_col].tolist()
        else:
            chunk_genes = chunk["accession"].tolist()
        return "(" + " OR ".join(chunk_genes) + ") AND reviewed:true"
        # TODO: handle edge case where chunk is >100 genes (search query limit)

    def fetch_functional_annotations(
        self,
        chunk: pd.DataFrame,
        stable_accession_col: str,
        *,
        limit: int = 100,  # entries per page of results (api default is 25)
    ) -> pd.DataFrame:
        """Search UniProtKB with a custom query.

        Args:
            query: UniProt search query (see https://www.uniprot.org/api-documentation/uniprotkb#operations-UniProtKB-searchCursor)
            limit: max number of results to return
        """

        query = self._generate_cluster_search_query(chunk, stable_accession_col)

        # _get() handles HTTP errors and retries internally
        response = self._get(
            path="/uniprotkb/search",
            params={
                "query": query,
                "format": "json",
                "size": str(limit),
                "fields": "cc_function",
            },
        )
        results = response.get("results") or []

        # Check if any entries were found
        num_accessions = len(chunk[stable_accession_col].unique())
        if not results:
            raise ValueError(
                f"No UniProt entries found for {num_accessions} accession(s). "
                f"Verify accessions are valid UniProt IDs."
            )

        # Extract functional annotations
        accession_function_annotations = []
        accessions_without_annotations = []

        for entry in results:
            accession = entry.get("primaryAccession")
            if not accession:
                continue

            function_texts: list[str] = []
            for comment in entry.get("comments") or []:
                if (comment or {}).get("commentType") != "FUNCTION":
                    continue
                for text in (comment or {}).get("texts") or []:
                    val = (text or {}).get("value")
                    if val:
                        function_texts.append(str(val))

            if not function_texts:
                accessions_without_annotations.append(str(accession))
                continue

            accession_function_annotations.append((str(accession), "\n".join(function_texts)))

        # Warn if some accessions lack functional annotations
        if accessions_without_annotations:
            warnings.warn(
                f"{len(accessions_without_annotations)} accession(s) found but lack FUNCTION annotations: "
                f"{accessions_without_annotations[:5]}"
                + ("..." if len(accessions_without_annotations) > 5 else "")
            )

        # Raise if no annotations found at all
        if not accession_function_annotations:
            raise ValueError(
                f"Found {len(results)} UniProt entries but none have FUNCTION annotations. "
                f"Accessions queried: {accessions_without_annotations[:10]}"
            )

        return pd.DataFrame(
            accession_function_annotations, columns=["accession", "UniProt_functional_annotation"]
        )

    def get_accession_from_gene_symbol(
        self,
        gene_symbol: str,
        organism_id: int,
        warn_on_fallback: bool = True,
    ) -> str:
        if not gene_symbol or gene_symbol == "nan":
            raise ValueError("Gene symbol is required")

        response = self._get(
            path="/uniprotkb/search",
            params={
                "query": f"(gene_exact:{gene_symbol}) AND (organism_id:{organism_id})",
                "format": "json",
                "size": "10",
                "fields": "accession,reviewed",
            },
        )

        results = response.get("results") or []
        if not results:
            warnings.warn(f"No UniProt entries found for gene_symbol '{gene_symbol}'")
            return ""

        reviewed = [r.get("primaryAccession") for r in results if r.get("reviewed")]
        reviewed = [str(a) for a in reviewed if a]
        if reviewed:
            return reviewed[0]  # explicit, best effort match
        if warn_on_fallback:
            warnings.warn(
                f"No reviewed UniProt entries found for gene_symbol '{gene_symbol}', falling back to unreviewed entries"
            )
        for r in results:
            acc = r.get("primaryAccession")
            if acc:
                return str(acc)
        return ""
