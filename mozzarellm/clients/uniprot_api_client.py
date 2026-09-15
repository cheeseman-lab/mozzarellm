from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import platform
import sqlite3
import time
import warnings
from typing import Any

import pandas as pd
import requests

##### CONSTANTS ##### (configurable)
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_TIME = 1.0
BASE_URL = "https://rest.uniprot.org"
CACHE_PATH_ENV = "MOZZARELLM_UNIPROT_CACHE"
CACHE_BUSY_TIMEOUT_MS = 60000
CACHE_DISABLING_VALUES = ("", "none", "off")


def _warn_cache_disabled(cache_path: str | None, error: Exception) -> None:
    """Say loudly, once per failure, that the cache is out of the loop.

    A cache that cannot be read must not become thousands of empty annotations:
    the lookups fall through to the API instead, and the operator is told why.
    """
    message = (
        f"mozzarellm UniProt cache at {cache_path} is unusable ({type(error).__name__}: {error}); "
        f"continuing without a cache -- lookups go to the UniProt API. Delete the file to rebuild "
        f"it, or set {CACHE_PATH_ENV} to a path on node-local storage."
    )
    logging.error(message)
    warnings.warn(message, stacklevel=3)


class UniProtClient:
    """Configurable UniProt REST client with an on-disk response cache and backoff.

    The cache file is ``cache_path``, else ``$MOZZARELLM_UNIPROT_CACHE`` (empty,
    "none" or "off" disables it), else the per-user cache directory. A cache that
    is corrupt, unwritable, or locked past the busy timeout is dropped with a loud
    warning and the lookups go to the API.
    """

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
        self._cache_path = self._resolve_cache_path(cache_path)
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
    def _resolve_cache_path(cache_path: str | os.PathLike[str] | None) -> str | None:
        """Resolve the cache file: explicit argument, then $MOZZARELLM_UNIPROT_CACHE, then default.

        Setting the env var to an empty value (or "none"/"off") turns the cache off
        outright; pointing it at node-local storage is what keeps concurrent cluster
        jobs off a shared network filesystem. A cache directory that cannot be created
        degrades to "no cache" rather than failing the run.
        """
        if cache_path is None:
            env_value = os.environ.get(CACHE_PATH_ENV)
            if env_value is not None:
                if env_value.strip().lower() in CACHE_DISABLING_VALUES:
                    return None
                cache_path = env_value.strip()
            else:
                cache_path = os.path.join(
                    UniProtClient._default_cache_dir("mozzarellm"), "uniprot_cache.sqlite3"
                )
        resolved = os.fspath(cache_path)
        parent = os.path.dirname(resolved)
        if parent:
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError as e:
                _warn_cache_disabled(resolved, e)
                return None
        return resolved

    @staticmethod
    def _init_cache(cache_path: str) -> sqlite3.Connection | None:
        """Open the on-disk cache, or return None when it cannot be used safely.

        The journal is the rollback journal, not WAL: WAL needs shared memory between
        the processes on one host and is documented as unsafe over a network
        filesystem, which is how concurrent annotation jobs corrupt a cache that lives
        on NFS. Setting the mode here also converts a cache left in WAL by an earlier
        version. An unwritable file, a failed open, or a database that fails its
        integrity check degrades to "no cache" instead of failing every lookup.

        Note: Uses Python's built-in sqlite3 module (SQLite 3.x).
        Developed with SQLite 3.37+. PRAGMA statements may change in future releases.
        """
        try:
            conn = sqlite3.connect(
                cache_path, timeout=CACHE_BUSY_TIMEOUT_MS / 1000.0, isolation_level=None
            )
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute(f"PRAGMA busy_timeout={CACHE_BUSY_TIMEOUT_MS}")
            check = conn.execute("PRAGMA quick_check(1)").fetchone()
            if not check or str(check[0]).lower() != "ok":
                raise sqlite3.DatabaseError(
                    f"integrity check failed: {check[0] if check else 'no result'}"
                )
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
        except (sqlite3.Error, OSError) as e:
            _warn_cache_disabled(cache_path, e)
            return None

    def _disable_cache(self, error: Exception) -> None:
        """Drop a cache that failed mid-run, so the remaining lookups still reach the API."""
        conn, self._cache_conn = self._cache_conn, None
        if conn is not None:
            with contextlib.suppress(sqlite3.Error):
                conn.close()
            _warn_cache_disabled(self._cache_path, error)

    def _make_cache_key(self, url: str, params: dict[str, Any] | None) -> str:
        params_json = json.dumps(params or {}, sort_keys=True, separators=(",", ":"))
        payload = f"{url}|{params_json}".encode()
        return hashlib.sha256(payload).hexdigest()

    def _cache_get(self, cache_key: str) -> dict[str, Any] | None:
        if self._cache_conn is None:
            return None
        try:
            row = self._cache_conn.execute(
                "SELECT response_json, created_at FROM uniprot_http_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        except sqlite3.Error as e:
            self._disable_cache(e)
            return None
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
        try:
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
        except sqlite3.Error as e:
            self._disable_cache(e)

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
                    raise last_error from None
        raise RuntimeError("UniProt request failed")

    ### QUERY GENERATION ###
    @staticmethod
    def _generate_cluster_search_query(chunk: pd.DataFrame, stable_accession_col: str) -> str:
        """Generate a search query for a chunk of gene-level data."""
        col = stable_accession_col if stable_accession_col in chunk.columns else "accession"
        chunk_genes = [
            str(a).strip()
            for a in chunk[col].tolist()
            if str(a).strip() and str(a).strip() != "NON_TARGETING_CONTROL"
        ]
        return "(" + " OR ".join(chunk_genes) + ") AND reviewed:true"

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

        col = stable_accession_col if stable_accession_col in chunk.columns else "accession"
        valid = chunk[
            chunk[col].map(
                lambda a: bool(str(a).strip()) and str(a).strip() != "NON_TARGETING_CONTROL"
            )
        ]

        # UniProt rejects very long OR queries, so batch into <=limit accessions per query.
        # _get() handles HTTP errors and retries internally.
        results = []
        for start in range(0, len(valid), limit):
            sub = valid.iloc[start : start + limit]
            query = self._generate_cluster_search_query(sub, stable_accession_col)
            response = self._get(
                path="/uniprotkb/search",
                params={
                    "query": query,
                    "format": "json",
                    "size": str(limit),
                    "fields": "cc_function",
                },
            )
            results.extend(response.get("results") or [])

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
                + ("..." if len(accessions_without_annotations) > 5 else ""),
                stacklevel=2,
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
            warnings.warn(f"No UniProt entries found for gene_symbol '{gene_symbol}'", stacklevel=2)
            return ""

        reviewed = [r.get("primaryAccession") for r in results if r.get("reviewed")]
        reviewed = [str(a) for a in reviewed if a]
        if reviewed:
            return reviewed[0]  # explicit, best effort match
        if warn_on_fallback:
            warnings.warn(
                f"No reviewed UniProt entries found for gene_symbol '{gene_symbol}', falling back to unreviewed entries",
                stacklevel=2,
            )
        for r in results:
            acc = r.get("primaryAccession")
            if acc:
                return str(acc)
        return ""
