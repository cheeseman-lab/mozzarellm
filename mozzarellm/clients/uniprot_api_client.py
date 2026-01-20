from __future__ import annotations

import hashlib
import json
import os
import platform
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class UniProtHit:
    accession: str
    primary_gene: str | None = None
    protein_name: str | None = None


class UniProtClient:
    """Configurable UniProt REST client with in-memory caching and backoff."""

    def __init__(
        self,
        base_url: str = "https://rest.uniprot.org",
        timeout: float = 30.0,  # timeout in seconds
        max_retries: int = 3,
        backoff_time: float = 1.0,  # initial backoff time in seconds
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

    @staticmethod
    def _default_cache_dir(app_name: str) -> str:
        system = platform.system()

        if system == "Windows":
            base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
            if base:
                return os.path.join(base, app_name)

        if system == "Darwin":
            return os.path.join(os.path.expanduser("~"), "Library", "Caches", app_name)

        base = os.environ.get("XDG_CACHE_HOME")
        if base:
            return os.path.join(base, app_name)
        return os.path.join(os.path.expanduser("~"), ".cache", app_name)

    @staticmethod
    def _init_cache(cache_path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(cache_path, timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
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

    def _cache_set(self, cache_key: str, url: str, params: dict[str, Any] | None, data: dict[str, Any]) -> None:
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

    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
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

    @staticmethod
    def _parse_primary_gene(entry: dict[str, Any]) -> str | None:
        genes = entry.get("genes") or []
        if not genes:
            return None
        gene_name = (genes[0] or {}).get("geneName") or {}
        value = gene_name.get("value")
        return str(value) if value else None

    @staticmethod
    def _parse_protein_name(entry: dict[str, Any]) -> str | None:
        pd = entry.get("proteinDescription") or {}
        rec = pd.get("recommendedName") or {}
        full = rec.get("fullName") or {}
        value = full.get("value")
        return str(value) if value else None

    def search_by_gene_symbol(
        self,
        gene_symbol: str,
        *,
        organism_id: int | None = None,
        limit: int = 5,
    ) -> list[UniProtHit]:
        """Search UniProtKB for a gene symbol.

        Args:
            gene_symbol: e.g. "TP53"
            organism_id: optional NCBI taxonomy id (e.g. 9606 for human)
            limit: max number of results to return
        """
        q = f"(gene:{gene_symbol})"
        if organism_id is not None:
            q = f"{q} AND (organism_id:{organism_id})"

        data = self._get(
            "/uniprotkb/search",
            params={
                "query": q,
                "format": "json",
                "size": str(limit),
                "fields": "accession,protein_name,genes",
            },
        )

        results = data.get("results") or []
        hits: list[UniProtHit] = []
        for r in results:
            acc = r.get("primaryAccession")
            if not acc:
                continue
            hits.append(
                UniProtHit(
                    accession=str(acc),
                    primary_gene=self._parse_primary_gene(r),
                    protein_name=self._parse_protein_name(r),
                )
            )
        return hits

    def accessions_for_gene_symbol(
        self,
        gene_symbol: str,
        *,
        organism_id: int | None = None,
        limit: int = 5,
    ) -> list[str]:
        hits = self.search_by_gene_symbol(gene_symbol, organism_id=organism_id, limit=limit)
        accs = [h.accession for h in hits]
        return list(accs)

    def function_text_for_accession(self, accession: str) -> str | None:
        """Fetch functional annotation text (when available) for a UniProt accession."""
        data = self._get(
            f"/uniprotkb/{accession}",
            params={"format": "json"},
        )

        comments = data.get("comments") or []
        function_texts: list[str] = []
        for c in comments:
            if (c or {}).get("commentType") != "FUNCTION":
                continue
            for t in c.get("texts") or []:
                val = (t or {}).get("value")
                if val:
                    function_texts.append(str(val))

        out = "\n".join(function_texts) if function_texts else None
        return out
