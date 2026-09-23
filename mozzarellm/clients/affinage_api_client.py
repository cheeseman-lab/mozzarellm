from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import time
import warnings
from typing import Any

import pandas as pd
import requests

from mozzarellm.clients.sqlite_cache import (
    init_cache,
    resolve_cache_path,
    warn_cache_disabled,
)

##### CONSTANTS ##### (configurable)
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 4
DEFAULT_BACKOFF_TIME = 1.0
BASE_URL = "https://affinage.wi.mit.edu"
ANNOTATION_COL = "affinage_functional_annotation"
AUDIT_NOTE_COL = "affinage_audit_note"
REFUSAL_PREFIXES = ("Parse failed", "No mechanistic", "Insufficient")
CACHE_PATH_ENV = "MOZZARELLM_AFFINAGE_CACHE"
CACHE_LABEL = "Affinage"
CACHE_FILENAME = "affinage_cache.sqlite3"
CACHE_SCHEMA_VERSION = 1
DEFAULT_NEGATIVE_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # a week; see _cache_get
CACHE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS affinage_annotation_cache (
        cache_key TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        url TEXT NOT NULL,
        params_json TEXT,
        record_json TEXT NOT NULL,
        is_negative INTEGER NOT NULL,
        created_at INTEGER NOT NULL
    )
"""


def _warn_cache_disabled(cache_path: str | None, error: Exception) -> None:
    warn_cache_disabled(CACHE_LABEL, cache_path, CACHE_PATH_ENV, error)


# The API's audit subtypes rendered as plain language, complete against the
# R1-R10 rulebook (cheeseman-lab/affinage, affinage/audit_rules.py: IDENTITY
# R1-R4, GROUNDING R5-R8, BEHAVIOR R9-R10). Unknown subtypes fall back to the
# verbatim value; the API's other fields (tier, uniprot_band, rules_fired,
# issue) stay machine-side for future policy use.
_AUDIT_SUBTYPES = {
    # IDENTITY -- wrong gene, or wrong product of the right gene
    "corpus_ungrounded": "gene weakly grounded in its cited literature",
    "alias_collision": "cited literature may belong to a different gene sharing an alias",
    "cross_species_homonym": "narrative may describe a same-named gene from another species",
    "paralog": "narrative may describe a different gene (paralog/alias)",
    "alt_product": "narrative describes an alternative product of this locus",
    # GROUNDING -- narrative under-extracts or misuses evidence
    "recall_miss": "narrative missed evidence available in UniProt",
    "fabrication": "cites a reference not found in the literature",
    "truncated_citation": "carries a truncated or malformed citation",
    "uncited_synthesis": "some claims lack citations",
    # BEHAVIOR -- generation anomaly / failure
    "memorization_empty_corpus": "asserts findings without supporting literature",
    "memorization_wrong_corpus": "content not drawn from its cited literature",
    "model_safety_refusal": "no usable narrative (model refusal)",
    "parse_failure": "no usable narrative (output could not be parsed)",
    "unexpected_refusal": "no usable narrative (model declined despite evidence)",
}


def _audit_note(audit_flag) -> str:
    """Human-readable one-line note from the API's audit_flag; '' when unflagged.

    Built from the API's own human-facing fields (verdict + subtype), not the
    rule-coded `issue` string — e.g. "Evidence-grounding concern: some claims
    lack citations" rather than "R6: ... overlap = 0.00% (n_cited=3, ...)".
    """
    if not audit_flag:
        return ""
    if isinstance(audit_flag, dict):
        verdict = str(audit_flag.get("verdict") or "audit-flagged")
        subtype = audit_flag.get("subtype")
        if subtype:
            detail = _AUDIT_SUBTYPES.get(str(subtype), str(subtype))
            return f"{verdict}: {detail}"
        return str(audit_flag.get("issue") or verdict)
    return "audit-flagged"


class AffinageClient:
    """Affinage API client for mechanistic narratives, gated on the API's audit_flag.

    Mirrors UniProtClient.fetch_functional_annotations so it drops into the bundle
    builder's annotation step. Symbols are HGNC alias-resolved server-side; genes
    that are flagged, refused, or not found return no annotation (left to the
    caller's backup). Infrastructure failures (timeouts, 5xx after retries) are
    raised so the bundle builder can warn and fall through to UniProt rather than
    silently degrading.

    Records persist in an on-disk cache so the panels of one screen, which share
    a gene set and differ only in cluster assignment, ask affinage for each gene
    once rather than once per panel. The cache file is ``cache_path``, else
    ``$MOZZARELLM_AFFINAGE_CACHE`` (empty, "none" or "off" disables it), else the
    per-user cache directory. A cache that is corrupt, unwritable, or locked past
    the busy timeout is dropped with a loud warning and the lookups go to the API.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_time: float = DEFAULT_BACKOFF_TIME,  # initial backoff time in seconds
        cache_path: str | os.PathLike[str] | None = None,
        cache_ttl_seconds: int | None = None,
        negative_cache_ttl_seconds: int | None = DEFAULT_NEGATIVE_CACHE_TTL_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")  # defense: remove trailing slash
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff_time
        self._session = requests.Session()
        self._cache: dict[str, dict | None] = {}

        self._cache_ttl_seconds = cache_ttl_seconds
        self._negative_cache_ttl_seconds = negative_cache_ttl_seconds
        self._cache_path = self._resolve_cache_path(cache_path)
        self._cache_conn = self._init_cache(self._cache_path) if self._cache_path else None

    ### CACHE METHODS ###
    @staticmethod
    def _resolve_cache_path(cache_path: str | os.PathLike[str] | None) -> str | None:
        """Resolve the cache file: explicit argument, then $MOZZARELLM_AFFINAGE_CACHE, then default."""
        return resolve_cache_path(
            cache_path,
            label=CACHE_LABEL,
            env_var=CACHE_PATH_ENV,
            default_filename=CACHE_FILENAME,
        )

    @staticmethod
    def _init_cache(cache_path: str) -> sqlite3.Connection | None:
        """Open the on-disk cache, or return None when it cannot be used safely."""
        return init_cache(
            cache_path,
            label=CACHE_LABEL,
            env_var=CACHE_PATH_ENV,
            table_sql=CACHE_TABLE_SQL,
        )

    def _disable_cache(self, error: Exception) -> None:
        """Drop a cache that failed mid-run, so the remaining lookups still reach the API."""
        conn, self._cache_conn = self._cache_conn, None
        if conn is not None:
            with contextlib.suppress(sqlite3.Error):
                conn.close()
            _warn_cache_disabled(self._cache_path, error)

    def _request_params(self, symbol: str) -> dict[str, Any]:
        """Everything besides the URL that decides the answer for a symbol.

        The endpoint takes the symbol in the path and nothing else -- it is
        human-only and has no organism parameter -- so this carries just the
        record schema version, which invalidates the cache when the shape of a
        stored record changes. Any future query parameter belongs here.
        """
        return {"schema": CACHE_SCHEMA_VERSION}

    def _make_cache_key(self, url: str, params: dict[str, Any] | None) -> str:
        params_json = json.dumps(params or {}, sort_keys=True, separators=(",", ":"))
        payload = f"{url}|{params_json}".encode()
        return hashlib.sha256(payload).hexdigest()

    def _cache_get(self, cache_key: str) -> dict[str, Any] | None:
        """Stored ``{"record": record}`` envelope for a key, or None on a miss.

        The envelope is what distinguishes a cached negative (record is None:
        the gene has no affinage record, or the narrative was a refusal) from a
        key that was never fetched. Negatives are answers worth keeping -- they
        cost the same request as a hit -- but they are the ones that change when
        the affinage corpus is regenerated, so they expire on their own shorter
        TTL while a real narrative is kept until ``cache_ttl_seconds`` says
        otherwise.
        """
        if self._cache_conn is None:
            return None
        try:
            row = self._cache_conn.execute(
                "SELECT record_json, is_negative, created_at FROM affinage_annotation_cache "
                "WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        except sqlite3.Error as e:
            self._disable_cache(e)
            return None
        if not row:
            return None

        record_json, is_negative, created_at = row
        ttl = self._negative_cache_ttl_seconds if is_negative else self._cache_ttl_seconds
        if ttl is not None and int(time.time()) - int(created_at) > ttl:
            return None

        try:
            envelope = json.loads(record_json)
        except Exception:
            return None
        if not isinstance(envelope, dict) or "record" not in envelope:
            return None
        return envelope

    def _cache_set(
        self, cache_key: str, symbol: str, params: dict[str, Any] | None, record: dict | None
    ) -> None:
        if self._cache_conn is None:
            return
        params_json = json.dumps(params or {}, sort_keys=True)
        record_json = json.dumps({"record": record}, sort_keys=True)
        try:
            self._cache_conn.execute(
                """
                INSERT INTO affinage_annotation_cache
                    (cache_key, symbol, url, params_json, record_json, is_negative, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    record_json = excluded.record_json,
                    is_negative = excluded.is_negative,
                    created_at = excluded.created_at
                """,
                (
                    cache_key,
                    symbol,
                    self._symbol_url(symbol),
                    params_json,
                    record_json,
                    int(record is None),
                    int(time.time()),
                ),
            )
        except sqlite3.Error as e:
            self._disable_cache(e)

    def _symbol_url(self, symbol: str) -> str:
        return f"{self.base_url}/api/mechanistic_narrative/{symbol}"

    def _get(self, path: str) -> dict | None:
        """Fetch JSON from the API.

        Returns the parsed body on 200, or None on 404 (the symbol is not in the
        DB; do not retry). Other HTTP / transport errors retry with exponential
        backoff; if all retries fail the last exception is raised so the caller
        can distinguish "no record" from "API unreachable".
        """
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._session.get(url, timeout=self.timeout)
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff * (2**attempt))  # backoff for Railway cold-starts
                else:
                    raise last_error from None
        raise RuntimeError("Affinage request failed")

    def get_annotation_record(self, symbol: str) -> dict | None:
        """Annotation record for a gene, or None if no usable narrative.

        Returns {"narrative": str, "audit_note": str}, served from the in-memory
        dict, then the on-disk cache, then the API. Audit-flagged narratives are
        surfaced, not dropped — the flag is advisory and carried through as
        audit_note for downstream weighting. Only genuinely unusable responses
        (not found, empty, or a refusal-message narrative) return None, and that
        None is cached too. Infrastructure failures propagate from `_get` as
        exceptions and are never cached.
        """
        sym = str(symbol).strip()
        if sym in self._cache:
            return self._cache[sym]

        params = self._request_params(sym)
        cache_key = self._make_cache_key(self._symbol_url(sym), params)
        envelope = self._cache_get(cache_key)
        if envelope is not None:
            self._cache[sym] = envelope["record"]
            return envelope["record"]

        data = self._get(f"/api/mechanistic_narrative/{sym}")
        record: dict | None = None
        if data is None:
            warnings.warn(f"Affinage: gene not found for symbol {sym!r}", stacklevel=2)
        else:
            narrative = data.get("mechanistic_narrative")
            if not narrative:
                warnings.warn(f"Affinage: empty narrative for {sym!r}", stacklevel=2)
            elif narrative.startswith(REFUSAL_PREFIXES):
                warnings.warn(
                    f"Affinage: refusal narrative for {sym!r} ({narrative.split('.')[0]!r})",
                    stacklevel=2,
                )
            else:
                note = _audit_note(data.get("audit_flag"))
                if note:
                    warnings.warn(
                        f"Affinage: audit-flagged narrative for {sym!r} ({note}); surfacing",
                        stacklevel=2,
                    )
                record = {"narrative": narrative, "audit_note": note}

        self._cache[sym] = record
        self._cache_set(cache_key, sym, params, record)
        return record

    def fetch_functional_annotations(self, chunk: pd.DataFrame, gene_column: str) -> pd.DataFrame:
        """Return [gene_column, affinage_functional_annotation, affinage_audit_note].

        Rows are the genes with usable narratives, mirroring UniProtClient's
        "found only" return shape; the caller merges and fills any backup.
        affinage_audit_note carries the API's audit concern ('' when clean).
        Warns with a summary of omitted symbols; raises if no symbol returned a
        usable narrative (matches UniProt's behavior on a zero-result batch).
        """
        symbols = [
            str(s).strip()
            for s in chunk[gene_column].dropna().unique()
            if str(s).strip() and str(s).strip() != "NON_TARGETING_CONTROL"
        ]
        rows: list[tuple[str, str, str]] = []
        missing: list[str] = []
        for symbol in symbols:
            record = self.get_annotation_record(symbol)
            if record is None:
                missing.append(symbol)
            else:
                rows.append((symbol, record["narrative"], record["audit_note"]))

        if missing:
            warnings.warn(
                f"{len(missing)}/{len(symbols)} symbol(s) lack Affinage annotations: "
                f"{missing[:5]}" + ("..." if len(missing) > 5 else ""),
                stacklevel=2,
            )

        if not rows:
            raise ValueError(
                f"No usable Affinage narratives for {len(symbols)} symbol(s). "
                f"Symbols queried: {missing[:10]}"
            )

        return pd.DataFrame(rows, columns=[gene_column, ANNOTATION_COL, AUDIT_NOTE_COL])
