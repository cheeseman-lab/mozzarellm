from __future__ import annotations

import time

import pandas as pd
import requests

##### CONSTANTS ##### (configurable)
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 4
DEFAULT_BACKOFF_TIME = 1.0
BASE_URL = "https://affinage.wi.mit.edu"
ANNOTATION_COL = "affinage_functional_annotation"
REFUSAL_PREFIXES = ("Parse failed", "No mechanistic", "Insufficient")


class AffinageClient:
    """Affinage API client for mechanistic narratives, gated on the API's audit_flag.

    Mirrors UniProtClient.fetch_functional_annotations so it drops into the bundle
    builder's annotation step. Symbols are HGNC alias-resolved server-side; genes
    that are flagged, refused, or not found return no annotation (left to the
    caller's backup).
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_time: float = DEFAULT_BACKOFF_TIME,  # initial backoff time in seconds
    ) -> None:
        self.base_url = base_url.rstrip("/")  # defense: remove trailing slash
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff_time
        self._session = requests.Session()
        self._cache: dict[str, str | None] = {}

    def _get(self, path: str) -> dict | None:
        url = f"{self.base_url}{path}"
        for attempt in range(self.max_retries):
            try:
                resp = self._session.get(url, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception:
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff * (2**attempt))  # backoff for Railway cold-starts
        return None

    def get_annotation(self, symbol: str) -> str | None:
        """Usable mechanistic narrative for a gene, or None if flagged/refused/missing."""
        sym = str(symbol).strip()
        if sym in self._cache:
            return self._cache[sym]
        result = None
        data = self._get(f"/api/mechanistic_narrative/{sym}")
        if data and not data.get("audit_flag"):
            narrative = data.get("mechanistic_narrative")
            if narrative and not narrative.startswith(REFUSAL_PREFIXES):
                result = narrative
        self._cache[sym] = result
        return result

    def fetch_functional_annotations(self, chunk: pd.DataFrame, gene_column: str) -> pd.DataFrame:
        """Return [gene_column, affinage_functional_annotation] for genes with usable narratives.

        Genes lacking a usable narrative are omitted, mirroring UniProtClient's
        "found only" return shape; the caller merges and fills any backup.
        """
        rows = [
            (symbol, annotation)
            for symbol in chunk[gene_column].dropna().unique()
            if (annotation := self.get_annotation(symbol))
        ]
        return pd.DataFrame(rows, columns=[gene_column, ANNOTATION_COL])
