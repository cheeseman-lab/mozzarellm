from __future__ import annotations

import time
import warnings

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
    caller's backup). Infrastructure failures (timeouts, 5xx after retries) are
    raised so the bundle builder can warn and fall through to UniProt rather than
    silently degrading.
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

    def get_annotation(self, symbol: str) -> str | None:
        """Usable mechanistic narrative for a gene, or None if flagged/refused/missing.

        Soft failures (not in DB, audit-flagged, refusal-prefix narrative, empty
        narrative) emit a warning and return None. Infrastructure failures
        propagate from `_get` as exceptions.
        """
        sym = str(symbol).strip()
        if sym in self._cache:
            return self._cache[sym]

        data = self._get(f"/api/mechanistic_narrative/{sym}")
        result: str | None = None
        if data is None:
            warnings.warn(f"Affinage: gene not found for symbol {sym!r}", stacklevel=2)
        elif data.get("audit_flag"):
            warnings.warn(f"Affinage: record audit-flagged for {sym!r}", stacklevel=2)
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
                result = narrative

        self._cache[sym] = result
        return result

    def fetch_functional_annotations(self, chunk: pd.DataFrame, gene_column: str) -> pd.DataFrame:
        """Return [gene_column, affinage_functional_annotation] for genes with usable narratives.

        Genes lacking a usable narrative are omitted, mirroring UniProtClient's
        "found only" return shape; the caller merges and fills any backup. Warns
        with a summary of omitted symbols; raises if no symbol returned a usable
        narrative (matches UniProt's behavior on a zero-result batch).
        """
        symbols = [
            str(s).strip()
            for s in chunk[gene_column].dropna().unique()
            if str(s).strip() and str(s).strip() != "NON_TARGETING_CONTROL"
        ]
        rows: list[tuple[str, str]] = []
        missing: list[str] = []
        for symbol in symbols:
            annotation = self.get_annotation(symbol)
            if annotation is None:
                missing.append(symbol)
            else:
                rows.append((symbol, annotation))

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

        return pd.DataFrame(rows, columns=[gene_column, ANNOTATION_COL])
