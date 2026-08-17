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
AUDIT_NOTE_COL = "affinage_audit_note"
REFUSAL_PREFIXES = ("Parse failed", "No mechanistic", "Insufficient")


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
        self._cache: dict[str, dict | None] = {}

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

        Returns {"narrative": str, "audit_note": str}. Audit-flagged narratives
        are surfaced, not dropped — the flag is advisory and carried through as
        audit_note for downstream weighting. Only genuinely unusable responses
        (not found, empty, or a refusal-message narrative) return None.
        Infrastructure failures propagate from `_get` as exceptions.
        """
        sym = str(symbol).strip()
        if sym in self._cache:
            return self._cache[sym]

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
        return record

    def get_annotation(self, symbol: str) -> str | None:
        """Mechanistic narrative for a gene, or None if unusable.

        Audit-flagged narratives are returned; the flag is available via
        get_annotation_record().
        """
        record = self.get_annotation_record(symbol)
        return record["narrative"] if record else None

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
