from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from pydantic import ValidationError
from mozzarellm.schemas.bundle_schemas import EvidenceBundle


def write_bundle(evidence_bundle_dict: dict, path: str | Path) -> Path:
    try:
        bundle = EvidenceBundle.model_validate(evidence_bundle_dict)
    except ValidationError as e:
        raise ValueError(f"Invalid bundle: {e}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(bundle.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")
    return p


def load_table(
    input_path: str | Path,
    sep: str | None = None,
    sheet_name: str | int | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Load a cluster/gene table from CSV/TSV/TXT/XLSX into a DataFrame. Extra pandas kwargs are forwarded to pandas read_csv/read_excel."""
    path = Path(input_path)
    suffix = path.suffix.lower()

    if suffix == ".xlsx":
        return pd.read_excel(path, sheet_name=sheet_name, **kwargs)

    if suffix in {".csv", ".tsv", ".txt"}:
        default_sep = "," if suffix == ".csv" else "\t"
        return pd.read_csv(path, sep=(sep or default_sep), **kwargs)

    raise ValueError(
        f"Unsupported input format for cluster table: {suffix}. "
        "Expected one of .csv, .tsv, .txt, .xlsx."
    )
