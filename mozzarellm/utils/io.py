from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .bundle_schemas import EvidenceBundle


def write_bundle(bundle: EvidenceBundle, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(bundle.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")
    return p


def read_bundle(path: str | Path) -> EvidenceBundle:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return EvidenceBundle.model_validate(data)


def iter_bundle_paths(run_dir: str | Path) -> Iterable[Path]:
    rd = Path(run_dir)
    yield from sorted(rd.glob("**/*__bundle__*.json"))


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
