from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

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
