from pathlib import Path
from typing import Any, Dict, Mapping
from datetime import datetime
import json
import pandas as pd
from mozzarellm.utils.bundle_schemas import (
    BundleGene,
    BundleGeneAnnotations,
    EvidenceBundle,
    ScreenContext,
)
from mozzarellm.utils.io import write_bundle
from mozzarellm.utils.retrieval import local_knowledge_context_retriever


# Default values
screen_name = "screen_name"
BUNDLE_OUTPUT_PATH_BASE = f"intermediates/{screen_name}__bundles/"
JSON_BYTE_CAP = 5_000  # 5 KB -- conservative cap, needs to be adjusted


def context_json_validator(data) -> bool:
    """Validate that the context JSON is valid and doesn't contain TODO fields."""
    if "TODO" in data.keys():
        raise ValueError(
            "Screen context JSON contains TODO field. Please double check the file and remove it."
        )
    if (
        len(json.dumps(data).encode("utf-8")) > JSON_BYTE_CAP
    ):  # intended as a model agnostic cap; chars is an alt option
        raise ValueError("Screen context JSON is too large. Please reduce the size of the file.")
    return True


def load_screen_context_json(
    path: str | Path | None,
    *,
    override: bool = False,  # optional kwarg
) -> Dict[str, Any] | None:
    """Load structured screen context from JSON."""
    try:
        if path is None:
            if override:  # testing convenience, to see how much context changes response
                return {}
            raise ValueError("Screen context path is required")

        json_path = Path(path)
        if not json_path.exists():
            raise FileNotFoundError(f"Screen context file not found: {json_path}")

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        context_json_validator(data)  # size and completion check
        model = ScreenContext.model_validate(data)  # schema validation
        return model.model_dump()
    except Exception as e:
        raise Exception(f"Error loading screen context JSON: {e}") from e


def load_cluster_table(
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


def build_evidence_bundles(
    screen_name: str | None = None,
    screen_context_path: str | Path | None = None,
    cluster_df: pd.DataFrame | None = None,
    gene_column: str = "genes",
    cluster_id_column: str = "cluster_id",
    use_retrieval: bool = True,
    override_screen_context: bool = False,
    knowledge_dir: str
    | Path
    | None = None,  # optionally change the directory where the knowledge files are stored
    top_k: int = 10,  #
) -> list[Path]:
    # validate required columns
    if cluster_id_column not in cluster_df.columns:
        raise ValueError(f"Missing column '{cluster_id_column}' in cluster table")
    if gene_column not in cluster_df.columns:
        raise ValueError(f"Missing column '{gene_column}' in cluster table")

    screen_context = load_screen_context_json(screen_context_path, override=override_screen_context)

    # Create output directory
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    pass # in progress
