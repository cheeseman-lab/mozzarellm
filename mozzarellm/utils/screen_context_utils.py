import json
from pathlib import Path
from typing import Dict, Any
from mozzarellm.schemas.bundle_schemas import ScreenContext

JSON_BYTE_CAP = 5_000  # 5 KB -- conservative cap, needs to be adjusted


def _context_json_validator(data) -> bool:
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

        _context_json_validator(data)  # size and completion check
        model = ScreenContext.model_validate(data)  # schema validation
        return model.model_dump()
    except Exception as e:
        raise Exception(f"Error loading screen context JSON: {e}") from e
