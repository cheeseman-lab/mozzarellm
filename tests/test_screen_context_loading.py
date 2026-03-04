"""
Unit tests for mozzarellm.utils.screen_context_utils.load_screen_context_json
"""

from __future__ import annotations

import json

import pytest

from mozzarellm.utils.screen_context_utils import load_screen_context_json


####################### FIXTURES #######################


@pytest.fixture
def valid_context() -> dict:
    """Minimal valid screen context dict; each test gets its own mutable copy."""
    return {
        "assay_type": "CRISPRi",
        "target_phenotype": "Cell growth",
        "organism": "Homo sapiens",
        "cell_line_or_system": "K562",
        "perturbation": {"type": "CRISPRi", "library_or_reagent": "some library"},
        "readout": {
            "measurement": "fitness",
            "instrument_or_platform": "sequencing",
            "primary_metric": "log2fc",
        },
        "clustering": {"method": "leiden", "parameters": {"resolution": 0.5}},
        "controls": {"negative_controls": "Non targeting", "positive_controls": "essential genes"},
        "provenance": {
            "dataset_name": "example",
            "citation": "Doe et al.",
            "data_source": "internal",
        },
        "notes": "some notes",
    }


@pytest.fixture
def write_json(tmp_path):
    """Factory that writes a dict to a temp JSON file and returns the path string."""

    def _write(data: dict) -> str:
        p = tmp_path / "screen_context.json"
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return str(p)

    return _write


####################### TEST FUNCTIONS #######################


def test_load_screen_context_json_valid(valid_context, write_json):
    loaded = load_screen_context_json(write_json(valid_context))
    assert isinstance(loaded, dict)
    assert loaded["assay_type"] == "CRISPRi"
    assert loaded["perturbation"]["type"] == "CRISPRi"


def test_load_screen_context_json_allows_extra_fields(valid_context, write_json):
    valid_context["extra_top_level"] = "ok"
    valid_context["perturbation"]["extra_nested"] = "ok"
    loaded = load_screen_context_json(write_json(valid_context))
    assert loaded["extra_top_level"] == "ok"
    assert loaded["perturbation"]["extra_nested"] == "ok"


def test_load_screen_context_json_rejects_todo_field(valid_context, write_json):
    valid_context["TODO"] = {"description": "remove me"}
    with pytest.raises(Exception) as e:
        load_screen_context_json(write_json(valid_context))
    assert "Screen context JSON contains TODO field." in str(e.value)


def test_load_screen_context_json_rejects_template_placeholders(valid_context, write_json):
    valid_context["assay_type"] = "required"
    with pytest.raises(Exception) as e:
        load_screen_context_json(write_json(valid_context))
    assert "template placeholder" in str(e.value)


def test_load_screen_context_json_rejects_wrong_types(valid_context, write_json):
    valid_context["perturbation"] = "not an object"
    with pytest.raises(Exception) as e:
        load_screen_context_json(write_json(valid_context))
    assert "valid dictionary" in str(e.value).lower() or "dict" in str(e.value).lower()


def test_load_screen_context_json_override_behavior():
    assert load_screen_context_json(None, override=True) == {}
    with pytest.raises(Exception) as e:
        load_screen_context_json(None, override=False)
    assert "path is required" in str(e.value).lower()
