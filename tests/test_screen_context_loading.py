from __future__ import annotations

import json

import pytest

from mozzarellm.pipeline.bundle_builder import load_screen_context_json


def _valid_screen_context_dict() -> dict:
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


def _write_json(tmp_path, data: dict) -> str:
    p = tmp_path / "screen_context.json"
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(p)


def test_load_screen_context_json_valid(tmp_path):
    path = _write_json(tmp_path, _valid_screen_context_dict())
    loaded = load_screen_context_json(path)
    assert isinstance(loaded, dict)
    assert loaded["assay_type"] == "CRISPRi"
    assert loaded["perturbation"]["type"] == "CRISPRi"


def test_load_screen_context_json_allows_extra_fields(tmp_path):
    data = _valid_screen_context_dict()
    data["extra_top_level"] = "ok"
    data["perturbation"]["extra_nested"] = "ok"
    path = _write_json(tmp_path, data)
    loaded = load_screen_context_json(path)
    assert loaded["extra_top_level"] == "ok"
    assert loaded["perturbation"]["extra_nested"] == "ok"


def test_load_screen_context_json_rejects_todo_field(tmp_path):
    data = _valid_screen_context_dict()
    data["TODO"] = {"description": "remove me"}
    path = _write_json(tmp_path, data)
    with pytest.raises(Exception) as e:
        load_screen_context_json(path)
    assert "Screen context JSON contains TODO field." in str(e.value)


def test_load_screen_context_json_rejects_template_placeholders(tmp_path):
    data = _valid_screen_context_dict()
    data["assay_type"] = "required"
    path = _write_json(tmp_path, data)
    with pytest.raises(Exception) as e:
        load_screen_context_json(path)
    assert "template placeholder" in str(e.value)


def test_load_screen_context_json_rejects_wrong_types(tmp_path):
    data = _valid_screen_context_dict()
    data["perturbation"] = "not an object"
    path = _write_json(tmp_path, data)
    with pytest.raises(Exception) as e:
        load_screen_context_json(path)
    assert "valid dictionary" in str(e.value).lower() or "dict" in str(e.value).lower()


def test_load_screen_context_json_override_behavior():
    assert load_screen_context_json(None, override=True) == {}
    with pytest.raises(Exception) as e:
        load_screen_context_json(None, override=False)
    assert "path is required" in str(e.value).lower()
