"""Prompt assembly: the defaults, overrides, and the byte-identity fixtures.

tests/fixtures/prompts/ holds the renders captured before the prompts were
reshaped (mozzarellm/prompt_components.py + utils/prompt_factory.py ->
mozzarellm/prompts/). Every assembly path must still produce them.
"""

import json
import re
from pathlib import Path

import pytest

from benchmarks.workflow.bench_routes import ROUTE_REGISTRY
from mozzarellm.prompts import (
    COMPONENTS,
    DEFAULT_ORDERS,
    EMBEDS,
    assemble,
    compose_stepwise_user_turns,
    default_order,
    make_cluster_analysis_system_prompt,
    render,
)
from mozzarellm.prompts.components import (
    GENE_CATEGORIZATION_RULES,
    STEP_LITERATURE_GAPFILL_BLANK,
    STEP_LITERATURE_VALIDATION,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "prompts"
_CONTEXT = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "inputs" / "whitney_screen_context.json"
)
_OVERRIDES = {"GCR": "GCR OVERRIDE TEXT\n", "PCC": "PCC OVERRIDE TEXT\n"}


@pytest.mark.parametrize(
    "name, kwargs",
    [
        ("standard", {"mode": "standard"}),
        ("standard_mcp", {"mode": "standard", "mcp": True}),
        ("cot", {"mode": "cot"}),
        ("cot_mcp", {"mode": "cot", "mcp": True}),
        ("stepwise_system", {"mode": "stepwise", "mcp": True}),
        (
            "cot_mcp_features_strength",
            dict(
                mode="cot",
                mcp=True,
                component_order=default_order("cot", True, features=True, strength=True),
            ),
        ),
        ("cot_overrides", {"mode": "cot", "component_overrides": _OVERRIDES}),
        ("standard_overrides", {"mode": "standard", "component_overrides": _OVERRIDES}),
    ],
)
def test_system_prompt_matches_pre_reshape_render(name, kwargs):
    expected = (_FIXTURES / f"{name}.txt").read_text(encoding="utf-8")
    assert (
        make_cluster_analysis_system_prompt(
            screen_name="whitney", screen_context_path=_CONTEXT, **kwargs
        )
        == expected
    )


@pytest.mark.parametrize(
    "name, kwargs",
    [
        ("stepwise_turns", {"mcp": False}),
        ("stepwise_turns_mcp", {"mcp": True}),
        ("stepwise_turns_overrides", {"mcp": True, "component_overrides": _OVERRIDES}),
    ],
)
def test_stepwise_turns_match_pre_reshape_render(name, kwargs):
    expected = json.loads((_FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    assert compose_stepwise_user_turns(**kwargs) == expected


def test_registry_is_texts_and_orders_reference_it():
    for key, value in COMPONENTS.items():
        assert isinstance(value, str) and value.strip(), key
    for (mode, mcp), order in DEFAULT_ORDERS.items():
        assert set(order) <= set(COMPONENTS) | {"SC"}, (mode, mcp)
        assert ("LIT" in order) == mcp
    for key, slots in EMBEDS.items():
        assert key in COMPONENTS and set(slots) <= set(COMPONENTS)
        for slot in slots:
            assert "{" + slot + "}" in COMPONENTS[key]


def test_render_fills_templates_and_propagates_base_overrides():
    default = render()
    assert GENE_CATEGORIZATION_RULES in default["cGCR"]
    assert "{GCR}" not in default["cGCR"]
    tuned = render({"GCR": "TUNED GCR RULES", "PCC": "TUNED PCC RUBRIC"})
    assert "TUNED GCR RULES" in tuned["cGCR"] and "TUNED PCC RUBRIC" in tuned["cPSC"]
    assert tuned["cPri"] == default["cPri"]
    # New with the reshape: the output step embeds O, so an O override re-renders cO.
    assert "O OVERRIDE" in render({"O": "O OVERRIDE"})["cO"]
    explicit = render({"GCR": "TUNED GCR RULES", "cGCR": "EXPLICIT COT STEP"})
    assert explicit["cGCR"] == "EXPLICIT COT STEP"
    with pytest.raises(ValueError, match="Unknown component keys"):
        render({"nope": "x"})


def test_assemble_numbers_steps_in_cot_and_injects_context():
    flat = assemble(["CAT", "SC", "O"], "{CTX}")
    assert flat.startswith("MISSION:") and "experimental context is provided: {CTX}" in flat
    numbered = assemble(default_order("cot"), "{CTX}", cot=True)
    assert len(re.findall(r"^STEP \d+ - ", numbered, re.MULTILINE)) == len(
        DEFAULT_ORDERS[("cot", False)]
    )
    with pytest.raises(ValueError, match="Unknown component key"):
        assemble(["CAT", "XX"], "{}")


def test_default_order_phenotype_steps_and_mode_guard():
    assert default_order("cot", True, features=True, strength=True)[-4:] == [
        "cFC",
        "cPC",
        "cPS",
        "cO",
    ]
    assert default_order("stepwise", False) == default_order("cot", False)
    with pytest.raises(ValueError):
        default_order("standard", False, features=True)
    with pytest.raises(ValueError):
        default_order("bogus")


def test_literature_slot_default_and_variant():
    assert COMPONENTS["LIT"] is STEP_LITERATURE_GAPFILL_BLANK
    assert COMPONENTS["LITV"] is STEP_LITERATURE_VALIDATION
    for text in (STEP_LITERATURE_VALIDATION, STEP_LITERATURE_GAPFILL_BLANK):
        assert "single valid JSON object" in text and "before the opening brace" in text


def test_bench_routes_mirror_the_default_orders():
    assert DEFAULT_ORDERS[("standard", False)] == ROUTE_REGISTRY["single_call"].component_order
    assert DEFAULT_ORDERS[("cot", False)] == ROUTE_REGISTRY["cot"].component_order
    assert DEFAULT_ORDERS[("cot", True)] == ROUTE_REGISTRY["cot_mcp"].component_order
    for name, route in ROUTE_REGISTRY.items():
        assert ("LIT" in route.component_order) == route.mcp, name


def test_saved_prompt_file(tmp_path):
    make_cluster_analysis_system_prompt(
        screen_name="s", override_screen_context=True, output_dir=tmp_path, prompt_filename="p"
    )
    assert (tmp_path / "p.txt").exists()
