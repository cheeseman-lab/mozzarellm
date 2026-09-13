"""Prompt assembly: the defaults, overrides, and byte identity with the pre-reshape code.

_PRE_RESHAPE holds SHA-256 digests of renders captured on the parent commit
(mozzarellm/prompt_components.py + utils/prompt_factory.py, before
mozzarellm/prompts/). Every assembly path must still produce those bytes;
render the same cases on the parent commit to reproduce the digests.
"""

import hashlib
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

_PRE_RESHAPE = {
    "cot": "f4feb9eaef5df840e448eaee129bc3139180ece54334bcaa60dab1da0c32363f",
    "cot_mcp": "de645740e3b92d9a930852075dc144edc82b3b7539e1a073be80e2d2d38795b2",
    "cot_mcp_features_strength": "31e54528057c684fe5c8a1afd5452b7da0eb20f210a63b3db9ce0bd2f65e3687",
    "cot_overrides": "a4218dee7d8771bcd6dc653b5fd332e6ea768d9dcf9b5ab199ae54e2dbafb75c",
    "standard": "9e9364e94f40c90e0973ed2dd70e1b6c7aa70b882d4468a8e723450e1f461034",
    "standard_mcp": "726120372b35fb4f5689d43a09495f426ad93db1359127200e550b9c4174aaa3",
    "standard_overrides": "354a76186d4041e43aee40dcdf9909e370a71436845c9df3e1e20124cb783201",
    "stepwise_system": "7f993f9a7720bb116e4791b1b5a1c715ceeef13ac8efaf7269bf3a7099c0b524",
    "stepwise_turns": "6030b204d4630bae82d09e9f72df8e2f483a418041e1f568ee2a8f41687bab96",
    "stepwise_turns_mcp": "e766667a0cda419cfad8a1b51c4715a5064d2e65d19c1e3e3a3c6d143d721b0e",
    "stepwise_turns_overrides": "48c241cb67676a0842799c6fa6b11fa0d7d38cd0292d04a2a6c3129b351cc8d5",
}
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
    turns = json.dumps(compose_stepwise_user_turns(**kwargs), indent=1, ensure_ascii=False)
    assert hashlib.sha256(turns.encode("utf-8")).hexdigest() == _PRE_RESHAPE[name]


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
