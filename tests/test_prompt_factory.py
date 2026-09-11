"""Tests for prompt_factory module."""

import re

from mozzarellm.prompt_components import (
    CANONICAL_COT_ORDER,
    COT_STEP_GENE_CATEGORIZATION,
    COT_STEP_OUTPUT,
    COT_STEP_PATHWAY_HYPOTHESIS,
    COT_STEP_VERIFICATION,
    derive_cot_overrides,
)
from mozzarellm.utils.prompt_factory import (
    assemble_from_component_order,
    compose_stepwise_user_turns,
    make_cluster_analysis_system_prompt,
)


def test_standard_mode_returns_string(tmp_path):
    """Standard mode returns a non-empty string."""
    result = make_cluster_analysis_system_prompt(
        screen_name="test_screen",
        override_screen_context=True,
        output_dir=tmp_path,
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_standard_mode_includes_task(tmp_path):
    """Standard mode includes CLUSTER_ANALYSIS_TASK."""
    result = make_cluster_analysis_system_prompt(
        screen_name="test_screen",
        override_screen_context=True,
        output_dir=tmp_path,
    )
    assert "MISSION:" in result


def test_standard_mode_includes_rules(tmp_path):
    """Standard mode includes categorization rules."""
    result = make_cluster_analysis_system_prompt(
        screen_name="test_screen",
        override_screen_context=True,
        output_dir=tmp_path,
    )
    assert "ESTABLISHED" in result
    assert "NOVEL_ROLE" in result
    assert "UNCHARACTERIZED" in result


def test_standard_mode_includes_output_format(tmp_path):
    """Standard mode includes OUTPUT_FORMAT_JSON."""
    result = make_cluster_analysis_system_prompt(
        screen_name="test_screen",
        override_screen_context=True,
        output_dir=tmp_path,
    )
    assert "cluster_id" in result
    assert "dominant_process" in result


def test_cot_mode_uses_steps(tmp_path):
    """CoT mode assembles steps with STEP N headers."""
    result = make_cluster_analysis_system_prompt(
        screen_name="test_screen",
        override_screen_context=True,
        mode="cot",
        output_dir=tmp_path,
    )
    step_headers = re.findall(r"^STEP \d+ - ", result, re.MULTILINE)
    assert len(step_headers) == len(CANONICAL_COT_ORDER)


def test_cot_mode_includes_task_as_step1(tmp_path):
    """CoT mode includes CLUSTER_ANALYSIS_TASK as step 1."""
    result = make_cluster_analysis_system_prompt(
        screen_name="test_screen",
        override_screen_context=True,
        mode="cot",
        output_dir=tmp_path,
    )
    assert "STEP 1 -" in result
    assert "MISSION:" in result


def test_override_cot_steps_custom_order(tmp_path):
    """override_CoT_steps allows custom step order."""
    custom_steps = [
        COT_STEP_VERIFICATION,
        COT_STEP_OUTPUT,
    ]
    result = make_cluster_analysis_system_prompt(
        screen_name="test_screen",
        override_screen_context=True,
        mode="cot",
        override_CoT_steps=custom_steps,
        output_dir=tmp_path,
    )
    # Should have exactly 2 steps
    step_headers = re.findall(r"^STEP \d+ - ", result, re.MULTILINE)
    assert len(step_headers) == 2
    assert "STEP 1 - VERIFICATION" in result
    assert "STEP 2 - FINAL JSON OUTPUT" in result


def test_override_cot_steps_permutation(tmp_path):
    """override_CoT_steps respects permutation order."""
    # Pathway first
    pathway_first = make_cluster_analysis_system_prompt(
        screen_name="test_screen",
        override_screen_context=True,
        mode="cot",
        override_CoT_steps=[
            COT_STEP_PATHWAY_HYPOTHESIS,
            COT_STEP_GENE_CATEGORIZATION,
        ],
        output_dir=tmp_path,
    )
    # Gene categorization first
    gene_first = make_cluster_analysis_system_prompt(
        screen_name="test_screen",
        override_screen_context=True,
        mode="cot",
        override_CoT_steps=[
            COT_STEP_GENE_CATEGORIZATION,
            COT_STEP_PATHWAY_HYPOTHESIS,
        ],
        output_dir=tmp_path,
    )
    # Verify order differs
    assert pathway_first.index("PATHWAY HYPOTHESIS") < pathway_first.index("GENE CATEGORIZATION")
    assert gene_first.index("GENE CATEGORIZATION") < gene_first.index("PATHWAY HYPOTHESIS")


def test_saves_prompt_to_file(tmp_path):
    """Prompt is saved to output directory."""
    make_cluster_analysis_system_prompt(
        screen_name="test_screen",
        override_screen_context=True,
        output_dir=tmp_path,
    )
    saved_files = list(tmp_path.glob("cluster_analysis_phase1_system_prompt_*.txt"))
    assert len(saved_files) == 1


def test_custom_template_string(tmp_path):
    """Custom template string overrides default construction."""
    custom_template = "CUSTOM TEMPLATE CONTENT"
    result = make_cluster_analysis_system_prompt(
        screen_name="test_screen",
        override_screen_context=True,
        template_string=custom_template,
        output_dir=tmp_path,
    )
    assert "CUSTOM TEMPLATE CONTENT" in result


def test_includes_screen_context(tmp_path):
    """Prompt includes screen context text. Default standard mode uses 'experimental
    context'; cot/stepwise modes use 'screen context'."""
    result = make_cluster_analysis_system_prompt(
        screen_name="test_screen",
        override_screen_context=True,
        output_dir=tmp_path,
    )
    assert "context" in result.lower()


def test_base_overrides_propagate_to_cot_slots():
    """GCR/NPR/UPR/PCC overrides rebuild the cot slots composed from them."""
    overrides = {
        "GCR": "TUNED GCR RULES",
        "NPR": "TUNED NPR RULES",
        "UPR": "TUNED UPR RULES",
        "PCC": "TUNED PCC RUBRIC",
    }
    prompt = assemble_from_component_order(
        CANONICAL_COT_ORDER, "{}", cot_mode=True, component_overrides=overrides
    )
    assert "TUNED GCR RULES" in prompt
    assert "TUNED NPR RULES" in prompt
    assert "TUNED UPR RULES" in prompt
    assert "TUNED PCC RUBRIC" in prompt
    turns = compose_stepwise_user_turns(mcp=False, component_overrides=overrides)
    joined = "\n".join(t["content"] for t in turns)
    assert "TUNED GCR RULES" in joined
    assert "TUNED NPR RULES" in joined
    assert "TUNED PCC RUBRIC" in joined


def test_explicit_cot_slot_override_beats_derived():
    """An explicit cot-slot override wins over one derived from a base override."""
    derived = derive_cot_overrides({"GCR": "TUNED GCR RULES", "cGCR": "EXPLICIT COT STEP"})
    assert derived["cGCR"] == "EXPLICIT COT STEP"


def test_no_base_overrides_leaves_cot_slots_canonical():
    """Without base overrides the cot slots are the canonical composed texts."""
    assert derive_cot_overrides({}) == {}
    prompt = assemble_from_component_order(CANONICAL_COT_ORDER, "{}", cot_mode=True)
    assert COT_STEP_GENE_CATEGORIZATION.strip() in prompt
