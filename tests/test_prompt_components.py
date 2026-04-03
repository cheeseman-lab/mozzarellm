"""Tests for prompt_components module."""

import re
import pytest

from mozzarellm.prompt_components import (
    CLUSTER_ANALYSIS_TASK,
    COT_STEP_PATHWAY_HYPOTHESIS,
    COT_STEP_PATHWAY_SELECTION,
    COT_STEP_GENE_CATEGORIZATION,
    COT_STEP_SUBCLASSIFICATION,
    COT_STEP_VERIFICATION,
    COT_STEP_OUTPUT,
    COT_STEPS_DEFAULT,
    assemble_cot_instructions,
)


def test_default_steps_returns_string():
    """Default assembly returns a non-empty string."""
    result = assemble_cot_instructions()
    assert isinstance(result, str)
    assert len(result) > 0


def test_default_steps_numbered_correctly():
    """Default assembly has correct number of steps."""
    result = assemble_cot_instructions()
    # Count "STEP N - " headers at line start (not embedded text)
    step_headers = re.findall(r"^STEP \d+ - ", result, re.MULTILINE)
    assert len(step_headers) == len(COT_STEPS_DEFAULT)


def test_default_steps_includes_task():
    """Default assembly includes CLUSTER_ANALYSIS_TASK as first step."""
    result = assemble_cot_instructions()
    assert "STEP 1 -" in result
    assert "MISSION:" in result  # CLUSTER_ANALYSIS_TASK starts with MISSION


def test_custom_steps_order():
    """Custom step list is assembled in provided order."""
    custom_steps = [
        COT_STEP_VERIFICATION,
        COT_STEP_OUTPUT,
    ]
    result = assemble_cot_instructions(custom_steps)

    assert "STEP 1 - VERIFICATION" in result
    assert "STEP 2 - FINAL JSON OUTPUT" in result
    assert result.index("STEP 1") < result.index("STEP 2")


def test_custom_steps_count():
    """Custom step list produces correct number of steps."""
    custom_steps = [COT_STEP_VERIFICATION, COT_STEP_OUTPUT]
    result = assemble_cot_instructions(custom_steps)

    # Count step headers only
    step_headers = re.findall(r"^STEP \d+ - ", result, re.MULTILINE)
    assert len(step_headers) == 2


def test_empty_list_returns_empty_string():
    """Empty step list returns empty string."""
    result = assemble_cot_instructions([])
    assert result == ""


def test_single_step():
    """Single step is numbered as STEP 1."""
    result = assemble_cot_instructions([COT_STEP_VERIFICATION])
    assert "STEP 1 - VERIFICATION" in result
    step_headers = re.findall(r"^STEP \d+ - ", result, re.MULTILINE)
    assert len(step_headers) == 1


def test_steps_separated_by_double_newline():
    """Steps are separated by double newlines."""
    custom_steps = [COT_STEP_VERIFICATION, COT_STEP_OUTPUT]
    result = assemble_cot_instructions(custom_steps)

    assert "\n\n" in result


def test_permutation_reorders_steps():
    """Permuting step order changes output order."""
    # Original order
    original = assemble_cot_instructions(
        [
            COT_STEP_PATHWAY_HYPOTHESIS,
            COT_STEP_GENE_CATEGORIZATION,
        ]
    )
    # Reversed order
    reversed_order = assemble_cot_instructions(
        [
            COT_STEP_GENE_CATEGORIZATION,
            COT_STEP_PATHWAY_HYPOTHESIS,
        ]
    )

    # In original, PATHWAY comes before GENE CATEGORIZATION
    assert original.index("PATHWAY HYPOTHESIS") < original.index("GENE CATEGORIZATION")
    # In reversed, GENE CATEGORIZATION comes before PATHWAY
    assert reversed_order.index("GENE CATEGORIZATION") < reversed_order.index("PATHWAY HYPOTHESIS")


# =============================================================================
# Screen context interleaving tests
# =============================================================================


def test_screen_context_replaces_placeholder():
    """Screen context replaces COT_SCREEN_CONTEXT placeholder."""
    from mozzarellm.prompt_components import COT_SCREEN_CONTEXT

    custom_steps = [COT_SCREEN_CONTEXT, COT_STEP_VERIFICATION]
    result = assemble_cot_instructions(custom_steps, screen_context='{"test": "context"}')

    assert '{"test": "context"}' in result
    assert "STEP 1 -" in result
    assert "STEP 2 - VERIFICATION" in result


def test_screen_context_preserves_header():
    """Screen context replacement preserves the COT_SCREEN_CONTEXT header text."""
    from mozzarellm.prompt_components import COT_SCREEN_CONTEXT

    custom_steps = [COT_SCREEN_CONTEXT]
    result = assemble_cot_instructions(custom_steps, screen_context='{"data": 1}')

    # Should contain both the header and the context
    assert COT_SCREEN_CONTEXT in result
    assert '{"data": 1}' in result


def test_screen_context_none_leaves_placeholder():
    """When screen_context is None, COT_SCREEN_CONTEXT remains as-is."""
    from mozzarellm.prompt_components import COT_SCREEN_CONTEXT

    custom_steps = [COT_SCREEN_CONTEXT, COT_STEP_VERIFICATION]
    result = assemble_cot_instructions(custom_steps, screen_context=None)

    # Placeholder should remain unchanged (just the header text)
    assert "STEP 1 - " + COT_SCREEN_CONTEXT in result
    assert "STEP 2 - VERIFICATION" in result


def test_screen_context_interleaved_position():
    """Screen context can be interleaved at any position."""
    from mozzarellm.prompt_components import COT_SCREEN_CONTEXT

    # Context in middle
    steps = [COT_STEP_PATHWAY_HYPOTHESIS, COT_SCREEN_CONTEXT, COT_STEP_VERIFICATION]
    result = assemble_cot_instructions(steps, screen_context='{"mid": true}')

    assert result.index("PATHWAY HYPOTHESIS") < result.index('{"mid": true}')
    assert result.index('{"mid": true}') < result.index("VERIFICATION")


def test_screen_context_with_default_steps():
    """Screen context works with COT_STEPS_DEFAULT."""
    result = assemble_cot_instructions(screen_context='{"screen": "test"}')

    # Default steps include COT_SCREEN_CONTEXT, so context should appear
    assert '{"screen": "test"}' in result


# =============================================================================
# Content verification tests
# =============================================================================


def test_default_steps_contain_expected_content():
    """Default assembly contains key content from each step."""
    result = assemble_cot_instructions()

    # Verify key content from various steps is present
    assert "MISSION:" in result  # CLUSTER_ANALYSIS_TASK
    assert "PATHWAY HYPOTHESIS" in result
    assert "PATHWAY SELECTION" in result
    assert "GENE CATEGORIZATION" in result
    assert "SUB-CLASSIFICATION" in result
    assert "VERIFICATION" in result
    assert "FINAL JSON OUTPUT" in result


def test_steps_contain_referenced_rules():
    """Steps that reference rules contain the actual rule content."""
    result = assemble_cot_instructions([COT_STEP_GENE_CATEGORIZATION])

    # GENE_CATEGORIZATION_RULES should be embedded
    assert "ESTABLISHED" in result
    assert "NOVEL_ROLE" in result
    assert "UNCHARACTERIZED" in result


def test_output_step_contains_json_format():
    """Output step contains JSON format specification."""
    result = assemble_cot_instructions([COT_STEP_OUTPUT])

    assert "cluster_id" in result
    assert "dominant_process" in result


# =============================================================================
# Edge cases
# =============================================================================


def test_duplicate_steps_allowed():
    """Duplicate steps are allowed and numbered separately."""
    custom_steps = [COT_STEP_VERIFICATION, COT_STEP_VERIFICATION]
    result = assemble_cot_instructions(custom_steps)

    step_headers = re.findall(r"^STEP \d+ - ", result, re.MULTILINE)
    assert len(step_headers) == 2
    assert "STEP 1 - VERIFICATION" in result
    assert "STEP 2 - VERIFICATION" in result


def test_step_numbering_is_sequential():
    """Step numbers are always sequential starting from 1."""
    custom_steps = [
        COT_STEP_PATHWAY_HYPOTHESIS,
        COT_STEP_PATHWAY_SELECTION,
        COT_STEP_GENE_CATEGORIZATION,
        COT_STEP_SUBCLASSIFICATION,
    ]
    result = assemble_cot_instructions(custom_steps)

    for i in range(1, 5):
        assert f"STEP {i} -" in result
