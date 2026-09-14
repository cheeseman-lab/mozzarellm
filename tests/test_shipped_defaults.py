"""The shipped prompt defaults are the benchmark-selected build.

The package's default cot+MCP system prompt must equal, byte for byte, the
system prompt of the order run of record (canonical O on cot_litb with the
walkup-tuned components) for every benchmark screen. The reference prompts
live in the benchmark outputs submodule; the test skips when it is not
checked out.
"""

from pathlib import Path

import pytest

from mozzarellm.prompts import make_cluster_analysis_system_prompt

_REPO = Path(__file__).resolve().parents[1]
_RUN_OF_RECORD = _REPO / "benchmarks" / "outputs" / "order" / "O_20260911_180759" / "prompts_used"
_INPUTS = _REPO / "benchmarks" / "inputs"
_SCREENS = ("whitney", "denali", "jebel", "aconcagua_interphase", "aconcagua_interphase_shuffled")


@pytest.mark.skipif(
    not _RUN_OF_RECORD.exists(), reason="benchmark outputs submodule not checked out"
)
@pytest.mark.parametrize("screen", _SCREENS)
def test_default_cot_mcp_prompt_is_the_run_of_record(screen):
    reference = (_RUN_OF_RECORD / f"system_prompt_O__cot_mcp_cot_{screen}.txt").read_text(
        encoding="utf-8"
    )
    assembled = make_cluster_analysis_system_prompt(
        screen_name=screen,
        screen_context_path=_INPUTS / f"{screen}_screen_context.json",
        mode="cot",
        mcp=True,
    )
    assert assembled == reference
