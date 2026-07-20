"""Tests for named (not numbered) mode registry keys in arch_bench_routes."""

from architecture_benchmarking_workflow.arch_bench_routes import MODE_REGISTRY


def test_modes_named_not_numbered():
    assert set(MODE_REGISTRY) >= {"single_call", "cot", "stepwise"}
    for dead in ("3a", "3b", "3c", "3a_mcp", "3a_mcp_gap", "3b_mcp", "3c_mcp"):
        assert dead not in MODE_REGISTRY
    assert MODE_REGISTRY["single_call"].component_order[0] == "CAT"
    assert MODE_REGISTRY["single_call"].name == "single_call"
