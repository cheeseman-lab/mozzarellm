"""Cross-module integrity tests for prompt_components.py.

Validates that the component registry, canonical orders, and route definitions
stay consistent as the codebase evolves.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mozzarellm.prompt_components import (  # noqa: E402
    CANONICAL_COT_MCP_ORDER,
    CANONICAL_COT_ORDER,
    CANONICAL_ZERO_SHOT_ORDER,
    COMPONENT_REGISTRY,
    GENE_CATEGORIZATION_RULES,
)
from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_routes import (  # noqa: E402
    ROUTE_REGISTRY,
)


class TestPromptComponentsRegistry:
    def test_canonical_orders_reference_valid_keys(self):
        valid_keys = set(COMPONENT_REGISTRY.keys()) | {"SC"}
        for order_name, order in (
            ("CANONICAL_ZERO_SHOT_ORDER", CANONICAL_ZERO_SHOT_ORDER),
            ("CANONICAL_COT_ORDER", CANONICAL_COT_ORDER),
            ("CANONICAL_COT_MCP_ORDER", CANONICAL_COT_MCP_ORDER),
        ):
            for key in order:
                assert key in valid_keys, (
                    f"{key!r} in {order_name} is not in COMPONENT_REGISTRY and is not 'SC'"
                )

    def test_registry_values_nonempty(self):
        for key, value in COMPONENT_REGISTRY.items():
            assert isinstance(value, str), f"COMPONENT_REGISTRY[{key!r}] is not a string"
            assert value.strip(), f"COMPONENT_REGISTRY[{key!r}] is empty or whitespace-only"

    def test_cot_embeds_baseline(self):
        gcr_text = GENE_CATEGORIZATION_RULES
        cgcr_text = COMPONENT_REGISTRY["cGCR"]

        if gcr_text.strip() in cgcr_text:
            pass  # CoT version literally contains the baseline text
        else:
            assert len(cgcr_text) > len(COMPONENT_REGISTRY["GCR"]), (
                "cGCR should be longer than GCR if it doesn't literally embed the baseline text"
            )

    def test_canonical_zero_shot_matches_3a(self):
        assert tuple(CANONICAL_ZERO_SHOT_ORDER) == ROUTE_REGISTRY["3a"].component_order

    def test_canonical_cot_matches_3b(self):
        assert tuple(CANONICAL_COT_ORDER) == ROUTE_REGISTRY["3b"].component_order

    def test_canonical_cot_mcp_matches_3b_mcp(self):
        assert tuple(CANONICAL_COT_MCP_ORDER) == ROUTE_REGISTRY["3b_mcp"].component_order

    def test_mcp_orders_include_lit(self):
        for route_name, route in ROUTE_REGISTRY.items():
            if route.mcp:
                assert "LIT" in route.component_order, (
                    f"MCP route {route_name!r} is missing 'LIT' in its component_order"
                )
            else:
                assert "LIT" not in route.component_order, (
                    f"Non-MCP route {route_name!r} should not contain 'LIT'"
                )

    def test_both_literature_variants_registered(self):
        # Two selectable MCP literature prompts: category-gated ("LIT") and
        # blank-annotation gap-fill ("LITB").
        category = COMPONENT_REGISTRY["LIT"]
        blank = COMPONENT_REGISTRY["LITB"]
        assert "NOVEL_ROLE and UNCHARACTERIZED" in category
        assert "GAP-FILL" in blank and "mcp_gapfill" in blank
        assert category != blank

    def test_feature_interp_orders(self):
        assert "cFC" in COMPONENT_REGISTRY, "cFC (Feature Coherence) should be defined"
        assert "cPC" in COMPONENT_REGISTRY, "cPC (Pathway Consistency) should be defined"

    def test_wording_v2_cat_references_correct_component(self):
        from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.wording_bench_alternates import (
            WORDING_ALTERNATE_SET_REGISTRY,
        )

        v2_cat = WORDING_ALTERNATE_SET_REGISTRY["wording_v2"]["CAT"]
        assert isinstance(v2_cat, str) and v2_cat.strip(), (
            "wording_v2 CAT alternate should be a non-empty string"
        )
        assert v2_cat != COMPONENT_REGISTRY["CAT"], (
            "wording_v2 CAT should differ from the canonical COMPONENT_REGISTRY CAT"
        )
