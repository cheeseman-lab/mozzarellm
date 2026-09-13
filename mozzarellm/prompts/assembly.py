"""Prompt assembly.

A prompt is an ordered list of component keys rendered against the texts in
components.py. The system prompt carries the stable policy (task, rules,
output format) plus the screen context; the user prompt carries one
cluster's evidence bundle.

The defaults below are the benchmark-selected build. Any caller can assemble
differently: pass a ``component_order`` for a different chain, or
``component_overrides`` ({key: text}) to reword any component. Both go
through the same ``render`` step, so validation never depends on the default.
"""

import json
from pathlib import Path

from mozzarellm.prompts.components import COMPONENTS, EMBEDS
from mozzarellm.utils.screen_context_utils import load_screen_context_json, validate_screen_context

MODES = ("standard", "cot", "stepwise")

# The default chain per (mode, mcp). Stepwise delivers the cot chain one turn
# at a time, so it shares the cot orders.
DEFAULT_ORDERS = {
    ("standard", False): ("CAT", "SC", "GCR", "NPR", "UPR", "PCC", "O"),
    ("standard", True): ("CAT", "SC", "GCR", "NPR", "UPR", "PCC", "LIT", "O"),
    ("cot", False): ("CAT", "SC", "cPH", "cGCR", "cPri", "cPSC", "cVer", "cO"),
    ("cot", True): ("CAT", "SC", "cPH", "cGCR", "cPri", "LIT", "cPSC", "cVer", "cO"),
}

# Keys whose turn attaches the MCP tools in stepwise delivery.
MCP_KEYS = ("LIT", "LITV")


def default_order(
    mode: str, mcp: bool = False, features: bool = False, strength: bool = False
) -> list[str]:
    """The default component chain, with phenotype steps included iff their data is.

    The feature steps (cFC, cPC) and the strength step (cPS) join the cot
    chain after verification and before the output step; a prompt never
    references data the bundles don't carry.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    order = list(DEFAULT_ORDERS[("standard" if mode == "standard" else "cot", mcp)])
    if mode == "standard" and (features or strength):
        raise ValueError("the phenotype steps are chain-of-thought components (mode='cot')")
    tail = order.pop()  # the output step stays last
    if features:
        order += ["cFC", "cPC"]
    if strength:
        order += ["cPS"]
    return order + [tail]


def render(component_overrides: dict[str, str] | None = None) -> dict[str, str]:
    """The component texts with overrides applied and the cot templates filled.

    A chain-of-thought step that embeds a base text (see EMBEDS) is rendered
    from the overridden base, so overriding GCR also rewords cGCR. An explicit
    override for the step itself wins.
    """
    overrides = component_overrides or {}
    texts = {**COMPONENTS, **overrides}
    for key, slots in EMBEDS.items():
        if key not in overrides:
            texts[key] = COMPONENTS[key].format(**{slot: texts[slot] for slot in slots})
    unknown = set(overrides) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown component keys: {sorted(unknown)}")
    return texts


def assemble(
    component_order: list[str],
    screen_context_text: str,
    cot: bool = False,
    component_overrides: dict[str, str] | None = None,
) -> str:
    """Join components in order; "SC" is the screen context; cot numbers the steps."""
    texts = render(component_overrides)
    parts = []
    for key in component_order:
        if key == "SC":
            parts.append("The following experimental context is provided: " + screen_context_text)
        elif key in texts:
            parts.append(texts[key])
        else:
            raise ValueError(f"Unknown component key: {key!r}. Valid: {sorted(texts)} + 'SC'")
    if cot:
        parts = [f"STEP {i + 1} - {part}" for i, part in enumerate(parts)]
    return "\n\n".join(parts)


def screen_context_text(
    screen_context_path: Path | None = None,
    screen_context: dict | None = None,
    placeholder: bool = False,
) -> str:
    """Load (or validate the given dict) and minify the screen context."""
    try:
        if screen_context is not None:
            ctx = validate_screen_context(screen_context)
        else:
            ctx = load_screen_context_json(screen_context_path, override=placeholder)
    except Exception as e:
        raise ValueError(f"Failed to load screen context: {e}") from e
    return json.dumps(ctx, ensure_ascii=False)


def make_cluster_analysis_system_prompt(
    *,
    screen_name: str,
    screen_context_path: Path | None = None,
    screen_context: dict | None = None,
    mode: str = "standard",
    mcp: bool = False,
    component_order: list[str] | None = None,
    component_overrides: dict[str, str] | None = None,
    override_screen_context: bool = False,
    output_dir: Path | None = None,
    prompt_filename: str | None = None,
) -> str:
    """The system prompt for one screen.

    Args:
        screen_name: Names the saved prompt file when ``output_dir`` is given.
        screen_context_path / screen_context: The screen's context, as a JSON
            file or an in-memory dict (validated through the same schema).
        mode: "standard" (flat), "cot" (numbered steps, one call), or
            "stepwise" (the system prompt holds only the task and context; the
            steps are delivered as user turns, see compose_stepwise_user_turns).
        mcp: Include the literature step (the "LIT" slot).
        component_order: A chain of component keys replacing the default.
        component_overrides: {key: text} rewording any component.
        override_screen_context: Use a placeholder context (testing).
        output_dir: Save the assembled prompt here for inspection.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    context = screen_context_text(screen_context_path, screen_context, override_screen_context)
    if component_order is None:
        component_order = default_order(mode, mcp)
        if mode == "stepwise":
            component_order = component_order[:2]  # task + context; the turns carry the rest
    prompt = assemble(
        component_order, context, cot=(mode == "cot"), component_overrides=component_overrides
    )
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        name = prompt_filename or f"{screen_name}_system_prompt"
        (output_dir / (name if name.endswith(".txt") else f"{name}.txt")).write_text(
            prompt, encoding="utf-8"
        )
    return prompt


def compose_stepwise_user_turns(
    mcp: bool, component_overrides: dict[str, str] | None = None
) -> list[dict]:
    """The per-turn user content for stepwise delivery.

    The task and screen context live in the system prompt; the remaining
    steps of the cot chain become numbered turns. Each turn carries a flag for
    whether it attaches the MCP tools. The client prepends the cluster bundle
    to turn 0.
    """
    texts = render(component_overrides)
    keys = default_order("cot", mcp)[2:]
    return [
        {"content": f"STEP {i + 1} - {texts[key]}", "mcp": mcp and key in MCP_KEYS}
        for i, key in enumerate(keys)
    ]


# =============================================================================
# USER PROMPT: one cluster's evidence bundle
# =============================================================================

# Per-gene feature lists never reach the model: the feature steps read the
# bounded cluster-level feature_coherence table (which carries the supporting
# gene lists). Each aggregate records the per-gene columns it was built from,
# so the strip knows the user's column names; this default covers bundles
# built before that record existed.
FEATURE_FIELDS = ("up_features", "down_features")

# Per-source annotation fields; a master bundle carries the superset of both.
SOURCE_FIELDS = {
    "uniprot": ("UniProt_functional_annotation",),
    "affinage": ("affinage_functional_annotation", "affinage_audit_note"),
}


def strip_feature_fields(
    bundle_obj: dict,
    fields: tuple[str, ...] = FEATURE_FIELDS,
    *,
    features: bool = True,
    strength: bool = True,
) -> None:
    """Remove screen-derived phenotype data from an evidence bundle in place.

    Per-gene feature columns are always removed: ``fields`` plus whatever the
    bundle's feature_coherence table records as its source columns.
    features/strength select which cluster-level aggregate to strip; strength
    also drops the per-gene rank column the phenotype_strength table names.
    """
    coherence = bundle_obj.get("feature_coherence") or {}
    strength_block = bundle_obj.get("phenotype_strength") or {}
    per_gene = set(fields) | set(coherence.get("columns") or ())
    if features:
        bundle_obj.pop("feature_coherence", None)
    if strength:
        bundle_obj.pop("phenotype_strength", None)
        if strength_block.get("column"):
            per_gene.add(strength_block["column"])
    for gene in bundle_obj.get("cluster_genes", []):
        if isinstance(gene, dict):
            for field in per_gene:
                gene.pop(field, None)


def strip_source_fields(bundle_obj: dict, source: str) -> None:
    """Reduce a master (superset) evidence bundle to one source's view, in place.

    source="both" is a no-op (the master is the both-sources view). Otherwise
    the other source's annotation fields are removed, and the kept source's
    empty annotations (None/"") are dropped rather than serialized as empty.
    """
    if source == "both":
        return
    if source not in SOURCE_FIELDS:
        raise ValueError(f"source must be one of 'uniprot', 'affinage', 'both'; got {source!r}")
    other = "affinage" if source == "uniprot" else "uniprot"
    for gene in bundle_obj.get("cluster_genes", []):
        if isinstance(gene, dict):
            for field in SOURCE_FIELDS[other]:
                gene.pop(field, None)
            for field in SOURCE_FIELDS[source]:
                if not gene.get(field):
                    gene.pop(field, None)


def make_single_cluster_analysis_user_prompt(
    cluster_id,
    screen_name,
    cluster_to_bundle_path_map,
    include_features=False,
    include_strength=False,
    source="both",
) -> str:
    """The user prompt for one cluster: its evidence bundle, reduced to what the chain reads."""
    bundle_obj = json.loads(
        Path(cluster_to_bundle_path_map[str(cluster_id)]).read_text(encoding="utf-8")
    )
    strip_feature_fields(bundle_obj, features=not include_features, strength=not include_strength)
    strip_source_fields(bundle_obj, source)
    bundle_text = json.dumps(bundle_obj, ensure_ascii=False)
    return (
        f"Here is the evidence bundle JSON for cluster {cluster_id}:\n\n```json\n{bundle_text}\n```"
    )
