"""Alternate prompt-component text for Phase 3 prompt-wording benchmarking.
This file defines named "version sets" of alternate text for one or more
component-registry keys. A version set answers:

    "For a given wording version, what alternate text is available for each
     prompt component?"

It does NOT decide which experiments to run (see ``wording_bench_targets.py``)
and it does NOT contain any runner (see ``bench_orchestrator.py``).


``WORDING_ALTERNATE_SET_REGISTRY`` maps ``source_name -> {component_key: text}``.
Each source maps a component-registry key (CAT, GCR, NPR, UPR, PCC, ...) to exactly
ONE replacement string. If two hypotheses need *different* text for the same
component (e.g. a concise CAT vs a multi-pathway CAT), they must live in different
version sets. Sources may be sparse: a set only needs to contain the components
required by the targets that reference it.

Adding a new alternate wording version requires editing ONLY this file:
    1. Add the alternate text constant(s) below.
    2. Add (or extend) an entry in ``WORDING_ALTERNATE_SET_REGISTRY``.
"""

from __future__ import annotations

from mozzarellm.prompt_components import CLUSTER_ANALYSIS_TASK_MULTI


# =============================================================================
# Alternate text constants — wording_v1
# =============================================================================
# These are the "concise / terse / imperative / qualitative" family of rewrites.
# Each is a drop-in replacement for the canonical text of a single component key.

# CAT (W1): shorter task framing — removes the "pathway is a lens, not the end
# goal" philosophy sentence. Tests whether the elaboration grounds the model's
# objective or merely consumes tokens.
CAT_ALT_V1 = """
MISSION: Analyze this gene cluster and:
1. Identify the dominant biological pathway that explains why these genes cluster together.
2. Categorize ALL genes as ESTABLISHED (known role in pathway), NOVEL_ROLE (characterized elsewhere), or UNCHARACTERIZED (no focused study).
3. Prioritize NOVEL_ROLE and UNCHARACTERIZED genes for follow-up.
"""

# GCR (W3): imperative decision-tree rewrite of the gene-classification rules.
# Tests whether instruction-tuned models follow imperative rules more consistently.
GCR_ALT_V1 = """
Categorize each gene into EXACTLY ONE of three categories using these rules IN ORDER:

STEP A — CATEGORIZE:

ESTABLISHED — assign if: at least one peer-reviewed paper directly demonstrates this gene's role
in the identified pathway (knockout/knockdown phenotype, biochemical interaction, or mechanistic
study). Review articles and guilt-by-association do NOT count.

NOVEL_ROLE — assign if: at least one paper has studied this gene's molecular function, but that
function is in a DIFFERENT pathway. The gene is characterized — just not in this context.

UNCHARACTERIZED — assign if: no paper has focused on this gene's molecular function in any
pathway in human cells. This includes completely unstudied genes, domain/homology-only
annotations, and genes studied only in non-human organisms.

Decision logic (apply in order, stop at first match):
1. Has any paper directly shown a molecular function for this gene? -> No -> UNCHARACTERIZED
2. Is that function in THIS specific pathway? -> Yes -> ESTABLISHED
3. Otherwise -> NOVEL_ROLE

STEP B — SUBCLASSIFY: For every NOVEL_ROLE and UNCHARACTERIZED gene, assign a sub-class
(rules follow in the next section).
"""

# NPR (W4): terse novel-role prioritization rules — prose examples removed, only
# name + one-line definition retained.
NPR_ALT_V1 = """
Sub-classes for NOVEL_ROLE genes (known function, but in a DIFFERENT pathway):
  NO_EVIDENCE — no data linking gene to this pathway.
  INDIRECT_EVIDENCE — logical connection (shared organelle, upstream regulator) but no direct link.
  PARTIAL_EVIDENCE — preliminary data (proteomics, co-expression) suggests a link; no focused study.
  CONTRADICTORY_EVIDENCE — known function is mechanistically incompatible with this pathway.
Assign exactly one sub-class per gene.
"""

# UPR (W4): terse uncharacterized prioritization rules — companion to NPR_ALT_V1.
UPR_ALT_V1 = """
Sub-classes for UNCHARACTERIZED genes (no focused study of molecular function in human cells):
  DARK_GENE — no name, no characterization.
  NASCENT — no standard name, but some preliminary functional data exists.
  ANNOTATED_ONLY — gene name and domain annotations present; no mechanistic study.
  NON_HUMAN_CHARACTERIZED — studied functionally only in non-human organism.
Assign exactly one sub-class per gene.
"""

# PCC (W5): qualitative confidence rubric — removes the percentage thresholds
# (>70%, 50-70%, ...) in favour of qualitative descriptors. Tests whether
# numerical anchors aid calibration or introduce spurious precision.
PCC_ALT_V1 = """
ASSESSING PATHWAY CONFIDENCE:

After identifying candidate pathway(s), evaluate how strongly the cluster supports them:

HIGH CONFIDENCE:
- A clear majority of cluster genes have documented roles in the pathway.
- Multiple anchor genes with strong, direct experimental evidence.
- Functional relationships between genes explain the observed phenotypic clustering.

MEDIUM CONFIDENCE:
- A moderate proportion of genes fit the pathway.
- Some well-established anchor genes plus additional plausible candidates.
- Functional logic is coherent but has gaps or minor inconsistencies.

LOW CONFIDENCE:
- Only a minority of genes fit the pathway.
- Few or no anchor genes; the proposed theme is broad.
- Significant heterogeneity — alternative pathways are equally plausible.

NO COHERENT PATHWAY:
- Genes belong to many unrelated processes; no single pathway explains the clustering.
- Cluster may contain negative control genes.
- Cannot identify a dominant biological process.

If there is no coherent pathway, set:
- "pathway_confidence": "Low"
- "dominant_process": "No coherent biological pathway"
- Explain in the "summary" field.

Remember: honest assessment matters more than forcing a clean answer. Low confidence clusters
may still contain valuable discovery candidates.
"""

# =============================================================================
# Alternate text constants — wording_v2
# =============================================================================
# A separate version set. Currently sparse: it only redefines CAT with the
# multi-pathway framing (W2). It exists as a distinct source because W2 needs a
# DIFFERENT CAT text than W1 (one component key -> one text per source).
# Expand this set with GCR/NPR/UPR/PCC alternates to rerun W3-W5 against a newer
# wording version via ``force_source: wording_v2``.

# CAT (W2): multi-pathway task framing — allows 1-3 pathways (min 3 genes each).
CAT_ALT_V2 = CLUSTER_ANALYSIS_TASK_MULTI


# =============================================================================
# Registry: source_name -> {component_key: alternate_text}
# =============================================================================
# Defined after the constants above so the dict literal can reference them at
# import time.

WORDING_ALTERNATE_SET_REGISTRY: dict[str, dict[str, str]] = {
    "wording_v1": {
        "CAT": CAT_ALT_V1,
        "GCR": GCR_ALT_V1,
        "NPR": NPR_ALT_V1,
        "UPR": UPR_ALT_V1,
        "PCC": PCC_ALT_V1,
    },
    "wording_v2": {
        "CAT": CAT_ALT_V2,
    },
}
