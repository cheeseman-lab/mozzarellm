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

from mozzarellm.prompt_components import (
    CLUSTER_ANALYSIS_TASK_MULTI,
)

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
# Alternate text constants — wording_v1 CoT components
# =============================================================================
# CoT components (cGCR, cPri, cPSC) embed their baseline counterparts at
# import time, so overriding GCR/NPR/UPR/PCC has NO effect on them.  To run
# wording experiments on CoT/stepwise routes (cot, stepwise) the registry must also
# contain standalone CoT-framed replacements keyed by the CoT shorthand.

# cGCR (W3 on CoT routes): imperative CoT gene-categorization step.
# Mirrors GCR_ALT_V1 but wrapped in the CoT framing that cGCR normally uses.
cGCR_ALT_V1 = f"""GENE CATEGORIZATION (cite evidence):
For each gene, assign to exactly one category: ESTABLISHED / NOVEL_ROLE / UNCHARACTERIZED
These are defined according to the following rules: {GCR_ALT_V1}
"""

# cPri (W4 on CoT routes): terse CoT sub-classification step.
# Mirrors NPR_ALT_V1 + UPR_ALT_V1 in the cPri frame.
cPri_ALT_V1 = f"""SUB-CLASSIFICATION:
For NOVEL_ROLE genes, assign one sub-class: NO_EVIDENCE / INDIRECT_EVIDENCE / PARTIAL_EVIDENCE / CONTRADICTORY_EVIDENCE
These are defined according to the following rules: {NPR_ALT_V1}
For UNCHARACTERIZED genes, assign one sub-class: DARK_GENE / NASCENT / ANNOTATED_ONLY / NON_HUMAN_CHARACTERIZED
These are defined according to the following rules: {UPR_ALT_V1}
Cite specific annotations that inform each classification."""

# cPSC (W5 on CoT routes): qualitative CoT pathway-selection step.
# Mirrors PCC_ALT_V1 in the cPSC frame.
cPSC_ALT_V1 = f"""PATHWAY SELECTION:
Once you have identified candidate pathway(s), evaluate how well EACH pathway explains the cluster using
these stringent criteria based on what percentage of genes fit the proposed pathway: {PCC_ALT_V1}
Now, select a dominant pathway based on:
  * Number of established genes with direct roles
  * Coherence of functional relationships
  * Quality of supporting evidence"""


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
# Alternate text constants — wording_v4 (conceptual-frame variants per component)
# =============================================================================
# Build-up program: per-component triad of conceptually-distinct framings under
# a shared scaffold so the only varying factor is the conceptual frame.
# Compared against W12 (schema-only floor) and W0 (canonical ceiling).

# CAT — three framings under shared scaffold "...identify [X] and establish how
# each gene contributes to this categorization."
CAT_FRAME_PATHWAY = """
MISSION: These genes were clustered by phenotypic similarity. Your goal is to identify the dominant biological pathway and establish how each gene contributes to this categorization.
"""

CAT_FRAME_MECHANISM = """
MISSION: These genes were clustered by phenotypic similarity. Your goal is to identify the shared molecular mechanism and establish how each gene contributes to this categorization.
"""

CAT_FRAME_GUARDED = """
MISSION: These genes were clustered by phenotypic similarity. Your goal is to identify the dominant biological pathway — or, if the cluster reflects noise rather than a coherent phenotype, note its absence — and establish how each gene contributes to this categorization.
"""

# GCR — three-level ladder (simple → +decision tree → +guard), all bundle-grounded.
# V2 - V1 measures the value of the decision tree procedure.
# V3 - V2 measures the value of the anti-guilt-by-association guard.
GCR_FRAME_SIMPLE = """
PRECONDITION: This step applies ONLY when a coherent biological pathway has been identified for the cluster. If no coherent pathway exists, leave `established_genes`, `novel_role_genes`, and `uncharacterized_genes` empty and skip this step — per-gene classification relative to a nonexistent pathway is undefined.

TASK: For each gene in the cluster, categorize as ESTABLISHED, NOVEL_ROLE, or UNCHARACTERIZED based on the evidence provided in the gene's bundle annotation.
- ESTABLISHED: the annotation directly demonstrates the gene's role in the identified pathway.
- NOVEL_ROLE: the annotation describes a documented function in a different pathway.
- UNCHARACTERIZED: the annotation is empty, generic, or limited to homology/domain information.
"""

GCR_FRAME_DECISION_TREE = """
PRECONDITION: This step applies ONLY when a coherent biological pathway has been identified for the cluster. If no coherent pathway exists, leave `established_genes`, `novel_role_genes`, and `uncharacterized_genes` empty and skip this step — per-gene classification relative to a nonexistent pathway is undefined.

TASK: For each gene in the cluster, categorize as ESTABLISHED, NOVEL_ROLE, or UNCHARACTERIZED based on the evidence provided in the gene's bundle annotation.
- ESTABLISHED: the annotation directly demonstrates the gene's role in the identified pathway.
- NOVEL_ROLE: the annotation describes a documented function in a different pathway.
- UNCHARACTERIZED: the annotation is empty, generic, or limited to homology/domain information.

Gating procedure (apply in order):
1. Does the bundle's annotation describe the gene's molecular function? → No → UNCHARACTERIZED.
2. Does that annotation describe a role in THIS pathway? → Yes → ESTABLISHED.
3. Otherwise → NOVEL_ROLE.
"""

GCR_FRAME_GUARDED = """
PRECONDITION: This step applies ONLY when a coherent biological pathway has been identified for the cluster. If no coherent pathway exists, leave `established_genes`, `novel_role_genes`, and `uncharacterized_genes` empty and skip this step — per-gene classification relative to a nonexistent pathway is undefined.

TASK: For each gene in the cluster, categorize as ESTABLISHED, NOVEL_ROLE, or UNCHARACTERIZED based on the evidence provided in the gene's bundle annotation.
- ESTABLISHED: the annotation directly demonstrates the gene's role in the identified pathway.
- NOVEL_ROLE: the annotation describes a documented function in a different pathway.
- UNCHARACTERIZED: the annotation is empty, generic, or limited to homology/domain information.

Gating procedure (apply in order):
1. Does the bundle's annotation describe the gene's molecular function? → No → UNCHARACTERIZED.
2. Does that annotation describe a role in THIS pathway? → Yes → ESTABLISHED.
3. Otherwise → NOVEL_ROLE.

When uncertain whether the bundle's evidence rises to the ESTABLISHED standard, default to NOVEL_ROLE — promoting on weak or indirect evidence misleads downstream follow-up.
"""

# NPR — three-level ladder (simple → +procedure → +guard), bundle-grounded.
NPR_FRAME_SIMPLE = """
PRECONDITION: This step applies ONLY to genes already categorized as NOVEL_ROLE per the gene classification rules. If no genes were categorized as NOVEL_ROLE (because no coherent pathway exists, or no genes fit the criteria), do nothing — no sub-classification is needed.

SUB-CLASSIFICATION (NOVEL_ROLE genes only): assign exactly one sub-class based on how the bundle's annotation relates to the identified pathway.
- NO_EVIDENCE: nothing in the annotation links the gene to this pathway.
- INDIRECT_EVIDENCE: the annotation shows a logical connection (shared organelle, upstream regulator) but no direct experimental link.
- PARTIAL_EVIDENCE: the annotation hints at a link (proteomics hit, co-expression) without focused mechanistic study.
- CONTRADICTORY_EVIDENCE: the annotation describes a function incompatible with this pathway.
"""

NPR_FRAME_DECISION_TREE = """
PRECONDITION: This step applies ONLY to genes already categorized as NOVEL_ROLE per the gene classification rules. If no genes were categorized as NOVEL_ROLE (because no coherent pathway exists, or no genes fit the criteria), do nothing — no sub-classification is needed.

SUB-CLASSIFICATION (NOVEL_ROLE genes only): assign exactly one sub-class based on how the bundle's annotation relates to the identified pathway.
- NO_EVIDENCE: nothing in the annotation links the gene to this pathway.
- INDIRECT_EVIDENCE: the annotation shows a logical connection (shared organelle, upstream regulator) but no direct experimental link.
- PARTIAL_EVIDENCE: the annotation hints at a link (proteomics hit, co-expression) without focused mechanistic study.
- CONTRADICTORY_EVIDENCE: the annotation describes a function incompatible with this pathway.

Procedure (apply in order):
1. Does the annotation describe a function incompatible with this pathway? → CONTRADICTORY_EVIDENCE.
2. Does the annotation hint at a preliminary link (proteomics/co-expression/correlative data)? → PARTIAL_EVIDENCE.
3. Does the annotation suggest a logical connection without direct experimental link? → INDIRECT_EVIDENCE.
4. Otherwise → NO_EVIDENCE.
"""

NPR_FRAME_GUARDED = """
PRECONDITION: This step applies ONLY to genes already categorized as NOVEL_ROLE per the gene classification rules. If no genes were categorized as NOVEL_ROLE (because no coherent pathway exists, or no genes fit the criteria), do nothing — no sub-classification is needed.

SUB-CLASSIFICATION (NOVEL_ROLE genes only): assign exactly one sub-class based on how the bundle's annotation relates to the identified pathway.
- NO_EVIDENCE: nothing in the annotation links the gene to this pathway.
- INDIRECT_EVIDENCE: the annotation shows a logical connection (shared organelle, upstream regulator) but no direct experimental link.
- PARTIAL_EVIDENCE: the annotation hints at a link (proteomics hit, co-expression) without focused mechanistic study.
- CONTRADICTORY_EVIDENCE: the annotation describes a function incompatible with this pathway.

Procedure (apply in order):
1. Does the annotation describe a function incompatible with this pathway? → CONTRADICTORY_EVIDENCE.
2. Does the annotation hint at a preliminary link (proteomics/co-expression/correlative data)? → PARTIAL_EVIDENCE.
3. Does the annotation suggest a logical connection without direct experimental link? → INDIRECT_EVIDENCE.
4. Otherwise → NO_EVIDENCE.

When uncertain between two adjacent levels, prefer the weaker classification. Overcrediting weak signal as PARTIAL or INDIRECT inflates apparent pathway connection.
"""

# UPR — three-level ladder (simple → +procedure → +guard), bundle-grounded.
UPR_FRAME_SIMPLE = """
PRECONDITION: This step applies ONLY to genes already categorized as UNCHARACTERIZED per the gene classification rules. If no genes were categorized as UNCHARACTERIZED (because no coherent pathway exists, or all genes have documented function), do nothing — no sub-classification is needed.

SUB-CLASSIFICATION (UNCHARACTERIZED genes only): assign exactly one sub-class based on the bundle's annotation for the gene.
- DARK_GENE: no name and no functional characterization in the annotation.
- NASCENT: no standard name, but the annotation contains preliminary functional data.
- ANNOTATED_ONLY: has a gene name and domain/motif annotations, but no mechanistic study.
- NON_HUMAN_CHARACTERIZED: the annotation describes functional studies only in non-human organisms.
"""

UPR_FRAME_DECISION_TREE = """
PRECONDITION: This step applies ONLY to genes already categorized as UNCHARACTERIZED per the gene classification rules. If no genes were categorized as UNCHARACTERIZED (because no coherent pathway exists, or all genes have documented function), do nothing — no sub-classification is needed.

SUB-CLASSIFICATION (UNCHARACTERIZED genes only): assign exactly one sub-class based on the bundle's annotation for the gene.
- DARK_GENE: no name and no functional characterization in the annotation.
- NASCENT: no standard name, but the annotation contains preliminary functional data.
- ANNOTATED_ONLY: has a gene name and domain/motif annotations, but no mechanistic study.
- NON_HUMAN_CHARACTERIZED: the annotation describes functional studies only in non-human organisms.

Procedure (apply in order):
1. Does the annotation describe functional studies in a non-human organism? → NON_HUMAN_CHARACTERIZED.
2. Does the gene have a standard name with domain/motif annotation but no mechanistic study? → ANNOTATED_ONLY.
3. Does the annotation contain any preliminary functional data (even without a standard name)? → NASCENT.
4. Otherwise → DARK_GENE.
"""

UPR_FRAME_GUARDED = """
PRECONDITION: This step applies ONLY to genes already categorized as UNCHARACTERIZED per the gene classification rules. If no genes were categorized as UNCHARACTERIZED (because no coherent pathway exists, or all genes have documented function), do nothing — no sub-classification is needed.

SUB-CLASSIFICATION (UNCHARACTERIZED genes only): assign exactly one sub-class based on the bundle's annotation for the gene.
- DARK_GENE: no name and no functional characterization in the annotation.
- NASCENT: no standard name, but the annotation contains preliminary functional data.
- ANNOTATED_ONLY: has a gene name and domain/motif annotations, but no mechanistic study.
- NON_HUMAN_CHARACTERIZED: the annotation describes functional studies only in non-human organisms.

Procedure (apply in order):
1. Does the annotation describe functional studies in a non-human organism? → NON_HUMAN_CHARACTERIZED.
2. Does the gene have a standard name with domain/motif annotation but no mechanistic study? → ANNOTATED_ONLY.
3. Does the annotation contain any preliminary functional data (even without a standard name)? → NASCENT.
4. Otherwise → DARK_GENE.

When uncertain, prefer the lower-information sub-class (DARK_GENE over NASCENT, NASCENT over ANNOTATED_ONLY). Having a domain prediction or a name should not, by itself, count as characterization — actual functional data must be present in the annotation.
"""

# PCC — three-level ladder (simple → +detailed criteria → +guard).
# Graded judgment, not categorical, so V2 adds rich per-level criteria rather
# than a decision tree. V3 adds an anti-fabrication guard that frames the
# "no coherent pathway" outcome as a valuable finding, not a failure.
PCC_FRAME_SIMPLE = """
PATHWAY CONFIDENCE: assess how well the identified pathway explains the cluster, based on the fraction of genes whose bundle annotations are consistent with the pathway.
- High: most genes have annotations consistent with the pathway.
- Medium: a majority are consistent; some genes are unclear or unrelated.
- Low: few genes are clearly consistent — or, if too few, the cluster lacks a coherent pathway entirely (use "dominant_process": "No coherent biological pathway", and leave `established_genes`, `novel_role_genes`, and `uncharacterized_genes` empty since per-gene classification relative to a nonexistent pathway is undefined).
"""

PCC_FRAME_DETAILED = """
PATHWAY CONFIDENCE: assess how well the identified pathway explains the cluster, based on the bundle's gene annotations.

High confidence:
- >70% of cluster genes have annotations consistent with the pathway
- Multiple well-established members with direct functional evidence
- Clear functional relationships explain the phenotypic clustering

Medium confidence:
- 50-70% of cluster genes are consistent with the pathway
- Some established members alongside plausible additional candidates
- Functional relationship is plausible but has gaps

Low confidence:
- 30-50% of cluster genes are clearly consistent
- Few established members; themes may be broad or general
- Significant heterogeneity in gene functions

No coherent pathway (use Low confidence, set "dominant_process": "No coherent biological pathway", and leave `established_genes`, `novel_role_genes`, and `uncharacterized_genes` empty — per-gene classification relative to a nonexistent pathway is undefined):
- <30% of cluster genes fit any proposed pathway
- Cluster contains many unrelated functions or nontargeting controls
"""

# =============================================================================
# VER — verification step (Stage 6).
# Appended to PCC text in wording_v6_stage6_* sources so it lands between
# PCC and OUTPUT_FORMAT_JSON in the assembled prompt. Conceptually a separate
# "second pass" step that explicitly verifies every gene in the input cluster
# appears in one of the classification arrays.
# =============================================================================

VER_SIMPLE = """

VERIFICATION: After completing your classification, confirm every gene listed in the input cluster genes appears in exactly one of established_genes, novel_role_genes, or uncharacterized_genes. If any gene was omitted, add it to the appropriate category.
"""

VER_PRECONDITION = """

VERIFICATION: This step applies ONLY when a coherent biological pathway was identified. If a coherent pathway exists, confirm every gene in the input cluster appears in exactly one classification array; add any omitted gene to the appropriate category. If no coherent pathway was identified, the classification arrays should remain empty — skip verification.
"""

VER_GUARDED = """

VERIFICATION: This step applies ONLY when a coherent biological pathway was identified. If a coherent pathway exists, confirm every gene in the input cluster appears in your output and add any omitted gene to the most defensible category. If no coherent pathway was identified, leave classification arrays empty. When uncertain whether to add an omitted gene, prefer leaving it unclassified — overcrediting a forced classification is worse than acknowledging a gap.
"""


PCC_FRAME_GUARDED = """
PATHWAY CONFIDENCE: assess how well the identified pathway explains the cluster, based on the bundle's gene annotations.

High confidence:
- >70% of cluster genes have annotations consistent with the pathway
- Multiple well-established members with direct functional evidence
- Clear functional relationships explain the phenotypic clustering

Medium confidence:
- 50-70% of cluster genes are consistent with the pathway
- Some established members alongside plausible additional candidates
- Functional relationship is plausible but has gaps

Low confidence:
- 30-50% of cluster genes are clearly consistent
- Few established members; themes may be broad or general
- Significant heterogeneity in gene functions

No coherent pathway (use Low confidence, set "dominant_process": "No coherent biological pathway", and leave `established_genes`, `novel_role_genes`, and `uncharacterized_genes` empty — per-gene classification relative to a nonexistent pathway is undefined):
- <30% of cluster genes fit any proposed pathway
- Cluster contains many unrelated functions or nontargeting controls

When uncertain between two adjacent levels, prefer the lower confidence. Flagging that a cluster lacks a coherent pathway is a valuable finding — clusters can be heterogeneous, noisy, or artifactual, and an honest "no coherent pathway" call is more useful than a forced label.
"""


# =============================================================================
# CoT-native ports of W23 decisions (Phase 4 mode anchor).
# cGCR embeds GCR_FRAME_DECISION_TREE; cPri embeds NPR/UPR_FRAME_SIMPLE;
# cPSC embeds PCC_FRAME_SIMPLE + VER_PRECONDITION. Same tuning as W23 zero-shot,
# just placed in CoT/stepwise component slots so routes cot/stepwise use W23's calls.
# =============================================================================

cGCR_W23_DECISION_TREE = f"""GENE CATEGORIZATION (cite evidence):
For each gene, assign to exactly one category: ESTABLISHED / NOVEL_ROLE / UNCHARACTERIZED
These are defined according to the following rules: {GCR_FRAME_DECISION_TREE}
"""

cPri_W23_SIMPLE = f"""SUB-CLASSIFICATION:
For NOVEL_ROLE genes, assign one sub-class:
NO_EVIDENCE / INDIRECT_EVIDENCE / PARTIAL_EVIDENCE / CONTRADICTORY_EVIDENCE.
These are defined according to the following rules: {NPR_FRAME_SIMPLE}
For UNCHARACTERIZED genes, assign one sub-class:
DARK_GENE / NASCENT / ANNOTATED_ONLY / NON_HUMAN_CHARACTERIZED.
These are defined according to the following rules: {UPR_FRAME_SIMPLE}
Cite specific annotations that inform each classification."""

cPSC_W23_SIMPLE_PRECONDITION = f"""PATHWAY SELECTION:
Once you have identified candidate pathway(s), evaluate how well EACH pathway explains
the cluster using the following criteria: {PCC_FRAME_SIMPLE}
Now, select a dominant pathway based on:
  * Number of established genes with direct roles
  * Coherence of functional relationships
  * Quality of supporting evidence
{VER_PRECONDITION}"""


# =============================================================================
# Registry: source_name -> {component_key: alternate_text}
# =============================================================================
# Defined after the constants above so the dict literal can reference them at
# import time.

WORDING_ALTERNATE_SET_REGISTRY: dict[str, dict[str, str]] = {
    # ----- Alexa's LOO swap sources (kept) -----------------------------------
    "wording_v1": {
        "CAT": CAT_ALT_V1,
        "GCR": GCR_ALT_V1,
        "NPR": NPR_ALT_V1,
        "UPR": UPR_ALT_V1,
        "PCC": PCC_ALT_V1,
        # CoT/stepwise equivalents — needed for overrides on cot/stepwise routes
        "cGCR": cGCR_ALT_V1,
        "cPri": cPri_ALT_V1,
        "cPSC": cPSC_ALT_V1,
    },
    "wording_v2": {
        "CAT": CAT_ALT_V2,
    },
    # ----- wording_v4 (build-up floor + Stage 1 CAT framings) ----------------
    # Floor: every prose component excised; OUTPUT_FORMAT_JSON stays canonical.
    "wording_v4_floor": {"CAT": "", "GCR": "", "NPR": "", "UPR": "", "PCC": ""},
    # Stage 1 CAT framings — the only single-component test in the build-up
    # (subsequent stages layer on the locked CAT winner, not on the floor).
    "wording_v4_cat_pathway": {
        "CAT": CAT_FRAME_PATHWAY,
        "GCR": "",
        "NPR": "",
        "UPR": "",
        "PCC": "",
    },
    "wording_v4_cat_mechanism": {
        "CAT": CAT_FRAME_MECHANISM,
        "GCR": "",
        "NPR": "",
        "UPR": "",
        "PCC": "",
    },
    "wording_v4_cat_guarded": {
        "CAT": CAT_FRAME_GUARDED,
        "GCR": "",
        "NPR": "",
        "UPR": "",
        "PCC": "",
    },
    # =========================================================================
    # wording_v6 — sequential build-up stacks (Stages 2+).
    # Each stage locks the winning components from prior stages and tests 3
    # variants of the current stage's component on top.
    #
    # Stage 1 locked: CAT = cat_guarded (best on pathway_semantic_match).
    # Stage 2 tests GCR variants on top of cat_guarded. Targets W10, W11, W12.
    # =========================================================================
    "wording_v6_stage2_gcr_simple": {
        "CAT": CAT_FRAME_GUARDED,
        "GCR": GCR_FRAME_SIMPLE,
        "NPR": "",
        "UPR": "",
        "PCC": "",
    },
    "wording_v6_stage2_gcr_decision_tree": {
        "CAT": CAT_FRAME_GUARDED,
        "GCR": GCR_FRAME_DECISION_TREE,
        "NPR": "",
        "UPR": "",
        "PCC": "",
    },
    "wording_v6_stage2_gcr_guarded": {
        "CAT": CAT_FRAME_GUARDED,
        "GCR": GCR_FRAME_GUARDED,
        "NPR": "",
        "UPR": "",
        "PCC": "",
    },
    # Stage 2 locked: GCR = gcr_decision_tree (best on pathway_semantic + subclass).
    # Stage 3 tests NPR variants on top of cat_guarded + gcr_decision_tree.
    # Targets W25, W26, W27.
    "wording_v6_stage3_npr_simple": {
        "CAT": CAT_FRAME_GUARDED,
        "GCR": GCR_FRAME_DECISION_TREE,
        "NPR": NPR_FRAME_SIMPLE,
        "UPR": "",
        "PCC": "",
    },
    "wording_v6_stage3_npr_decision_tree": {
        "CAT": CAT_FRAME_GUARDED,
        "GCR": GCR_FRAME_DECISION_TREE,
        "NPR": NPR_FRAME_DECISION_TREE,
        "UPR": "",
        "PCC": "",
    },
    "wording_v6_stage3_npr_guarded": {
        "CAT": CAT_FRAME_GUARDED,
        "GCR": GCR_FRAME_DECISION_TREE,
        "NPR": NPR_FRAME_GUARDED,
        "UPR": "",
        "PCC": "",
    },
    # Stage 3 locked: NPR = npr_simple (best pathway_semantic + coverage; subclass
    # regressed 2.8 pp vs Stage 2 — documented in docs/benchmark_program.md).
    # Stage 4 tests UPR variants on top of the locked stack.
    # Targets W28, W29, W30.
    "wording_v6_stage4_upr_simple": {
        "CAT": CAT_FRAME_GUARDED,
        "GCR": GCR_FRAME_DECISION_TREE,
        "NPR": NPR_FRAME_SIMPLE,
        "UPR": UPR_FRAME_SIMPLE,
        "PCC": "",
    },
    "wording_v6_stage4_upr_decision_tree": {
        "CAT": CAT_FRAME_GUARDED,
        "GCR": GCR_FRAME_DECISION_TREE,
        "NPR": NPR_FRAME_SIMPLE,
        "UPR": UPR_FRAME_DECISION_TREE,
        "PCC": "",
    },
    "wording_v6_stage4_upr_guarded": {
        "CAT": CAT_FRAME_GUARDED,
        "GCR": GCR_FRAME_DECISION_TREE,
        "NPR": NPR_FRAME_SIMPLE,
        "UPR": UPR_FRAME_GUARDED,
        "PCC": "",
    },
    # Stage 4 locked: UPR = upr_simple (best UNCHARACTERIZED-subclass + coverage).
    # Stage 5 tests PCC variants — explicit abstention/off-ramp component.
    # Targets W31, W32, W33.
    "wording_v6_stage5_pcc_simple": {
        "CAT": CAT_FRAME_GUARDED,
        "GCR": GCR_FRAME_DECISION_TREE,
        "NPR": NPR_FRAME_SIMPLE,
        "UPR": UPR_FRAME_SIMPLE,
        "PCC": PCC_FRAME_SIMPLE,
    },
    "wording_v6_stage5_pcc_detailed": {
        "CAT": CAT_FRAME_GUARDED,
        "GCR": GCR_FRAME_DECISION_TREE,
        "NPR": NPR_FRAME_SIMPLE,
        "UPR": UPR_FRAME_SIMPLE,
        "PCC": PCC_FRAME_DETAILED,
    },
    "wording_v6_stage5_pcc_guarded": {
        "CAT": CAT_FRAME_GUARDED,
        "GCR": GCR_FRAME_DECISION_TREE,
        "NPR": NPR_FRAME_SIMPLE,
        "UPR": UPR_FRAME_SIMPLE,
        "PCC": PCC_FRAME_GUARDED,
    },
    # Stage 5 locked: PCC = pcc_simple (best abstention + classification + subclass).
    # Optimal so far: W19 = cat_guarded + gcr_dt + npr_simple + upr_simple + pcc_simple.
    # Stage 6 tests appending a VER (verification) step that re-checks
    # gene-level coverage. Conceptually equivalent to CoT's cVer step but
    # promoted into zero-shot. Targets W22, W23, W24.
    "wording_v6_stage6_ver_simple": {
        "CAT": CAT_FRAME_GUARDED,
        "GCR": GCR_FRAME_DECISION_TREE,
        "NPR": NPR_FRAME_SIMPLE,
        "UPR": UPR_FRAME_SIMPLE,
        "PCC": PCC_FRAME_SIMPLE + VER_SIMPLE,
    },
    "wording_v6_stage6_ver_precondition": {
        "CAT": CAT_FRAME_GUARDED,
        "GCR": GCR_FRAME_DECISION_TREE,
        "NPR": NPR_FRAME_SIMPLE,
        "UPR": UPR_FRAME_SIMPLE,
        "PCC": PCC_FRAME_SIMPLE + VER_PRECONDITION,
        # CoT-native ports — carry W23 decisions into cGCR/cPri/cPSC so the
        # same text applies when W23 is used on route cot/stepwise (mode axis).
        "cGCR": cGCR_W23_DECISION_TREE,
        "cPri": cPri_W23_SIMPLE,
        "cPSC": cPSC_W23_SIMPLE_PRECONDITION,
    },
    "wording_v6_stage6_ver_guarded": {
        "CAT": CAT_FRAME_GUARDED,
        "GCR": GCR_FRAME_DECISION_TREE,
        "NPR": NPR_FRAME_SIMPLE,
        "UPR": UPR_FRAME_SIMPLE,
        "PCC": PCC_FRAME_SIMPLE + VER_GUARDED,
    },
}
