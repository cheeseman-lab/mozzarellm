"""Reasoned candidate framings for the build-up prompt walkup.

The walkup starts from a blank prompt (W0) and adds one component per stage,
testing 3 NEW framings per component (no canonical option) and carrying the
winner forward. Directions are informed by the first flat run:

- GCR / NPR / PCC reframings genuinely helped their goal metric, so we test the
  validated 3-level ladders (simple -> +procedure/detail -> +guard).
- CAT reframings were ~neutral-to-slightly-negative, so we test 3 distinct
  task-lens directions rather than a ladder.
- UPR reframings HURT unchar-subclass (-0.11 to -0.31) -- the v6 rules over-demote
  named/annotated genes toward DARK_GENE. So all three UPR candidates are
  re-authored corrections.

CANDIDATES[component] = [(candidate_id, rationale, text), ...]  -- exactly 3 each.
Each `text` is the full component block that fills that slot in the assembled
single_call prompt.
"""

from architecture_benchmarking_workflow.bench_wording_alternates import (
    GCR_FRAME_DECISION_TREE,
    GCR_FRAME_GUARDED,
    GCR_FRAME_SIMPLE,
    NPR_FRAME_DECISION_TREE,
    NPR_FRAME_SIMPLE,
    PCC_FRAME_DETAILED,
    PCC_FRAME_GUARDED,
    PCC_FRAME_SIMPLE,
)

# =============================================================================
# Newly authored texts (where reuse is inappropriate)
# =============================================================================
# All CAT candidates are multi-pathway-AWARE: they license 1-3 pathways as a guard
# against forcing a heterogeneous cluster into one (matching the output-JSON guard
# in OUTPUT_FORMAT_JSON). Multi-pathway is a constant, not a competing candidate;
# the three lenses vary the task framing around it.

# CAT — lean task statement; tests whether the elaboration earns its tokens.
CAT_CONCISE = """
MISSION: Analyze this gene cluster and:
1. Identify the biological pathway(s) — a single dominant process, or 2-3 if the cluster genuinely
   spans several — that explain why these genes cluster together.
2. Categorize ALL genes as ESTABLISHED (known role in the pathway), NOVEL_ROLE (characterized
   elsewhere), or UNCHARACTERIZED (no focused study).
3. Prioritize NOVEL_ROLE and UNCHARACTERIZED genes for follow-up.
"""

# CAT — lead with the recall-first mission (understudied genes are the deliverable)
# and license an honest "no coherent pathway" from the very first component.
CAT_DISCOVERY_FIRST = """
MISSION: Functional genomics experiments cluster genes by phenotypic similarity. The deliverable is
to surface UNDERSTUDIED genes (UNCHARACTERIZED and NOVEL_ROLE) that merit experimental follow-up.
To do that:
1. Identify the pathway(s) that best explain why these genes cluster — a single dominant process, or
   2-3 if the cluster genuinely spans several. This is the lens for discovery, not the end goal.
2. Categorize ALL genes relative to that pathway (ESTABLISHED / NOVEL_ROLE / UNCHARACTERIZED).
3. Prioritize the understudied genes for follow-up.

If the genes do not share a coherent pathway, say so honestly rather than forcing a label — a clear
"no coherent pathway" call is a valid and useful outcome.
"""

# CAT — anchor on the pathway first, since every downstream categorization depends on it.
CAT_PATHWAY_ANCHORED = """
MISSION: Every downstream call depends on correctly identifying what unites this cluster, so anchor
there first.
1. Determine the biological pathway(s) that explain the cluster — commit to a single dominant
   process where one clearly fits, or 2-3 distinct processes if the cluster genuinely spans them. If
   no process explains a substantial share of the genes, declare no coherent pathway.
2. Categorize ALL genes against that pathway (ESTABLISHED / NOVEL_ROLE / UNCHARACTERIZED).
3. Prioritize UNCHARACTERIZED and NOVEL_ROLE genes for follow-up.
"""

# UPR correction A — the v6 guard demoted named, domain-annotated genes to DARK/NASCENT. Make the
# name+domain signal count: a named gene with domain annotation is ANNOTATED_ONLY, not DARK.
UPR_NAME_RESPECTING = """
PRECONDITION: This step applies ONLY to genes already categorized as UNCHARACTERIZED per the gene
classification rules. If no genes were categorized as UNCHARACTERIZED, do nothing.

SUB-CLASSIFICATION (UNCHARACTERIZED genes only): assign exactly one sub-class from the gene's bundle
annotation. A standard gene name plus domain/motif annotation IS meaningful signal — do NOT demote a
named, domain-annotated gene to DARK_GENE merely because no mechanistic study exists.
- DARK_GENE: no standard name AND no domain/functional annotation whatsoever.
- NASCENT: no standard name, but the annotation contains preliminary functional data.
- ANNOTATED_ONLY: has a standard name and/or domain/motif annotation but no mechanistic study — this
  is the default for named genes that lack a focused functional study.
- NON_HUMAN_CHARACTERIZED: the annotation describes functional studies only in non-human organisms.
"""

# UPR correction B — rank by the HIGHEST level of evidence actually present, with no downward
# tiebreak (the v6 "prefer lower" guard was the likely source of the unchar-subclass loss).
UPR_EVIDENCE_LADDER = """
PRECONDITION: This step applies ONLY to genes already categorized as UNCHARACTERIZED per the gene
classification rules. If no genes were categorized as UNCHARACTERIZED, do nothing.

SUB-CLASSIFICATION (UNCHARACTERIZED genes only): assign the sub-class matching the HIGHEST level of
evidence present in the annotation. Evaluate in order and assign the first that matches; do not bias
toward lower-information classes.
- NON_HUMAN_CHARACTERIZED: functional studies described only in a non-human organism.
- ANNOTATED_ONLY: a standard name with domain/motif annotation, no mechanistic study.
- NASCENT: no standard name, but some preliminary functional data present.
- DARK_GENE: none of the above — no name, no domain annotation, no functional data.
"""

# UPR correction C — revert to the canonical 4-subclass definitions (which scored 0.714), just
# grounded in the bundle annotation and with no aggressive guard.
UPR_CANONICAL_GROUNDED = """
PRECONDITION: This step applies ONLY to genes already categorized as UNCHARACTERIZED per the gene
classification rules. If no genes were categorized as UNCHARACTERIZED, do nothing.

SUB-CLASSIFICATION (UNCHARACTERIZED genes only): assign exactly one sub-class based on the gene's
bundle annotation.
- DARK_GENE: no name, no functional characterization whatsoever.
- NASCENT: no standard name, but some preliminary functional data exists.
- ANNOTATED_ONLY: has a gene name and domain/motif annotations, but no mechanistic study.
- NON_HUMAN_CHARACTERIZED: functionally studied in a non-human organism only.
Assign exactly one sub-class per gene.
"""

# NPR correction — two-sided, diagnostic-driven. The model reproduces the conservative reviewer:
# its dominant error is INDIRECT under-credited to NO (7 genes, chromatin/RNA genes in
# chromatin/RNA clusters). A blanket "shared context = INDIRECT" (the earlier context_anchored)
# recovered those but OVERSHOT — it collapsed correct PARTIAL genes (WDR61, ZC3H18) down to
# INDIRECT and over-promoted unrelated NO genes (KDM4A, YEATS4, ZWINT) up to INDIRECT, netting
# ~zero. This reframes the NO/INDIRECT boundary as same-process vs unrelated-process AND adds an
# explicit PARTIAL guard (data touching the pathway stays PARTIAL) and an unrelated-process guard
# (a shared generic compartment is not same-process) to hold both rungs.
NPR_PROCESS_SCOPED = """
PRECONDITION: This step applies ONLY to genes already categorized as NOVEL_ROLE per the gene classification rules. If no genes were categorized as NOVEL_ROLE (because no coherent pathway exists, or no genes fit the criteria), do nothing — no sub-classification is needed.

SUB-CLASSIFICATION (NOVEL_ROLE genes only): assign exactly one sub-class based on how the bundle's annotation relates to the identified pathway.
- NO_EVIDENCE: the gene's characterized function is in a process UNRELATED to this pathway.
- INDIRECT_EVIDENCE: the gene acts in the SAME broad process or machinery as the pathway (e.g. a chromatin factor in a chromatin/centromere cluster; an RNA-processing factor in an RNA cluster) but the annotation reports no experimental link to the specific pathway.
- PARTIAL_EVIDENCE: the annotation reports actual data touching this pathway — a physical interaction, proteomics/co-IP hit, co-expression, or a functional assay — but no focused mechanistic study.
- CONTRADICTORY_EVIDENCE: the annotation describes a function incompatible with this pathway.

Procedure (apply in order):
1. Does the annotation describe a function incompatible with this pathway? → CONTRADICTORY_EVIDENCE.
2. Does the annotation report experimental data touching this pathway (interaction, proteomics/co-IP, co-expression, functional assay)? → PARTIAL_EVIDENCE. Do NOT down-rank a gene that has such data to INDIRECT.
3. Does the gene operate in the same broad process or machinery as the pathway, with no such data? → INDIRECT_EVIDENCE.
4. Otherwise (function is in an unrelated process) → NO_EVIDENCE. Sharing only a generic compartment (e.g. "nuclear", "cytoplasmic") is NOT the same process — keep NO_EVIDENCE.
"""

# =============================================================================
# Candidate banks (3 per component, each with a rationale)
# =============================================================================

CANDIDATES: dict[str, list[tuple[str, str, str]]] = {
    "CAT": [
        (
            "concise",
            "lean task statement (multi-pathway-aware) — tests whether the elaboration grounds the model or just spends tokens",
            CAT_CONCISE,
        ),
        (
            "discovery_first",
            "lead with the recall-first mission + honest 'no coherent pathway' (multi-pathway-aware)",
            CAT_DISCOVERY_FIRST,
        ),
        (
            "pathway_anchored",
            "anchor on rigorously fixing the pathway(s) first, since every downstream call depends on it (multi-pathway-aware)",
            CAT_PATHWAY_ANCHORED,
        ),
    ],
    "GCR": [
        ("simple", "definitions only — baseline categorization rules", GCR_FRAME_SIMPLE),
        (
            "decision_tree",
            "ordered gating procedure cuts category ambiguity — the first-run winner",
            GCR_FRAME_DECISION_TREE,
        ),
        (
            "guarded",
            "decision-tree + default-to-NOVEL-when-uncertain — anti-overcrediting, recall-first",
            GCR_FRAME_GUARDED,
        ),
    ],
    "NPR": [
        (
            "simple",
            "flat NO/INDIRECT/PARTIAL/CONTRADICTORY definitions — baseline",
            NPR_FRAME_SIMPLE,
        ),
        (
            "decision_tree",
            "ordered procedure over the four evidence levels — first-run winner",
            NPR_FRAME_DECISION_TREE,
        ),
        (
            "process_scoped",
            "two-sided: reframes NO/INDIRECT as unrelated-vs-same-process + guards PARTIAL (data stays PARTIAL) and NO (generic compartment isn't same-process) — recovers INDIRECT under-credits without collapsing PARTIAL",
            NPR_PROCESS_SCOPED,
        ),
    ],
    "UPR": [
        (
            "name_respecting",
            "CORRECTED: a name+domain gene is ANNOTATED_ONLY, not DARK — undoes the v6 over-demotion that broke unchar-subclass",
            UPR_NAME_RESPECTING,
        ),
        (
            "evidence_ladder",
            "CORRECTED: rank by highest evidence present with no downward tiebreak (the likely source of the v6 loss)",
            UPR_EVIDENCE_LADDER,
        ),
        (
            "canonical_grounded",
            "CORRECTED: revert to the canonical 4-subclass rules (scored 0.714), bundle-grounded, no guard",
            UPR_CANONICAL_GROUNDED,
        ),
    ],
    "PCC": [
        ("simple", "qualitative high/medium/low by fraction consistent — lean", PCC_FRAME_SIMPLE),
        ("detailed", "explicit >70/50-70/30-50 percent thresholds per level", PCC_FRAME_DETAILED),
        (
            "guarded",
            "thresholds + prefer-lower-confidence + honest no-pathway framing — corrects the model running 'hot' on coherence",
            PCC_FRAME_GUARDED,
        ),
    ],
}

# Component order for the build-up (matches single_call component_order).
WALKUP_ORDER = ("CAT", "GCR", "NPR", "UPR", "PCC")
# Goal metric per stage (what select_winner optimizes at that stage).
STAGE_GOAL = {
    "CAT": "category",
    "GCR": "category",
    "NPR": "novel_subclass",
    "UPR": "unchar_subclass",
    "PCC": "coherence",
}
