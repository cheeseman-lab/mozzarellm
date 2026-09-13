"""
Prompt templates and instructions for gene cluster analysis.

Organized in assembly order: components appear in the file in the same order
they are concatenated by the prompt factory.

Standard mode: TASK → SCREEN_CONTEXT → GENE_CATEGORIZATION → NOVEL_RULES →
               UNCHARACTERIZED_RULES → PATHWAY_CONFIDENCE → OUTPUT_FORMAT

CoT mode:      TASK → SCREEN_CONTEXT → PATHWAY_HYPOTHESIS → GENE_CATEGORIZATION →
               SUBCLASSIFICATION → PATHWAY_SELECTION → VERIFICATION → OUTPUT_FORMAT
"""

# =============================================================================
# CORE TASK
# =============================================================================

CLUSTER_ANALYSIS_TASK = """MISSION: Every downstream call depends on correctly identifying what unites this cluster, so anchor
there first.
1. Determine the biological pathway(s) that explain the cluster — commit to a single dominant
   process where one clearly fits, or 2-3 distinct processes if the cluster genuinely spans them. If
   no process explains a substantial share of the genes, declare no coherent pathway.
2. Categorize ALL genes against that pathway (ESTABLISHED / NOVEL_ROLE / UNCHARACTERIZED).
3. Prioritize UNCHARACTERIZED and NOVEL_ROLE genes for follow-up.
"""

# =============================================================================
# GENE CATEGORIZATION & CLASSIFICATION RULES
# =============================================================================

GENE_CATEGORIZATION_RULES = """PRECONDITION: This step applies ONLY when a coherent biological pathway has been identified for the cluster. If no coherent pathway exists, leave `established_genes`, `novel_role_genes`, and `uncharacterized_genes` empty and skip this step — per-gene classification relative to a nonexistent pathway is undefined.

TASK: For each gene in the cluster, categorize as ESTABLISHED, NOVEL_ROLE, or UNCHARACTERIZED based on the evidence provided in the gene's bundle annotation.
- ESTABLISHED: the annotation documents a role in THIS cluster's pathway.
- NOVEL_ROLE: the annotation documents function, but not in this pathway — its membership here is the new evidence.
- UNCHARACTERIZED: the annotation offers nothing to relate to any process.

Gating procedure (apply in order):
1. Does the annotation offer ANY functional signal — a described function, a domain or motif, a process, an interaction? If NO (nothing to relate): UNCHARACTERIZED. A sparse annotation whose evidence still threads to some process is NOT uncharacterized — sparse but relatable means step 2 decides.
2. Is the documented function part of THIS cluster's pathway? Yes: ESTABLISHED. No: NOVEL_ROLE. Judge against the pathway itself, not the gene's prominence — a well-studied gene with no documented role in this pathway is NOVEL_ROLE; general fame never makes a gene ESTABLISHED.
"""

NOVEL_CLASSIFICATION_RULES = """PRECONDITION: This step applies ONLY to genes already categorized as NOVEL_ROLE per the gene classification rules. If no genes were categorized as NOVEL_ROLE (because no coherent pathway exists, or no genes fit the criteria), do nothing — no sub-classification is needed.

SUB-CLASSIFICATION (NOVEL_ROLE genes only): assign exactly one sub-class based on how the bundle's annotation relates to the identified pathway.
- NO_EVIDENCE: nothing in the annotation links the gene to this pathway.
- INDIRECT_EVIDENCE: the annotation shows a logical connection (shared organelle, upstream regulator) but no direct experimental link.
- PARTIAL_EVIDENCE: the annotation reports actual data touching this pathway — a physical interaction, proteomics/co-IP hit, co-expression, or a functional assay — without focused mechanistic study.
- CONTRADICTORY_EVIDENCE: the annotation describes a function incompatible with this pathway.

PARTIAL_EVIDENCE always requires such reported data: a connection that is merely plausible, however strong the logic, is INDIRECT_EVIDENCE at most — never PARTIAL_EVIDENCE without data in the annotation.
"""

UNCHARACTERIZED_CLASSIFICATION_RULES = """PRECONDITION: This step applies ONLY to genes already categorized as UNCHARACTERIZED per the gene
classification rules. If no genes were categorized as UNCHARACTERIZED, do nothing.

SUB-CLASSIFICATION (UNCHARACTERIZED genes only): assign exactly one sub-class based on the gene's
bundle annotation.
- DARK_GENE: identified only by a raw identifier (e.g. an Ensembl ENSG or LOC-style ID) with no
  annotation of any kind.
- NASCENT: no standard name, but some preliminary functional data exists.
- ANNOTATED_ONLY: carries an official gene symbol and/or domain/motif annotation, but no
  functional study.
- NON_HUMAN_CHARACTERIZED: functionally studied in a non-human organism only.

Identifier test: a gene carrying an official gene symbol is NEVER DARK_GENE, even when its bundle
contains no annotation text — an assigned symbol means the gene has been named and catalogued, and
ANNOTATED_ONLY is the correct call. Reserve DARK_GENE for genes whose only identity is a raw
database identifier.
"""

# =============================================================================
# PATHWAY CONFIDENCE ASSESSMENT
# =============================================================================

PATHWAY_CONFIDENCE_CRITERIA = """PATHWAY CONFIDENCE: report how confident you are in your dominant_process call itself — an assessment of your own call, not merely of how well the genes fit it. Weigh three things:
- how much of the cluster the pathway explains;
- how many genes are outsiders the call cannot place — count them explicitly, INCLUDING every UNCHARACTERIZED gene whose annotation offers nothing to relate to the pathway;
- whether a different biological process could explain the cluster comparably well. Before assigning a level, briefly consider the strongest alternative explanation.

High confidence:
- The call would survive being wrong about any single gene; outsiders are absent or a token few; no credible alternative process.

Medium confidence:
- The call is the best available explanation, but it rests partly on inference (many members are not documented participants), a notable share of the cluster is outsiders it cannot place, or a plausible alternative process exists. A notable outsider share caps confidence at Medium even when every placed gene fits perfectly.

Low confidence:
- The call is tentative: an alternative explains the cluster about as well, or the cluster is heterogeneous enough that the dominant process may be an artifact of a subset.

No coherent pathway (use Low confidence, set "dominant_process": "No coherent biological pathway", and leave `established_genes`, `novel_role_genes`, and `uncharacterized_genes` empty — per-gene classification relative to a nonexistent pathway is undefined):
- No process explains a substantial share of the genes, or the cluster contains many unrelated functions or nontargeting controls.
"""

# =============================================================================
# OUTPUT FORMAT
# =============================================================================

OUTPUT_FORMAT_JSON = """
Provide a concise analysis in this exact JSON format. Include every gene, and place each gene in exactly one category between
ESTABLISHED, NOVEL_ROLE, and UNCHARACTERIZED — never list the same gene in two categories. Only fill the additional fields for
genes that are in the NOVEL_ROLE or UNCHARACTERIZED categories.

{
  "cluster_id": "[CLUSTER_ID]",  // IMPORTANT: Use the exact cluster_id provided in the prompt
  "dominant_process": "pathway name (semicolon-separated if 2-3)",
  "pathway_confidence": "High/Medium/Low",
  "established_genes": ["GeneA", "GeneB"],  // known role in this cluster's dominant process
  "uncharacterized_genes": [  // little or no functional annotation
    {
      "gene": "GeneC",
      "class": "DARK_GENE | NASCENT | ANNOTATED_ONLY | NON_HUMAN_CHARACTERIZED",
      "rationale": "explanation of categorization and subclassification",
      "evidence": "quote(s) from annotations or citations, if available"
    }
  ],
  "novel_role_genes": [  // documented function elsewhere, no known role in this process
    {
      "gene": "GeneD",
      "class": "NO_EVIDENCE | INDIRECT_EVIDENCE | PARTIAL_EVIDENCE | CONTRADICTORY_EVIDENCE",
      "rationale": "explanation of categorization and subclassification",
      "evidence": "quote(s) from annotations or citations, if available"
    }
  ],
  "summary": "key findings summary"
}
"""

# =============================================================================
# LITERATURE (mode-agnostic MCP step) — two selectable variants, one slot ("LIT"):
#   "LIT"  STEP_LITERATURE_GAPFILL_BLANK — evidence-gated (blank-annotation genes only);
#          the benchmark-selected default
#   "LITV" STEP_LITERATURE_VALIDATION    — category-gated (NOVEL_ROLE/UNCHARACTERIZED genes)
# Both used in single_mcp / cot_mcp / stepwise_mcp — exactly 2 MCP tool calls.
# =============================================================================

LITERATURE_VALIDATION_OUTPUT_FORMAT = """
The "literature_validation" field per gene must contain:
- "literature_support": "none" | "weak" | "moderate" | "strong"
- "relevant_papers": up to 3 entries, each {"pmid": "...", "title": "...", "year": "...", "key_finding": "..."}
- "pathway_connection": one sentence — how this gene is implicated in the pathway based on literature (null if none found)
- "suggested_reclassification": null | "ESTABLISHED" | "NOVEL_ROLE" | "UNCHARACTERIZED"
- "suggested_subclass": null | one of the valid subclass values for the gene's (possibly reclassified) category:
    NOVEL_ROLE: NO_EVIDENCE | INDIRECT_EVIDENCE | PARTIAL_EVIDENCE | CONTRADICTORY_EVIDENCE
    UNCHARACTERIZED: DARK_GENE | NASCENT | ANNOTATED_ONLY | NON_HUMAN_CHARACTERIZED
    ESTABLISHED: null (no subclasses)
- "rationale": one sentence — why reclassification/subclass update is or isn't warranted
"""

STEP_LITERATURE_VALIDATION = f"""LITERATURE VALIDATION (constrained MCP):
Validate NOVEL_ROLE and UNCHARACTERIZED genes against PubMed using the attached PubMed MCP tools.

Procedure (follow EXACTLY):
1. Extract a 2-3 word PubMed keyword from the dominant pathway you identify. Strip subprocess descriptors, complex names, parenthetical qualifiers, and em-dash extensions — keep only the core process name.
2. ONE `search_articles` call with: `(GENE1[tiab] OR GENE2[tiab] OR ... OR GENEN[tiab]) AND <keyword>`, max_results=30. The [tiab] tag on EVERY gene symbol is mandatory.
3. ONE `get_article_metadata` call with all returned PMIDs.
4. For each paper, judge relevance against your FULL pathway annotation (not just the keyword). A paper about "ribosome biogenesis in mitochondria" is peripheral to a "40S SSU processome" cluster.

Hard constraints:
- EXACTLY 2 tool calls total (1 search + 1 metadata). Do not call any tool more than once.
- Do NOT search per-gene. Do NOT call any other tools.
- Use the tools to validate gene categorizations against the literature; do NOT use them to brainstorm pathways.

Update categorizations where warranted (e.g., genes with direct pathway evidence → ESTABLISHED). The updated categorizations should be reflected in your final pathway selection and confidence assessment.

Also note whether the literature changes your pathway hypothesis itself — e.g., literature reveals a more specific subprocess, a different dominant pathway, or merges/splits your candidates. Record this as a pathway revision.

In the final output, include:
- A `literature_validation` field on each NOVEL_ROLE and UNCHARACTERIZED gene in the final classification, per the schema:
{LITERATURE_VALIDATION_OUTPUT_FORMAT}
- A top-level `literature_informed_reclassifications` array listing every gene whose category changed from your pre-literature categorization to post-validation. Each entry: {{"gene": "...", "initial_category": "ESTABLISHED|NOVEL_ROLE|UNCHARACTERIZED", "final_category": "ESTABLISHED|NOVEL_ROLE|UNCHARACTERIZED", "driving_pmids": ["..."], "rationale": "one sentence — what literature justified the move"}}. If nothing changed, use an empty array.
- A top-level `literature_informed_pathway_revision` object: {{"pre_literature_pathway": "your tentative pathway BEFORE literature validation", "post_literature_pathway": "your final pathway AFTER literature validation (may be the same)", "pathway_changed": true/false, "rationale": "one sentence — what literature drove the change, or why it stayed the same"}}.

CRITICAL OUTPUT CONSTRAINT: Your entire response MUST be a single valid JSON object and nothing else. Start with `{{` and end with `}}`. Do NOT write any preamble, plan, or commentary about your searches — no "Based on my analysis...", no "According to PubMed...", no restating of the query. Do NOT write any text before the opening brace or after the closing brace. Report every literature finding ONLY inside JSON fields (rationale, literature_validation), never as prose.
"""

STEP_LITERATURE_GAPFILL_BLANK = """LITERATURE GAP-FILL (evidence-gated MCP):
Some genes in the evidence bundle have NO functional annotation provided (the annotation field is empty, or absent entirely). For those genes ONLY, use the attached PubMed MCP tools to retrieve functional evidence. Genes that already have annotation text MUST NOT be looked up, regardless of how you classify them.

BEFORE ANY TOOL USE — count the GAP set: genes whose functional-annotation field is empty or absent. This count fixes your ENTIRE tool budget:
- GAP set EMPTY (zero blank genes) → make ZERO tool calls. Do not search anything at all. Go straight to classification. Most clusters land here.
- GAP set NON-EMPTY → make EXACTLY TWO tool calls, no more: (1) ONE `search_articles` with all gap genes OR'd together `(GAP1[tiab] OR GAP2[tiab] OR ...)`, max_results=30, [tiab] on every symbol; (2) ONE `get_article_metadata` on the returned PMIDs. Then STOP calling tools permanently.

ABSOLUTE tool rules (violating any of these breaks the run):
- The 2-call cap is HARD. Never exceed it under any circumstance.
- Issue the search EXACTLY ONCE. NEVER repeat, re-word, refine, or re-run a search — not even if it returns few results, zero results, or nothing useful. If the search returns nothing for a gene, record "no literature found" for that gene and move on. Re-searching for any reason is FORBIDDEN.
- NEVER search a gene that already has annotation text — only the blank/GAP genes.
- Do NOT search per-gene, and do NOT use the tools to explore or brainstorm pathways.

For each GAP gene, extract a one-line functional summary from the retrieved literature, or record "no literature found".

Classify GAP genes on equal footing with the pre-annotated genes using the retrieved evidence: a GAP gene with direct pathway literature → ESTABLISHED/NOVEL_ROLE as warranted; a GAP gene with no retrievable literature → UNCHARACTERIZED (DARK_GENE).

In the final output, add a top-level `mcp_gapfill` array — one entry per GAP gene: {"gene": "...", "evidence_found": true|false, "driving_pmids": ["..."], "retrieved_summary": "..."}. Empty array if there were no GAP genes.

CRITICAL OUTPUT CONSTRAINT: Your entire response MUST be a single valid JSON object and nothing else. Start with `{` and end with `}`. Do NOT write any preamble, plan, or commentary about your searches — no "Based on my analysis...", no "According to PubMed...", no restating of the query. Do NOT write any text before the opening brace or after the closing brace. Report every literature finding ONLY inside JSON fields (rationale, mcp_gapfill), never as prose.
"""

# =============================================================================
# FEATURE COHERENCE + PATHWAY CONSISTENCY (feature-interp mode)
# =============================================================================

FEATURE_COHERENCE_OUTPUT_FORMAT = """
The top-level "feature_coherence" field must contain:
- "concrete": true | false — true only if essential up/down forms a coherent signature driven by overlapping gene subsets
- "essential_up": array of {"feature": "...", "frac_up": float, "supporting_genes": ["..."]} (empty when not concrete)
- "essential_down": array of {"feature": "...", "frac_down": float, "supporting_genes": ["..."]} (empty when not concrete)
- "mixed_or_unsupported": array of {"feature": "...", "frac_up": float, "frac_down": float}
- "rationale": one or two sentences citing fractions and gene-subset overlap; no biology
"""

STEP_FEATURE_COHERENCE = f"""FEATURE COHERENCE (recall — discrete table, no biology):

Each evidence bundle includes a `feature_coherence` field with a per-feature breakdown
across the cluster: for each feature, `n_up` / `frac_up` and `n_down` / `frac_down` of
the cluster genes calling it differentially significant in that direction, along with
the corresponding `up_genes` / `down_genes` lists. This is the data for this step.
The table is bounded: only features reaching `min_frac` of the cluster in at least one
direction are listed (`n_features_shown` of `n_features_total`); features below that
coverage cannot be essential and are absent by design, not by lack of signal.
If the experimental context describes how these features were derived (what they
measure, the ranking, the cutoff), read them in that light.

Use the `up_genes` / `down_genes` lists to verify that candidate "essential" features
are driven by an OVERLAPPING gene subset (not disjoint subsets that just sum to a
high fraction).

Procedure:

1. From `feature_coherence.features`, identify "essential" features:
   - Strong UP: high `frac_up` AND `frac_down` near zero. The supporting gene set must
     be cohesive — features that aggregate to a high fraction but are driven by largely
     non-overlapping gene subsets are NOT essential.
   - Strong DOWN: high `frac_down` AND `frac_up` near zero. Same gene-overlap criterion.
2. List "mixed_or_unsupported" features — those with both directions modest, or with
   conflicting directional signal. Do not include features with no signal at all.
3. Set `concrete`:
   - true when essential_up + essential_down forms a coherent feature signature: multiple
     features with strong directional agreement, driven by overlapping gene subsets.
   - false when no features have strong agreement, OR the candidates with agreement are
     driven by disjoint gene subsets.
4. Write `rationale` (one or two sentences). Cite features by name and gene-fractions.
   When `concrete` is false, briefly state which lens failed (no agreement, or disjoint
   gene subsets, or both). NO biology, NO mechanisms, NO pathway concepts in this step.

Hard guardrails:
- This step is recall over a discrete table. Do not introduce mechanisms or biology.
- Do not invent feature names; only cite features present in `feature_coherence.features`.
- Off-ramp: if no features pass the criteria, set `concrete: false`, leave
  `essential_up` and `essential_down` empty, and explain in the rationale.
- Do not compute or state new biological themes here. The next step does the interpretation.

In the final output, include:
- A top-level `feature_coherence` object, per the schema:
{FEATURE_COHERENCE_OUTPUT_FORMAT}"""

PATHWAY_CONSISTENCY_OUTPUT_FORMAT = """
The top-level "pathway_consistency" field must contain:
- "verdict": "consistent" | "partial" | "inconsistent" | "no_signal" (required "no_signal" when feature_coherence.concrete is false)
- "rationale": one or two sentences anchored to dominant_process; cite essential features by name; no new biology
- "confidence_revision": null | one sentence (only set when essential signature materially changes confidence in dominant_process)
"""

STEP_PATHWAY_CONSISTENCY = f"""PATHWAY CONSISTENCY (bounded interpretation, anchored to the call):

Using the essential feature signature you produced in FEATURE COHERENCE and the
`dominant_process` you have already called, judge consistency.

Procedure:

1. Set `verdict`:
   - "consistent": the essential up/down features track with what `dominant_process`
     would imply.
   - "partial": some essential features are consistent, others are not.
   - "inconsistent": the essential signature contradicts `dominant_process`.
   - "no_signal": REQUIRED when `feature_coherence.concrete` was false.

2. Write `rationale` (ONE OR TWO sentences). Cite essential features by name and tie
   them to `dominant_process`. Do not introduce biological mechanisms, pathway-adjacent
   processes, or any concepts beyond what the literal pathway name in `dominant_process`
   implies. The rationale's job is to CONNECT the recalled feature signature to the
   pathway call, NOT to explain new biology.

3. Optional `confidence_revision`: only populate when the essential feature signature
   materially changes confidence in `dominant_process`. The justification must reference
   essential features by name, not individual gene claims. Otherwise leave it null.

Hard guardrails:
- DO NOT modify `dominant_process` based on the feature signature. If features
  contradict it, that is a confidence concern, not a re-call of the pathway.
- DO NOT introduce new biological concepts, mechanisms, or pathways. The downstream
  human-driven MCP exploration handles synthesis; this step is a bounded cross-check.
- If `feature_coherence.concrete` is false, `verdict` must be "no_signal" and
  `confidence_revision` must be null. No exceptions.

In the final output, include:
- A top-level `pathway_consistency` object, per the schema:
{PATHWAY_CONSISTENCY_OUTPUT_FORMAT}"""

PHENOTYPE_STRENGTH_OUTPUT_FORMAT = """
The top-level "phenotype_strength" field must contain:
- "verdict": "strong" | "mixed" | "weak"
- "weak_members": [gene symbols from the weakest quartile of the screen; empty if none]
- "rationale": one or two sentences citing the table's ranks and fractions; no new biology
- "confidence_revision": null | one sentence (only set when the strength profile materially changes confidence in dominant_process)
"""

STEP_PHENOTYPE_STRENGTH = f"""PHENOTYPE STRENGTH (recall over a discrete table, then a bounded verdict):

Each evidence bundle includes a `phenotype_strength` table: per-gene perturbation-phenotype
ranks of the form "N/M" — rank among the M genes of this screen by strength relative to
non-targeting controls, 1 = strongest — plus `median_rank`, `strongest_quartile_frac`,
`weakest_quartile_frac`, and the `ranked_genes` list (strongest first). This is the data for
this step. If the experimental context describes how strength was measured, read the ranks
in that light.

Procedure:
1. Verdict on the cluster's phenotypic signal, from the table:
   - "strong": ranks concentrate toward the strong end — the clustering rests on robust
     phenotypes.
   - "mixed": a strong core plus weak members; list the weak members.
   - "weak": ranks concentrate toward the weak end — the clustering may be noise-dominated.
2. Write `rationale` (one or two sentences citing `median_rank` and the quartile fractions).
3. Set `confidence_revision` only when the strength profile materially changes confidence in
   `dominant_process` — a coherent call resting on weak phenotypes deserves tempered
   confidence, stated in one sentence. Otherwise leave it null.

Hard guardrails:
- Strength tempers confidence; it never re-calls the pathway or re-categorizes a gene. A
  coherent cluster of weak phenotypes is "right call, weak signal". Do not modify
  `dominant_process` or gene categories in this step.
- Cite only ranks present in the table; genes absent from `ranked_genes` carry no rank and
  are not treated as weak.

In the final output, include:
- A top-level `phenotype_strength` object, per the schema:
{PHENOTYPE_STRENGTH_OUTPUT_FORMAT}"""

# =============================================================================
# CHAIN-OF-THOUGHT STEPS
# =============================================================================


COT_STEP_PATHWAY_HYPOTHESIS = """PATHWAY HYPOTHESIS:
- Review gene annotations
- Identify the candidate pathway(s) the annotations support — commit to a single dominant process where one clearly fits, or 2-3 distinct processes if the cluster genuinely spans them
- Note which annotations support each candidate
- If no process explains a substantial share of the genes, say so — an honest "no coherent pathway" call is a valid outcome"""

def build_cot_step_gene_categorization(gcr: str = GENE_CATEGORIZATION_RULES) -> str:
    """Compose the cot GENE CATEGORIZATION step from the (possibly overridden) GCR text."""
    return f"""GENE CATEGORIZATION (cite evidence):
For each gene, assign to exactly one category: ESTABLISHED / NOVEL_ROLE / UNCHARACTERIZED
These are defined according to the following rules: {gcr}
"""


def build_cot_step_subclassification(
    npr: str = NOVEL_CLASSIFICATION_RULES, upr: str = UNCHARACTERIZED_CLASSIFICATION_RULES
) -> str:
    """Compose the cot SUB-CLASSIFICATION step from the (possibly overridden) NPR/UPR texts."""
    return f"""SUB-CLASSIFICATION:
For NOVEL_ROLE genes, assign one sub-class: NO_EVIDENCE / INDIRECT_EVIDENCE / PARTIAL_EVIDENCE / CONTRADICTORY_EVIDENCE
These are defined according to the following rules: {npr}
For UNCHARACTERIZED genes, assign one sub-class: DARK_GENE / NASCENT / ANNOTATED_ONLY / NON_HUMAN_CHARACTERIZED
These are defined according to the following rules: {upr}
Cite specific annotations that inform each classification."""


def build_cot_step_pathway_selection(pcc: str = PATHWAY_CONFIDENCE_CRITERIA) -> str:
    """Compose the cot PATHWAY SELECTION step from the (possibly overridden) PCC text."""
    return f"""PATHWAY SELECTION:
Once you have identified candidate pathway(s), evaluate how well EACH pathway explains the cluster using
these stringent criteria based on what percentage of genes fit the proposed pathway: {pcc}
Now, select a dominant pathway based on:
  * Number of established genes with direct roles
  * Coherence of functional relationships
  * Quality of supporting evidence"""


COT_STEP_GENE_CATEGORIZATION = build_cot_step_gene_categorization()

COT_STEP_SUBCLASSIFICATION = build_cot_step_subclassification()

COT_STEP_PATHWAY_SELECTION = build_cot_step_pathway_selection()

COT_STEP_VERIFICATION = """VERIFICATION:
- Check for contradictions
- Verify all genes are classified (no omissions)
- Check that the confidence level follows the stated confidence criteria, not general impressions
- Note any gaps in evidence that limit conclusions"""

COT_STEP_OUTPUT = f"""FINAL JSON OUTPUT:
- Compile structured JSON with all required fields
- Ensure cluster_id matches input exactly
- Include concise summary highlighting key findings and evidence quality
According to {OUTPUT_FORMAT_JSON}"""

# =============================================================================
# COMPONENT REGISTRY & CANONICAL ORDERS
# =============================================================================
# Shorthand keys for each prompt component, used by prompt_factory when
# assembling prompts in an arbitrary order (e.g. for benchmarking).
#
# Baseline components:
#   CAT  = Cluster Analysis Task  (always present)
#   SC   = Screen Context         (always present, injected per-case — NOT in registry)
#   GCR  = Gene Categorization Rules
#   NPR  = Novel Classification Rules
#   UPR  = Uncharacterized Classification Rules
#   PCC  = Pathway Confidence Criteria
#   O    = Output format (JSON)
#
# CoT-specific components:
#   cPH  = Pathway Hypothesis step
#   cPSC = Pathway Selection & Confidence step (references PCC)
#   cGCR = Gene Categorization step            (references GCR)
#   cPri = Sub-classification (references NPR & UPR)
#   cVer = Verification step
#   cO   = Final JSON Output step              (references O)
#   cFC  = Feature Coherence step  (feature-interp mode; emits feature_coherence)
#   cPC  = Pathway Consistency step (feature-interp mode; emits pathway_consistency)
#
# NOTE: "SC" is not in the registry because screen context is dynamic
# (varies per case). It is handled specially during assembly.

COMPONENT_REGISTRY = {
    "CAT": CLUSTER_ANALYSIS_TASK,
    "GCR": GENE_CATEGORIZATION_RULES,
    "NPR": NOVEL_CLASSIFICATION_RULES,
    "UPR": UNCHARACTERIZED_CLASSIFICATION_RULES,
    "PCC": PATHWAY_CONFIDENCE_CRITERIA,
    "O": OUTPUT_FORMAT_JSON,
    "LIT": STEP_LITERATURE_GAPFILL_BLANK,
    "LITV": STEP_LITERATURE_VALIDATION,
    "cPH": COT_STEP_PATHWAY_HYPOTHESIS,
    "cGCR": COT_STEP_GENE_CATEGORIZATION,
    "cPri": COT_STEP_SUBCLASSIFICATION,
    "cPSC": COT_STEP_PATHWAY_SELECTION,
    "cVer": COT_STEP_VERIFICATION,
    "cFC": STEP_FEATURE_COHERENCE,
    "cPC": STEP_PATHWAY_CONSISTENCY,
    "cPS": STEP_PHENOTYPE_STRENGTH,
    "cO": COT_STEP_OUTPUT,
}


def derive_cot_overrides(component_overrides: dict[str, str]) -> dict[str, str]:
    """Propagate base-component overrides into the cot slots composed from them.

    The cot steps cGCR/cPri/cPSC embed the GCR/NPR+UPR/PCC texts at composition
    time, so an override of a base component would otherwise never reach the cot
    and stepwise routes. Rebuilds each affected cot slot from the overridden base
    texts using the same composition templates; an explicit override for a cot
    slot always wins over a derived one.
    """
    derived: dict[str, str] = {}
    if "GCR" in component_overrides:
        derived["cGCR"] = build_cot_step_gene_categorization(component_overrides["GCR"])
    if "NPR" in component_overrides or "UPR" in component_overrides:
        derived["cPri"] = build_cot_step_subclassification(
            component_overrides.get("NPR", NOVEL_CLASSIFICATION_RULES),
            component_overrides.get("UPR", UNCHARACTERIZED_CLASSIFICATION_RULES),
        )
    if "PCC" in component_overrides:
        derived["cPSC"] = build_cot_step_pathway_selection(component_overrides["PCC"])
    return {**derived, **component_overrides}

CANONICAL_ZERO_SHOT_ORDER = ["CAT", "SC", "GCR", "NPR", "UPR", "PCC", "O"]
CANONICAL_ZERO_SHOT_MCP_ORDER = ["CAT", "SC", "GCR", "NPR", "UPR", "PCC", "LIT", "O"]
CANONICAL_COT_ORDER = ["CAT", "SC", "cPH", "cGCR", "cPri", "cPSC", "cVer", "cO"]
CANONICAL_COT_MCP_ORDER = ["CAT", "SC", "cPH", "cGCR", "cPri", "LIT", "cPSC", "cVer", "cO"]
def build_cot_component_order(
    mcp: bool = False, features: bool = False, strength: bool = False
) -> list[str]:
    """The cot component order with phenotype steps included iff their data is.

    Feature-interpretation steps (cFC, cPC) and the phenotype-strength step
    (cPS) enter the chain only when the corresponding bundle fields exist —
    a prompt never references data the bundles don't carry. LIT keeps its
    canonical position (after cPri); phenotype steps sit after cVer, before cO.
    """
    order = list(CANONICAL_COT_MCP_ORDER if mcp else CANONICAL_COT_ORDER)
    tail = order.pop()  # cO stays last
    if features:
        order += ["cFC", "cPC"]
    if strength:
        order += ["cPS"]
    return order + [tail]


CANONICAL_FEATURE_INTERP_COT_ORDER = build_cot_component_order(features=True)
