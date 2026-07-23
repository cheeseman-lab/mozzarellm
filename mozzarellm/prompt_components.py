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

CLUSTER_ANALYSIS_TASK = """
MISSION: Functional genomics experiments cluster genes by phenotypic similarity. Your goal is to:
1. Identify the dominant biological pathway that explains why these genes cluster together
2. Categorize ALL genes relative to this pathway (ESTABLISHED / UNCHARACTERIZED / NOVEL_ROLE)
3. Prioritize understudied genes (UNCHARACTERIZED and NOVEL_ROLE) for follow-up experiments

The pathway is not the end goal - it's the lens for discovering which genes merit investigation.
"""

CLUSTER_ANALYSIS_TASK_MULTI = """
MISSION: Functional genomics experiments cluster genes by phenotypic similarity. Your goal is to:
1. Identify 1-3 biological pathways that together explain why these genes cluster together
2. Categorize ALL genes relative to their best-fit pathway (ESTABLISHED / UNCHARACTERIZED / NOVEL_ROLE)
3. Prioritize understudied genes (UNCHARACTERIZED and NOVEL_ROLE) for follow-up experiments

A pathway requires at least 3 genes to be reported. The pathways are not the end goal — they are the lens for discovering which genes merit investigation.
"""

# =============================================================================
# GENE CATEGORIZATION & CLASSIFICATION RULES
# =============================================================================

GENE_CATEGORIZATION_RULES = """
STEP A — CATEGORIZE each gene into exactly one of three categories:

1. ESTABLISHED:
   At least one peer-reviewed paper directly demonstrates this gene's functional role
   in the identified pathway (e.g., knockout/knockdown phenotype, biochemical interaction,
   or mechanistic study within this pathway). Review articles or guilt-by-association
   do not count — there must be direct experimental evidence in this specific pathway.

2. NOVEL_ROLE:
   At least one paper has studied this gene's molecular function, but that function is
   in a DIFFERENT pathway. The gene is characterized — just not in this context.

3. UNCHARACTERIZED:
   No paper has focused on this gene's molecular function in any pathway in human cells.
   This includes completely unstudied genes, genes with only domain/homology annotations,
   and genes characterized only in non-human organisms.

BOUNDARY RULES (apply in order):
- Has any paper focused on this gene's molecular function? → No → UNCHARACTERIZED (stop)
- Does that paper show a role in THIS specific pathway? → Yes → ESTABLISHED (stop)
- Otherwise → NOVEL_ROLE

STEP B — CLASSIFY: For NOVEL_ROLE and UNCHARACTERIZED genes, assign a sub-class
(see classification rules below).
"""

NOVEL_CLASSIFICATION_RULES = """
Sub-classes for NOVEL_ROLE genes (genes with established functions in OTHER pathways):

  NO_EVIDENCE: No data linking this gene to the identified pathway.
  INDIRECT_EVIDENCE: A logical connection exists based on shared biology (e.g., same organelle, upstream regulator) but no direct experimental link.
  PARTIAL_EVIDENCE: Preliminary data (e.g., proteomics hit, co-expression) suggests a link to this pathway, but no focused mechanistic study. If a focused study exists, recategorize as ESTABLISHED.
  CONTRADICTORY_EVIDENCE: The gene's known function is incompatible with this pathway.

Assign exactly one sub-class per gene.
"""

UNCHARACTERIZED_CLASSIFICATION_RULES = """
Sub-classes for UNCHARACTERIZED genes (no focused study of molecular function in human cells):

  DARK_GENE: No name, no functional characterization whatsoever.
  NASCENT: No standard name, but some preliminary functional data exists.
  ANNOTATED_ONLY: Has a gene name and domain/motif annotations, but no mechanistic study.
  NON_HUMAN_CHARACTERIZED: Functionally studied in a non-human organism only.

Assign exactly one sub-class per gene.
"""

# =============================================================================
# PATHWAY CONFIDENCE ASSESSMENT
# =============================================================================

PATHWAY_CONFIDENCE_CRITERIA = """
ASSESSING PATHWAY CONFIDENCE:

After identifying candidate pathway(s), evaluate how well they explain the cluster using
these stringent criteria based on what percentage of genes fit the proposed pathway(s):

HIGH CONFIDENCE:
- >70% of genes in the cluster fit the proposed pathway(s)
- Multiple well-established genes with strong literature support in the pathway(s)
- Clear functional relationships between genes that explain the observed phenotypic clustering

MEDIUM CONFIDENCE:
- 50-70% of genes in the cluster fit the proposed pathway(s)
- Some established genes from the pathway(s), with additional plausible supporting genes
- Functional relationship is plausible but has some gaps or uncertainties

LOW CONFIDENCE:
- 30-50% of genes in the cluster fit the proposed pathway(s)
- Few established pathway genes; themes may be broad or general
- Significant heterogeneity in gene functions within the cluster

NO COHERENT PATHWAY:
- <30% of genes in the cluster fit any proposed pathway(s)
- Genes belong to many unrelated pathways
- Cluster contains nontargeting control genes
- Cannot identify a dominant biological process

If there is no coherent pathway, set:
- "pathway_confidence": "Low"
- "dominant_process": "No coherent biological pathway"
- And explain the reasoning clearly in the "summary" field

Remember: The goal is to honestly assess pathway support, not to force-fit genes into pathways.
Low confidence clusters may still contain valuable discovery opportunities if individual genes
are understudied.
"""

# =============================================================================
# OUTPUT FORMAT
# =============================================================================

OUTPUT_FORMAT_JSON = """
Provide a concise analysis in this exact JSON format:
{
  "cluster_id": "[CLUSTER_ID]",  // IMPORTANT: Use the exact cluster_id provided in the prompt
  "dominant_process": "pathway name (or comma-separated if multiple)",
  "pathway_confidence": "High/Medium/Low",
  "established_genes": ["GeneA", "GeneB"],
  "uncharacterized_genes": [
    {
      "gene": "GeneC",
      "class": "DARK_GENE | NASCENT | ANNOTATED_ONLY | NON_HUMAN_CHARACTERIZED",
      "rationale": "explanation of categorization and subclassification",
      "evidence": "quote(s) from annotations or citations, if available"
    }
  ],
  "novel_role_genes": [
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
# LITERATURE VALIDATION (mode-agnostic MCP step) — two selectable variants:
#   "LIT"  STEP_LITERATURE_VALIDATION    — category-gated (NOVEL_ROLE/UNCHARACTERIZED genes)
#   "LITB" STEP_LITERATURE_GAPFILL_BLANK — evidence-gated (blank-annotation genes only)
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
- A top-level `literature_informed_pathway_revision` object: {{"pre_literature_pathway": "your tentative pathway BEFORE literature validation", "post_literature_pathway": "your final pathway AFTER literature validation (may be the same)", "pathway_changed": true/false, "rationale": "one sentence — what literature drove the change, or why it stayed the same"}}."""

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

CRITICAL OUTPUT CONSTRAINT: Your entire response MUST be a single valid JSON object and nothing else. Start with `{` and end with `}`. Do NOT write any preamble, plan, or commentary about your searches — no "Based on my analysis...", no "According to PubMed...", no restating of the query. Do NOT write any text before the opening brace or after the closing brace. Report every literature finding ONLY inside JSON fields (rationale, mcp_gapfill), never as prose."""

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

You may also use the per-gene `up_features` / `down_features` lists in `cluster_genes`
ONLY to verify that candidate "essential" features are driven by an OVERLAPPING gene
subset (not disjoint subsets that just sum to a high fraction).

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

# =============================================================================
# CHAIN-OF-THOUGHT STEPS
# =============================================================================

COT_SCREEN_CONTEXT = "Review the provided screen context:"

COT_STEP_PATHWAY_HYPOTHESIS = """PATHWAY HYPOTHESIS (2-3 candidates):
- Review gene annotations
- List 2-3 candidate pathways with supporting genes
- Note which annotations support each hypothesis"""

COT_STEP_GENE_CATEGORIZATION = f"""GENE CATEGORIZATION (cite evidence):
For each gene, assign to exactly one category: ESTABLISHED / NOVEL_ROLE / UNCHARACTERIZED
These are defined according to the following rules: {GENE_CATEGORIZATION_RULES}
"""

COT_STEP_SUBCLASSIFICATION = f"""SUB-CLASSIFICATION:
For NOVEL_ROLE genes, assign one sub-class: NO_EVIDENCE / INDIRECT_EVIDENCE / PARTIAL_EVIDENCE / CONTRADICTORY_EVIDENCE
These are defined according to the following rules: {NOVEL_CLASSIFICATION_RULES}
For UNCHARACTERIZED genes, assign one sub-class: DARK_GENE / NASCENT / ANNOTATED_ONLY / NON_HUMAN_CHARACTERIZED
These are defined according to the following rules: {UNCHARACTERIZED_CLASSIFICATION_RULES}
Cite specific annotations that inform each classification."""

COT_STEP_PATHWAY_SELECTION = f"""PATHWAY SELECTION:
Once you have identified candidate pathway(s), evaluate how well EACH pathway explains the cluster using
these stringent criteria based on what percentage of genes fit the proposed pathway: {PATHWAY_CONFIDENCE_CRITERIA}
Now, select a dominant pathway based on:
  * Number of established genes with direct roles
  * Coherence of functional relationships
  * Quality of supporting evidence"""

COT_STEP_VERIFICATION = """VERIFICATION:
- Check for contradictions
- Verify all genes are classified (no omissions)
- Adjust confidence if evidence is weak or contradictory
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
    "LIT": STEP_LITERATURE_VALIDATION,
    "LITB": STEP_LITERATURE_GAPFILL_BLANK,
    "cPH": COT_STEP_PATHWAY_HYPOTHESIS,
    "cGCR": COT_STEP_GENE_CATEGORIZATION,
    "cPri": COT_STEP_SUBCLASSIFICATION,
    "cPSC": COT_STEP_PATHWAY_SELECTION,
    "cVer": COT_STEP_VERIFICATION,
    "cFC": STEP_FEATURE_COHERENCE,
    "cPC": STEP_PATHWAY_CONSISTENCY,
    "cO": COT_STEP_OUTPUT,
}

CANONICAL_ZERO_SHOT_ORDER = ["CAT", "SC", "GCR", "NPR", "UPR", "PCC", "O"]
CANONICAL_ZERO_SHOT_MCP_ORDER = ["CAT", "SC", "GCR", "NPR", "UPR", "PCC", "LIT", "O"]
CANONICAL_COT_ORDER = ["CAT", "SC", "cPH", "cGCR", "cPri", "cPSC", "cVer", "cO"]
CANONICAL_COT_MCP_ORDER = ["CAT", "SC", "cPH", "cGCR", "cPri", "LIT", "cPSC", "cVer", "cO"]
CANONICAL_FEATURE_INTERP_COT_ORDER = [
    "CAT",
    "SC",
    "cPH",
    "cGCR",
    "cPri",
    "cPSC",
    "cVer",
    "cFC",
    "cPC",
    "cO",
]
CANONICAL_FEATURE_INTERP_COT_MCP_ORDER = [
    "CAT",
    "SC",
    "cPH",
    "cGCR",
    "cPri",
    "LIT",
    "cPSC",
    "cVer",
    "cFC",
    "cPC",
    "cO",
]
