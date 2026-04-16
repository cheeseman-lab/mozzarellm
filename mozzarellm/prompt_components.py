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
(see classification rules below) and a priority score (1-10).
"""

NOVEL_CLASSIFICATION_RULES = """
Sub-classes for NOVEL_ROLE genes (genes with established functions in OTHER pathways):

  NO_EVIDENCE: No data linking this gene to the identified pathway.
  INDIRECT_EVIDENCE: A logical connection exists based on shared biology (e.g., same organelle, upstream regulator) but no direct experimental link.
  PARTIAL_EVIDENCE: Preliminary data (e.g., proteomics hit, co-expression) suggests a link to this pathway, but no focused mechanistic study. If a focused study exists, recategorize as ESTABLISHED.
  CONTRADICTORY_EVIDENCE: The gene's known function is incompatible with this pathway.

Assign exactly one sub-class per gene. Then assign a priority score (1-10) for follow-up,
considering sub-class, evidence quality, pathway relevance, and experimental tractability.
"""

UNCHARACTERIZED_CLASSIFICATION_RULES = """
Sub-classes for UNCHARACTERIZED genes (no focused study of molecular function in human cells):

  DARK_GENE: No name, no functional characterization whatsoever.
  NASCENT: No standard name, but some preliminary functional data exists.
  ANNOTATED_ONLY: Has a gene name and domain/motif annotations, but no mechanistic study.
  NON_HUMAN_CHARACTERIZED: Functionally studied in a non-human organism only.

Assign exactly one sub-class per gene. Then assign a priority score (1-10) for follow-up,
considering sub-class, evidence quality, pathway relevance, and experimental tractability.
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
# LITERATURE VALIDATION PROMPT
# Single call using MCP_VALIDATION_PROMPT, constrained to 2 MCP tool calls.
# =============================================================================

# MATTEO EDIT: review — confirm "suggested_subclass" covers the right subclass values for both
# NOVEL_ROLE (NO_EVIDENCE/INDIRECT_EVIDENCE/PARTIAL_EVIDENCE/CONTRADICTORY_EVIDENCE) and
# UNCHARACTERIZED (DARK_GENE/NASCENT/ANNOTATED_ONLY/NON_HUMAN_CHARACTERIZED)
LITERATURE_VALIDATION_OUTPUT_FORMAT = """
The "literature_validation" field per gene should contain:
- "literature_support": "none" | "weak" | "moderate" | "strong"
- "relevant_papers": up to 3 entries, each {"title": "...", "year": "...", "source": "pubmed|biorxiv", "key_finding": "..."}
- "pathway_connection": one sentence — how this gene is implicated in the pathway based on literature (null if none found)
- "suggested_reclassification": null | "ESTABLISHED" | "NOVEL_ROLE" | "UNCHARACTERIZED"
- "suggested_subclass": null | updated subclass if literature changes the evidence picture
- "rationale": one sentence — why reclassification/subclass update is or isn't warranted
"""

# Placeholders: {flagged_genes_json}, {pathway}, {literature_validation_output_format}
MCP_VALIDATION_PROMPT = """
I need to validate gene-pathway associations from a functional genomics screen and produce an amended classification.

## Pathway context
Full cluster annotation: "{pathway}"

## Genes to validate
The following genes were classified as NOVEL_ROLE or UNCHARACTERIZED relative to this pathway:

{flagged_genes_json}

## Task — follow these steps EXACTLY

### Step 1: Extract a search keyword
From the full cluster annotation above, derive the SIMPLEST 2-3 word PubMed keyword that captures the broad pathway. Examples:
- "Ribosome biogenesis — 40S small subunit (SSU) processome and pre-rRNA processing; translation initiation (eIF3 complex)" → `ribosome biogenesis`
- "DNA damage response and double-strand break repair via homologous recombination" → `DNA damage repair`
- "mTOR signaling and lysosomal biogenesis" → `mTOR signaling`

Keep it short and broad — this is the recall step.

### Step 2: Make ONE search call
Call `search_articles` exactly ONCE with this query structure:
  `(GENE1[tiab] OR GENE2[tiab] OR ... OR GENEN[tiab]) AND <keyword from Step 1>`

Use [tiab] field tags on EVERY gene symbol to restrict matches to title/abstract (this prevents noise from common-word gene names like "RAN").

Set `max_results=30`.

### Step 3: Make ONE metadata call
Call `get_article_metadata` exactly ONCE with all the PMIDs from Step 2.

### Step 4: Validate using the FULL cluster annotation
For each gene, examine the returned papers and decide:
- Does this paper actually address the SPECIFIC subprocess in the full annotation (not just the broad keyword)?
- A paper about "ribosome biogenesis in mitochondria" may be peripheral to a cluster about the "40S SSU processome" — flag accordingly.

Use the full annotation, not the search keyword, to make this judgment.

## CRITICAL constraints
- Make EXACTLY 2 tool calls total (1 search + 1 metadata)
- Do NOT search per-gene
- Do NOT call any tool more than once
- Do NOT call any other tools (no related_articles, no full_text, no biorxiv)

## Output
Produce an amended version of the gene entries:
- Add a "literature_validation" field to each entry
- If literature strongly supports a role in the full pathway annotation, suggest reclassification to ESTABLISHED
- If literature refines the evidence picture (e.g., NO_EVIDENCE → INDIRECT_EVIDENCE), update the subclass
- If no relevant hits, record literature_support: "none"

Return ONLY the amended gene entries as a JSON object with "novel_role_genes" and
"uncharacterized_genes" lists, preserving the original schema.
{literature_validation_output_format}
"""

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

COT_STEPS_DEFAULT = [
    CLUSTER_ANALYSIS_TASK,
    COT_SCREEN_CONTEXT,
    COT_STEP_PATHWAY_HYPOTHESIS,
    COT_STEP_GENE_CATEGORIZATION,
    COT_STEP_SUBCLASSIFICATION,
    COT_STEP_PATHWAY_SELECTION,
    COT_STEP_VERIFICATION,
    COT_STEP_OUTPUT,
]


def assemble_cot_instructions(
    steps: list[str] | None = None,
    screen_context: str | None = None,
) -> str:
    """Assemble COT instructions from modular steps.

    Args:
        steps: List of COT step strings. Defaults to COT_STEPS_DEFAULT.
        screen_context: Optional screen context JSON string. If provided and
            COT_SCREEN_CONTEXT is in steps, it will be replaced with the
            context header + actual context.

    Returns:
        Formatted COT instructions with numbered steps.
    """
    if steps is None:
        steps = COT_STEPS_DEFAULT

    if screen_context is not None:
        steps = [
            f"{COT_SCREEN_CONTEXT}\n{screen_context}" if step == COT_SCREEN_CONTEXT else step
            for step in steps
        ]

    numbered = [f"STEP {i + 1} - {step}" for i, step in enumerate(steps)]
    return "\n\n".join(numbered)


# =============================================================================
# PHENOTYPIC-STRENGTH-CONFIDENCE CROSS-CHECK (inserted by prompt_factory if phenotypic strength available)
# =============================================================================
# PLACEHOLDER: To be implemented
#
# Purpose: Cross-validate pathway confidence against phenotypic strength to identify edge cases
# Timing: AFTER establishing pathway confidence in Section 5
#
# This section should:
# - Present the phenotypic strength for the cluster (e.g., "8.5/10" or "strong"/"weak")
# - Cross-check against the confidence level just assigned
# - Identify and flag four scenarios:
#   * HIGH confidence + HIGH strength → Affirm: "Well-supported, strong signal - ideal case"
#   * LOW confidence + LOW strength → Affirm: "Weak signal - appropriately uncertain"
#   * HIGH confidence + LOW strength → FLAG: "Reconsider - is the pathway too broad/generic? Are you overcalling confidence?"
#   * LOW confidence + HIGH strength → FLAG: "Deep dive needed - strong effect suggests important biology. You may be missing the true pathway or this could be novel."
# - For mismatched cases, prompt re-examination of the pathway hypothesis
# - Request updated reasoning in the summary if confidence should be adjusted
#
# Example structure:
# """
# PHENOTYPIC-STRENGTH-CONFIDENCE CROSS-CHECK:
# Your pathway assignment: {confidence_level} confidence
# Phenotypic strength: {strength_level} (score: {strength_value})
#
# Evaluate the alignment:
# - ALIGNED (High/High or Low/Low): Your assessment is consistent with effect strength
# - MISMATCH (High confidence + Low strength): Are you overcalling confidence? Is the pathway too generic?
# - MISMATCH (Low confidence + High strength): Strong phenotype suggests important biology - dig deeper for the true pathway. This could be a discovery opportunity.
#
# If mismatched, revisit your pathway hypothesis and explain your reasoning in the summary.
# """

PHENOTYPIC_STRENGTH_CONFIDENCE_EVALUATION = None  # Placeholder for future implementation


# =============================================================================
# MECHANISTIC HYPOTHESIS FROM FEATURE DIRECTIONALITY (inserted by prompt_factory if features available)
# =============================================================================
# PLACEHOLDER: To be implemented
#
# Purpose: Generate mechanistic hypotheses based on which features are up/down regulated
# Timing: AFTER confidence assessment and phenotypic strength check - uses pathway context to interpret directionality
#
# This section should:
# - Present up-regulated vs down-regulated features/genes/imaging features
# - Guide hypothesis generation about WHAT IS HAPPENING mechanistically in this cluster
# - Connect directional changes to specific pathway components or processes
# - For perturbation screens (Perturb-seq): suggest which pathway branch/component is being affected
# - For imaging screens (OPS): suggest which cellular processes are altered based on feature changes
# - Frame as hypothesis generation to guide experiments, NOT as validation/refinement of the pathway
# - Create a bridge between pathway identification and experimental design
#
# Example structure:
# """
# MECHANISTIC HYPOTHESIS (based on feature directionality):
# The following features show consistent directional changes across genes in this cluster:
#
# UP-REGULATED: {up_features}
# DOWN-REGULATED: {down_features}
#
# Given the {pathway_name} pathway you identified, generate mechanistic hypotheses:
# - What might be happening at the molecular/cellular level? (e.g., "Upregulation of X with downregulation of Y suggests activation of the upstream regulatory branch")
# - Which specific components or branches of the pathway are likely affected?
# - Are these changes consistent with activation, inhibition, feedback, or compensation within the pathway?
# - What cellular process is being altered to produce these specific directional changes?
# - Does this suggest a particular mechanistic model within the broader pathway?
#
# Frame your mechanistic hypothesis to set up follow-up experimental validation.
# """

FEATURE_DIRECTIONALITY_HYPOTHESIS = None  # Placeholder for future implementation


# =============================================================================
# FOLLOW-UP EXPERIMENT SUGGESTIONS (inserted by prompt_factory if enabled)
# =============================================================================
# PLACEHOLDER: To be implemented
#
# Purpose: Suggest specific, actionable experiments to validate pathway assignment and test mechanistic hypotheses
# Timing: FINAL analytical section - after pathway, confidence, phenotypic strength, and mechanistic hypothesis
#
# This section should:
# - Suggest 2-4 concrete, specific follow-up experiments
# - Target three goals:
#   1. Validate the pathway assignment (confirm pathway involvement)
#   2. Test high-priority genes (validate UNCHARACTERIZED and NOVEL_ROLE genes)
#   3. Test mechanistic hypotheses (if feature directionality data available)
# - Be specific: name genes to test, specific assays/techniques, expected readouts
# - Consider the experimental modality (genetic perturbations, biochemical assays, imaging, epistasis, etc.)
# - Tailor suggestions to confidence level and phenotypic strength:
#   * High confidence + high strength: focus on mechanism and novel genes
#   * Mismatched cases: suggest experiments to resolve the discrepancy
# - Prioritize experiments that test high-priority (score ≥8) genes
# - Be actionable and implementable in a real lab setting
#
# Example structure:
# """
# SUGGESTED FOLLOW-UP EXPERIMENTS:
# Based on your analysis (pathway: {pathway_name}, confidence: {confidence}, phenotypic strength: {strength}, mechanism: {hypothesis}), suggest 2-4 specific experiments:
#
# 1. VALIDATE PATHWAY ASSIGNMENT:
#    - Experiment: [specific assay/technique]
#    - Genes to test: [which established genes as positive controls]
#    - Expected outcome: [what would confirm pathway involvement]
#
# 2. TEST HIGH-PRIORITY GENES:
#    - For UNCHARACTERIZED genes: [specific experiment to test pathway role]
#    - For NOVEL_ROLE genes: [specific experiment to test proposed new function]
#    - Focus on priority ≥8 genes: [list specific genes]
#    - Expected outcomes: [what would validate their roles]
#
# 3. TEST MECHANISTIC HYPOTHESIS (if directionality available):
#    - Experiment: [assay to test your mechanistic model]
#    - Readouts: [what measurements distinguish between mechanisms]
#    - Expected outcomes: [results that support/refute hypothesis]
#
# 4. RESOLVE AMBIGUITY (if applicable):
#    - For low confidence: [experiment to increase confidence in pathway]
#    - For strength mismatch: [experiment to explain the discrepancy]
#    - For heterogeneous clusters: [experiment to identify subclusters or alternative pathways]
#
# Be specific and actionable: name genes, assays, techniques, and expected results.
# """

FOLLOW_UP_EXPERIMENT_SUGGESTIONS = None  # Placeholder for future implementation


# =============================================================================
# ON HANDLING RETRIEVED EVIDENCE (inserted by prompt_factory when RAG enabled)
# =============================================================================
# Evidence snippets from knowledge base retrieval are inserted here
