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
# LITERATURE VALIDATION (mode-agnostic MCP step)
# Used in single_mcp / cot_mcp / stepwise_mcp — exactly 2 MCP tool calls.
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

STEPS_DEFAULT = [
    CLUSTER_ANALYSIS_TASK,
    COT_SCREEN_CONTEXT,
    COT_STEP_PATHWAY_HYPOTHESIS,
    COT_STEP_GENE_CATEGORIZATION,
    COT_STEP_SUBCLASSIFICATION,
    COT_STEP_PATHWAY_SELECTION,
    COT_STEP_VERIFICATION,
    COT_STEP_OUTPUT,
]

# MCP variant: literature validation is inserted AFTER sub-classification but
# BEFORE pathway selection/verification, so updated gene categories can affect
# the percent-fit calculation and therefore pathway confidence.
STEPS_DEFAULT_MCP = [
    CLUSTER_ANALYSIS_TASK,
    COT_SCREEN_CONTEXT,
    COT_STEP_PATHWAY_HYPOTHESIS,
    COT_STEP_GENE_CATEGORIZATION,
    COT_STEP_SUBCLASSIFICATION,
    STEP_LITERATURE_VALIDATION,
    COT_STEP_PATHWAY_SELECTION,
    COT_STEP_VERIFICATION,
    COT_STEP_OUTPUT,
]

