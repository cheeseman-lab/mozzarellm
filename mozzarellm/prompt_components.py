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
# LITERATURE VALIDATION PROMPTS
# Mode A (Structured MCP): CoT call → MCP Query Structurer → STRUCTURED_MCP_REFINEMENT_PROMPT
# Mode B (Direct MCP): single call using DIRECT_MCP_VALIDATION_PROMPT
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
DIRECT_MCP_VALIDATION_PROMPT = """
I need to validate gene-pathway associations from a functional genomics screen and produce an amended classification.

## Pathway context
Pathway: "{pathway}"

## Genes to validate
The following genes were classified as NOVEL_ROLE or UNCHARACTERIZED relative to this pathway:

{flagged_genes_json}

## Task
For each gene above, search PubMed and bioRxiv using the query:
  "<GENE_SYMBOL> {pathway}"

Use this exact query structure — do not search for the gene's function broadly.
Retrieve at most 3 papers per gene. Only papers with direct relevance to "{pathway}" count.

Based on what you find, produce an amended version of the gene entries:
- Add a "literature_validation" field to each entry
- If literature strongly supports a role in "{pathway}", suggest reclassification to ESTABLISHED
- If literature refines the evidence picture (e.g., NO_EVIDENCE → INDIRECT_EVIDENCE), update the subclass
- If no relevant hits, record literature_support: "none"

Return ONLY the amended gene entries as a JSON object with "novel_role_genes" and
"uncharacterized_genes" lists, preserving the original schema.
{literature_validation_output_format}
"""

# Placeholders: {cluster_id}, {pathway}, {pathway_confidence}, {call_1_summary}, {flagged_genes_json},
#               {literature_validation_output_format}
STRUCTURED_MCP_REFINEMENT_PROMPT = """
You are performing a targeted literature search to refine a gene cluster analysis.

## Cluster context (from initial analysis)
Cluster: {cluster_id}
Pathway: "{pathway}" (confidence: {pathway_confidence})
Initial analysis summary: {call_1_summary}

## Genes requiring literature review
The following genes were flagged as NOVEL_ROLE or UNCHARACTERIZED relative to "{pathway}":

{flagged_genes_json}

## Task
For each gene above, search PubMed and bioRxiv using the query:
  "<GENE_SYMBOL> {pathway}"

Use this exact query structure — do not search for the gene's function broadly.
Retrieve at most 3 papers per gene per source. Only papers with direct relevance to
"{pathway}" count — ignore hits about the gene's function in unrelated contexts.

Based on what you find:
- NOVEL_ROLE gene with direct pathway evidence → suggest reclassification to ESTABLISHED
- Evidence that refines the subclass (e.g., NO_EVIDENCE → INDIRECT_EVIDENCE) → update subclass
- UNCHARACTERIZED gene with new characterization data in this pathway → update subclass
- No relevant hits → record literature_support: "none"

Return ONLY a JSON object with "novel_role_genes" and "uncharacterized_genes" lists,
preserving the original schema.
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

# MATTEO EDIT: implement when phenotypic strength data is available from Brieflow.
# Cross-validates pathway confidence against phenotypic effect size — flags mismatches
# (high confidence + weak phenotype → overcalling; strong phenotype + low confidence → dig deeper).
PHENOTYPIC_STRENGTH_CONFIDENCE_EVALUATION = None

# MATTEO EDIT: implement when feature directionality (up/down regulated features) is available.
# Generates mechanistic hypotheses from which imaging features or expression changes are
# up vs. down across the cluster — bridges pathway ID to experimental design.
FEATURE_DIRECTIONALITY_HYPOTHESIS = None

# MATTEO EDIT: implement after PHENOTYPIC_STRENGTH and FEATURE_DIRECTIONALITY are in place.
# Suggests 2-4 concrete follow-up experiments targeting: pathway validation, high-priority
# novel/uncharacterized genes, and mechanistic hypotheses.
FOLLOW_UP_EXPERIMENT_SUGGESTIONS = None
