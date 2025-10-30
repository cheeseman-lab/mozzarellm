"""
Prompt templates and instructions for gene cluster analysis.

This module contains modular prompt components organized in the order they appear
in the final assembled prompt. Components are automatically concatenated by the
prompt factory.
"""

# =============================================================================
# SECTION 1: CORE TASK (always first)
# =============================================================================

CLUSTER_ANALYSIS_TASK = """
Analyze gene cluster {cluster_id} to identify the dominant biological pathway and classify genes:

Genes: {gene_list}

For each cluster:
1. Identify the dominant biological pathway, focusing on specific molecular mechanisms rather than general terms
2. For clusters with coherent biological signatures, classify each gene into one of three mutually exclusive categories:
   - ESTABLISHED: Well-known members of the identified pathway with clear functional roles in this pathway
   - UNCHARACTERIZED: Genes with minimal to no functional annotation in ANY published literature
   - NOVEL_ROLE: Genes with published functional annotation in OTHER pathways that may have additional roles in the dominant pathway

3. For both UNCHARACTERIZED and NOVEL_ROLE genes:
   - Assign a priority score (1-10) for follow-up investigation
   - Provide a rationale explaining why this gene merits investigation

4. Provide a concise summary of the key findings for each cluster
"""


# =============================================================================
# SECTION 2: GENE CLASSIFICATION & PRIORITIZATION RULES (always second)
# =============================================================================

GENE_CLASSIFICATION_RULES = """
When classifying and prioritizing genes, apply these specific criteria:

1. ESTABLISHED PATHWAY GENES:
   - Genes with well-documented roles in the identified pathway
   - Supported by multiple publications demonstrating direct involvement
   - Often serve as canonical members or markers of the pathway

2. UNCHARACTERIZED GENES:
   A gene is considered UNCHARACTERIZED only if MOST of these criteria are met:
   - Limited or no experimental validation of function in any pathway
   - Few (0-2) publications specifically focused on this gene
   - Unknown molecular function or biological process

   Priority scoring for UNCHARACTERIZED genes:
   - 8–10: Virtually unstudied (0–1 publications); uncharacterized molecular function; potential for novel discovery
   - 6–7: Extremely limited data; may have 1–2 preliminary findings but little known function
   - 4–5: Some characterization exists, but function remains unclear or incomplete
   - 1–3: Partial evidence for involvement in known pathways; not a strong candidate for novel discovery

3. NOVEL_ROLE GENES:
   These are genes with established functions in other pathways, but plausibly contribute to the identified pathway in a novel way.

   Priority scoring for NOVEL_ROLE genes:
   - 8–10: Compelling rationale for a previously unrecognized role in this pathway; role would be surprising and high-impact; minimal existing literature makes this a major discovery risk
   - 6–7: Some indirect or tangential evidence suggesting a new role; not previously linked to this pathway but fits plausibly
   - 4–5: Functional overlap or localization hints at a novel connection, but likely already speculated or partially known
   - 1–3: Existing data already supports involvement in this pathway; not a novel role — deprioritize

High scores should only be assigned if the novel connection is plausible but not already established. Genes with substantial evidence for the pathway should receive a lower priority, as they are not truly "novel."

IMPORTANT CONSIDERATIONS:
- Do NOT use mere presence in the cluster as evidence for prioritization
- Be conservative when assigning genes to the "NOVEL ROLE" or "UNCHARACTERIZED" categories. Do so only when there is a meaningful deviation from known pathway biology or a lack of functional annotation, respectively
- For any gene with substantial literature, it should NOT be classified as UNCHARACTERIZED
- The goal is not to speculate but to flag only the most promising candidates for follow-up
"""


# =============================================================================
# SECTION 3: SCREEN CONTEXT (inserted by prompt_factory)
# =============================================================================
# This section is where benchmark-specific or default context will be inserted.
# Benchmarks provide their own SCREEN_CONTEXT describing the experimental approach.

DEFAULT_SCREEN_CONTEXT = """
Genes grouped within a cluster exhibit similar profiles in this functional genomics
analysis, suggesting they may participate in related biological processes or pathways.
"""


# =============================================================================
# SECTION 4: PATHWAY CONFIDENCE ASSESSMENT (always included after screen context)
# =============================================================================

PATHWAY_CONFIDENCE_CRITERIA = """
When evaluating pathway confidence, apply these stringent criteria based on what percentage of
genes in the cluster are explained by the proposed pathway:

HIGH CONFIDENCE:
- >70% of genes in the cluster fit the proposed pathway
- Multiple well-established genes with strong literature support in this specific pathway
- Clear functional relationship between genes that explains the observed phenotypic clustering
- Genes represent different aspects or components of the same biological process

MEDIUM CONFIDENCE:
- 50-70% of genes in the cluster fit the proposed pathway
- Some established genes from the pathway, with additional plausible supporting genes
- Functional relationship is plausible but has some gaps or uncertainties
- Some genes in the cluster have unclear relationship to the proposed pathway

LOW CONFIDENCE:
- 30-50% of genes in the cluster fit the proposed pathway
- Few established pathway genes, but a plausible functional theme
- Significant heterogeneity in gene functions within the cluster
- The proposed pathway is very broad or general

CLUSTERS WITH NO COHERENT PATHWAY:
- <30% of genes in the cluster fit any single proposed pathway
- Clusters where genes belong to many unrelated pathways
- Clusters containing nontargeting control genes
- Clusters where you cannot identify a dominant biological process

For clusters with no coherent pathway, set:
- "pathway_confidence": "Low"
- "dominant_process": "No coherent biological pathway"
- And explain the reasoning clearly in the "summary" field

The goal is NOT to force-fit clusters into pathways, but to identify clusters where a clear
biological signal emerges from the phenotypic grouping. Assess what percentage of genes are
actually explained by the pathway - if most genes don't fit, the confidence should be low.
"""


# =============================================================================
# SECTION 5: RETRIEVED EVIDENCE (inserted by prompt_factory when RAG enabled)
# =============================================================================
# Evidence snippets from knowledge base retrieval are inserted here


# =============================================================================
# SECTION 6: CHAIN-OF-THOUGHT INSTRUCTIONS (inserted by prompt_factory when CoT enabled)
# =============================================================================

ENHANCED_COT_INSTRUCTIONS = """
STEP 1 - PATHWAY HYPOTHESIS (2-3 candidates):
- Review retrieved evidence [cite snippet numbers] and gene annotations
- List 2-3 candidate pathways with supporting genes
- Note which evidence snippets support each hypothesis

STEP 2 - PATHWAY SELECTION:
- Select dominant pathway based on:
  * Number of established genes with direct roles
  * Coherence of functional relationships
  * Quality of supporting evidence (prioritize high-relevance snippets)
- Assign confidence level (High/Medium/Low/None) using strict criteria from screen context

STEP 3 - GENE CLASSIFICATION (cite evidence):
For each gene, determine ONE category:
- ESTABLISHED: Well-documented role in selected pathway [cite evidence]
- UNCHARACTERIZED: Minimal functional annotation anywhere [note lack of evidence]
- NOVEL_ROLE: Known in other pathways but plausible new role here [cite supporting evidence]

STEP 4 - PRIORITIZATION (scores 1-10):
- For UNCHARACTERIZED genes: Score based on the following criteria:
  * Score 8-10: Gene absent from all retrieved evidence but fits pathway mechanistically
  * Score 4-7: Gene mentioned in 1-2 snippets with indirect pathway connection
  * Score 1-3: Gene well-documented in pathway across multiple evidence sources
- For NOVEL_ROLE genes: Score based on surprise/impact of proposed new role relative to known functions
- Cite specific evidence snippets that inform each priority score

STEP 5 - VERIFICATION:
- Check for contradictions between evidence snippets
- Verify all genes are classified (no omissions)
- Adjust confidence if evidence is weak or contradictory
- Note any gaps in evidence that limit conclusions

STEP 6 - FINAL JSON OUTPUT:
- Compile structured JSON with all required fields
- Ensure cluster_id matches input exactly
- Include concise summary highlighting key findings and evidence quality
"""

CONCISE_COT_INSTRUCTIONS = """
1) Identify 2-3 candidate pathways citing key genes and evidence snippets [numbers].
2) Classify each gene as ESTABLISHED / UNCHARACTERIZED / NOVEL_ROLE with 1-line rationale.
3) Assign priority scores (1-10) based on novelty and impact; cite supporting evidence.
4) Note contradictions or gaps in evidence; adjust confidence accordingly.
"""


# =============================================================================
# SECTION 7: GENE ANNOTATIONS (inserted by prompt_factory if provided)
# =============================================================================
# Gene-specific functional annotations from UniProt/databases are inserted here


# =============================================================================
# SECTION 8: OUTPUT FORMAT (always last)
# =============================================================================

OUTPUT_FORMAT_JSON = """
Provide a concise analysis in this exact JSON format:
{
  "cluster_id": "[CLUSTER_ID]",  # IMPORTANT: Use the exact cluster_id provided in the prompt
  "dominant_process": "specific pathway name",
  "pathway_confidence": "High/Medium/Low",
  "established_genes": ["GeneA", "GeneB"],
  "uncharacterized_genes": [
    {
      "gene": "GeneC",
      "priority": 8,
      "rationale": "explanation"
    }
  ],
  "novel_role_genes": [
    {
      "gene": "GeneD",
      "priority": 7,
      "rationale": "explanation"
    }
  ],
  "summary": "key findings summary"
}
"""
