"""
Prompt templates and instructions for gene cluster analysis.

This module contains modular prompt components organized in the order they appear
in the final assembled prompt. Components are automatically concatenated by the
prompt factory.
"""

# =============================================================================
# CORE TASK (always first)
# =============================================================================

CLUSTER_ANALYSIS_TASK = """
MISSION: Functional genomics experiments cluster genes by phenotypic similarity. Your goal is to:
1. Identify the dominant biological pathway that explains why these genes cluster together
2. Classify ALL genes relative to this pathway (ESTABLISHED / UNCHARACTERIZED / NOVEL_ROLE)
3. Prioritize understudied genes (UNCHARACTERIZED and NOVEL_ROLE) for follow-up experiments

The pathway is not the end goal - it's the lens for discovering which genes merit investigation.
"""

# =============================================================================
# GENE CLASSIFICATION & PRIORITIZATION RULES (framework for analysis)
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
# PATHWAY CONFIDENCE ASSESSMENT (comes AFTER data to enable assessment)
# =============================================================================

PATHWAY_CONFIDENCE_CRITERIA = """
ASSESSING PATHWAY CONFIDENCE:

Once you have identified a candidate pathway, evaluate how well it explains the cluster using
these stringent criteria based on what percentage of genes fit the proposed pathway:

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

NO COHERENT PATHWAY:
- <30% of genes in the cluster fit any single proposed pathway
- Clusters where genes belong to many unrelated pathways
- Clusters containing nontargeting control genes
- Clusters where you cannot identify a dominant biological process

For clusters with no coherent pathway, set:
- "pathway_confidence": "Low"
- "dominant_process": "No coherent biological pathway"
- And explain the reasoning clearly in the "summary" field

Remember: The goal is to honestly assess pathway support, not to force-fit genes into pathways.
Low confidence clusters may still contain valuable discovery opportunities if individual genes
are understudied.
"""

# =============================================================================
# CHAIN-OF-THOUGHT INSTRUCTIONS (inserted by prompt_factory when CoT enabled)
# =============================================================================

ENHANCED_COT_INSTRUCTIONS = """
STEP 1 - PATHWAY HYPOTHESIS (2-3 candidates):
- Review gene annotations
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
# OUTPUT FORMAT (always last)
# =============================================================================

# this could be passed tp claude as a tool
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
