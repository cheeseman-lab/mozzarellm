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

GENE_CLASSIFICATION_RULES = f"""
When classifying and prioritizing genes, apply these specific criteria:

1. ESTABLISHED PATHWAY GENES:
   - Genes with well-documented roles in the identified pathway
   - Supported by multiple publications demonstrating direct involvement
   - Often serve as canonical members or markers of the pathway

2. UNCHARACTERIZED GENES:
   No manuscript has established this gene's molecular function in human cells. This ranges from completely unstudied genes to genes with domain annotations or non-human characterization only.

3. NOVEL_ROLE GENES:
   At least one manuscript has focused on this gene's molecular function, but in a different pathway. The gene's known role is outside the identified pathway — it may represent a novel connection or a contradictory hit.

BOUNDARY RULES:
- ESTABLISHED vs NOVEL_ROLE: If a gene already has published evidence for this specific pathway, it is ESTABLISHED — not NOVEL_ROLE. NOVEL_ROLE is reserved for genes whose known function is in a different pathway.
- ESTABLISHED vs UNCHARACTERIZED: A gene is characterized if at least one manuscript focuses on its molecular function (e.g., a paper titled after the gene, or a study that dissects its mechanism). Such genes are ESTABLISHED or NOVEL_ROLE, never UNCHARACTERIZED — even if their role in this particular pathway is unclear.
- NOVEL_ROLE vs UNCHARACTERIZED: If no manuscript has focused on a gene's molecular function, it is UNCHARACTERIZED — not NOVEL_ROLE. NOVEL_ROLE requires an established function in another pathway.
"""
NOVEL_CLASSIFICATION_RULES = """
Classification classes for NOVEL_ROLE genes (genes with established functions in other pathways):

  NO_EVIDENCE: No clear tie to the pathway at hand, but could represent new biology.
  INDIRECT_EVIDENCE: Has a logical tie to the pathway based on basic rules of cell biology.
  PARTIAL_EVIDENCE: Has some preliminary functional data tying to the pathway, but not established as a core player.
  CONTRADICTORY_EVIDENCE: Completely illogical tie to this pathway; incompatible with existing literature of the pathway and the known function of this gene.

Assign each NOVEL_ROLE gene exactly one class. Even for CONTRADICTORY_EVIDENCE genes, provide a confidence assessment on likelihood of pathway involvement.

Then assign a separate priority score (1-10) representing overall follow-up priority, considering the class, evidence quality, pathway relevance, and experimental tractability.
"""

UNCHARACTERIZED_CLASSIFICATION_RULES = """
Classification classes for UNCHARACTERIZED genes (genes with limited or no functional annotation):

  DARK_GENE: Unnamed gene with no functional characterization.
  NASCENT: Unnamed gene with some preliminary functional characterization.
  ANNOTATED_ONLY: Named gene (e.g., transmembrane domain, coiled-coil, transcription factor domain) but poorly characterized.
  NON_HUMAN_CHARACTERIZED: Characterized but not in human cells.

Assign each UNCHARACTERIZED gene exactly one class. Even for NON_HUMAN_CHARACTERIZED genes, provide a confidence assessment on likelihood of pathway involvement.

Then assign a separate priority score (1-10) representing overall follow-up priority, considering the class, evidence quality, pathway relevance, and experimental tractability.
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

If there is no coherent pathway, set:
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

COT_INSTRUCTIONS = f"""
STEP 1 - PATHWAY HYPOTHESIS (2-3 candidates):
- Review gene annotations
- List 2-3 candidate pathways with supporting genes
- Note which annotations support each hypothesis

STEP 2 - PATHWAY SELECTION:
- Select dominant pathway based on:
  * Number of established genes with direct roles
  * Coherence of functional relationships
  * Quality of supporting evidence (prioritize high-relevance snippets)
- Assign confidence level using the following criteria: {PATHWAY_CONFIDENCE_CRITERIA}

STEP 3 - GENE CLASSIFICATION (cite evidence):
For each gene, determine ONE category according to the following rules: {GENE_CLASSIFICATION_RULES}

STEP 4 - PRIORITIZATION (scores 1-10):
- For NOVEL_ROLE genes: Score based on the following criteria: {NOVEL_CLASSIFICATION_RULES}
- For UNCHARACTERIZED genes: Score based on the following criteria: {UNCHARACTERIZED_CLASSIFICATION_RULES}
- Cite specific annotations that inform each priority score

STEP 5 - VERIFICATION:
- Check for contradictions
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

# this could be passed to claude as a tool
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
      "class": "DARK_GENE",
      "priority": 8,
      "rationale": "explanation"
    }
  ],
  "novel_role_genes": [
    {
      "gene": "GeneD",
      "class": "NO_EVIDENCE",
      "priority": 7,
      "rationale": "explanation"
    }
  ],
  "summary": "key findings summary"
}
"""
NEW_OUTPUT_FORMAT_JSON = """
Provide a concise analysis in this exact JSON format:
{
  "cluster_id": "[CLUSTER_ID]",  # IMPORTANT: Use the exact cluster_id provided in the prompt
  "dominant_process": "specific pathway name",
  "pathway_confidence": "High/Medium/Low",
  "established_genes": ["GeneA", "GeneB"],
  "uncharacterized_genes": [
    {
      "gene": "GeneC",
      "class": "DARK_GENE",
      "rationale": "explanation"
      "evidence": "quote from annotations, and citations if present"
    }
  ],
  "novel_role_genes": [
    {
      "gene": "GeneD",
      "class": "NO_EVIDENCE",
      "priority": 7,
      "rationale": "explanation"
      "evidence": "quote from annotations, and citations if present"
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
