# mozzarellm/prompts.py

# Single cluster analysis prompt
DEFAULT_CLUSTER_PROMPT = """
Analyze gene cluster {cluster_id} to identify the dominant biological pathway and classify genes:

Genes: {gene_list}

Follow these steps:
1. Identify the dominant biological pathway, focusing on specific molecular mechanisms rather than general terms
2. Classify genes into THREE categories using these definitions:
   - ESTABLISHED: Well-known members of the identified pathway with clear functional roles in this pathway
   - UNCHARACTERIZED: Genes with minimal to no functional annotation in ANY published literature
   - NOVEL_ROLE: Genes with published functional annotation in OTHER pathways that may have additional roles in the dominant pathway

3. For both UNCHARACTERIZED and NOVEL_ROLE genes:
   - Assign a priority score (1-10) for follow-up investigation
   - Provide a rationale explaining why this gene merits investigation

4. Provide a concise summary of the key findings
"""

# Batch analysis prompt
DEFAULT_BATCH_PROMPT = """
Analyze the following gene clusters to identify dominant biological pathways and classify genes:

{clusters_text}

For each cluster:
1. Identify the dominant biological pathway, focusing on specific molecular mechanisms rather than general terms
2. Classify genes into THREE categories using these definitions:
   - ESTABLISHED: Well-known members of the identified pathway with clear functional roles in this pathway
   - UNCHARACTERIZED: Genes with minimal to no functional annotation in ANY published literature
   - NOVEL_ROLE: Genes with published functional annotation in OTHER pathways that may have additional roles in the dominant pathway

3. For both UNCHARACTERIZED and NOVEL_ROLE genes:
   - Assign a priority score (1-10) for follow-up investigation
   - Provide a rationale explaining why this gene merits investigation

4. Provide a concise summary of the key findings for each cluster
"""

# Optimized context for robustly analyzing gene clusters
ROBUST_SCREEN_CONTEXT = """
Genes grouped within a cluster tend to exhibit similar morphological phenotypes in this context, suggesting that they may participate in the same biological process or pathway. However, not all clusters will correspond to a defined or coherent biological pathway.

When evaluating pathway confidence, apply these stringent criteria:

HIGH CONFIDENCE:
- Multiple well-established genes (≥3) with strong literature support in the same specific pathway
- Clear functional relationship between genes that explains the observed phenotypic clustering
- Genes represent different aspects or components of the same biological process
- The pathway assignment explains >60% of genes in the cluster

MEDIUM CONFIDENCE:
- Some established genes (1-2) from a specific pathway, with additional supporting genes
- Functional relationship is plausible but has some gaps or uncertainties
- Some genes in the cluster have unclear relationship to the proposed pathway
- The pathway assignment explains 40-60% of genes in the cluster

LOW CONFIDENCE:
- Few or no established pathway genes, but a plausible functional theme
- Significant heterogeneity in gene functions within the cluster
- The proposed pathway is very broad or general
- The pathway assignment explains <40% of genes in the cluster

CLUSTERS WITH NO COHERENT PATHWAY:
- For clusters with no clear functional relationship among genes
- Clusters where genes belong to many unrelated pathways
- Clusters containing nontargeting control genes
- Clusters where you cannot identify a dominant biological process

For clusters with no coherent pathway, set:
- "pathway_confidence": "Low"
- "dominant_process": "No coherent biological pathway"
- And explain the reasoning clearly in the "summary" field

The goal is NOT to force-fit clusters into pathways, but to identify clusters where a clear biological signal emerges from the phenotypic grouping. Mark clusters without a coherent biological signature as indicated above rather than assigning biologically implausible pathways.
"""

# Optimized context for analyzing gene clusters with specific pathway focus
ROBUST_CLUSTER_PROMPT = """
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
