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

# Output format for single cluster
CLUSTER_OUTPUT_FORMAT = """
Provide a concise analysis in this exact JSON format:
{
  "cluster_id": "[CLUSTER_ID]",  # IMPORTANT: Use the exact cluster_id provided in the prompt
  "dominant_process": "specific pathway name",
  "pathway_confidence": "High/Medium/Low/No coherent pathway",
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

# HeLa interphase screen info
HELA_SCREEN_INFO = """
These gene clusters were derived from an optical pooled screening (OPS) approach in interphase HeLa cells.

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
