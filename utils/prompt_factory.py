def make_gene_analysis_prompt(genes, gene_features=None):
    """
    Create a prompt for gene set analysis with explicit score request

    Args:
        genes: List of gene identifiers
        gene_features: Optional dict of additional gene features

    Returns:
        prompt: Formatted prompt string
    """
    gene_list = ", ".join(genes)

    prompt = f"""
I have a set of genes and need to identify their shared biological function or pathway.

Genes: {gene_list}

Please analyze this gene set and:
1. Provide a concise, specific name for the biological function/pathway represented by these genes
2. Give a confidence score (0.0-1.0) that reflects your certainty in this annotation
3. Explain your reasoning with references to key genes and their roles

Format your response as:
FUNCTION NAME: [your concise function name]
CONFIDENCE SCORE: [0.0-1.0 numerical score]
ANALYSIS: [your detailed explanation]
"""

    if gene_features:
        feature_text = "\nAdditional gene information:\n"
        for gene, features in gene_features.items():
            if gene in genes:
                feature_text += f"{gene}: {features}\n"
        prompt += feature_text

    return prompt


def make_cluster_analysis_prompt(cluster_id, genes, gene_features=None):
    """
    Create a prompt for gene cluster analysis with concise JSON output focusing on both
    novel genes and known genes with potential novel pathway roles.

    Args:
        cluster_id: Identifier for the cluster
        genes: List of gene identifiers in the cluster
        gene_features: Optional dict of additional gene features

    Returns:
        prompt: Formatted prompt string
    """
    gene_list = ", ".join(genes)

    prompt = f"""
Analyze gene cluster {cluster_id} to identify the dominant biological function and prioritize genes in TWO distinct categories:

Genes: {gene_list}

Follow these steps:
1. Identify the dominant biological process, focusing on specific pathways rather than general terms

2. Classify genes into THREE categories using these strict definitions:
   - ESTABLISHED: Well-known members of the identified pathway with clear functional roles in this pathway
   - CHARACTERIZED_OTHER_PATHWAY: Genes with published functional annotation in OTHER pathways, not the dominant one
   - UNCHARACTERIZED: Genes with minimal to no functional annotation in ANY published literature

3. For UNCHARACTERIZED genes, apply these stringent criteria - a gene is only UNCHARACTERIZED if ALL of these are true:
   - Limited or no experimental validation of function in any pathway
   - Few or no publications specifically focused on this gene
   - Unknown molecular function or biological process
   - No characterized protein domains that clearly indicate function
   - Lacks established interaction partners that would suggest function

4. For UNCHARACTERIZED genes, assign a follow-up priority score (1-10) based ONLY on the absence of biological work:
   - 8-10: Highest priority - completely unstudied gene (virtually no publications)
   - 6-7: High priority - extremely limited characterization (1-2 preliminary studies)
   - 4-5: Medium priority - some preliminary characterization but function remains unclear
   - 1-3: Lower priority - more characterized than initially appeared

5. For CHARACTERIZED_OTHER_PATHWAY genes, assign a priority score (1-10) based ONLY on potential biological interplay with ESTABLISHED genes:
   - 8-10: Highest priority - strong evidence for novel interaction with established pathway genes
   - 6-7: High priority - good evidence for interaction with established pathway genes
   - 4-5: Medium priority - some evidence suggesting possible pathway connection
   - 1-3: Lower priority - limited evidence of interaction with established pathway genes

NOTE: Do NOT use the mere presence of a gene in this cluster as evidence for prioritization. Evaluate only based on biological evidence from the literature.

Provide a concise analysis in this exact JSON format:
{{
  "cluster_id": "{cluster_id}",
  "dominant_process": "specific pathway name",
  "pathway_confidence": "High/Medium/Low",
  "established_genes": ["GeneA", "GeneB"],
  "uncharacterized_genes": [
    {{
      "gene": "GeneC",
      "priority": 8,
      "rationale": "one-sentence explanation focusing ONLY on its lack of characterization"
    }}
  ],
  "novel_role_genes": [
    {{
      "gene": "GeneD",
      "priority": 7,
      "rationale": "one-sentence explanation focusing ONLY on potential interaction with established pathway genes"
    }}
  ],
  "summary": "one-sentence hypothesis about the most promising genes from both categories"
}}

For clusters with no clear dominant function, use:
{{
  "cluster_id": "{cluster_id}",
  "dominant_process": "Functionally diverse cluster",
  "pathway_confidence": "Low",
  "established_genes": [],
  "uncharacterized_genes": [],
  "novel_role_genes": []
}}

Important:
- The entire response should be a single JSON object for the cluster
- Be extremely strict about what qualifies as "UNCHARACTERIZED" - any gene with published functional studies should be classified as CHARACTERIZED_OTHER_PATHWAY
- For UNCHARACTERIZED genes, priority should be based SOLELY on how little is known about the gene
- For CHARACTERIZED_OTHER_PATHWAY genes, priority should be based SOLELY on evidence for interaction with the established pathway genes
- Do NOT use co-clustering as evidence for prioritization
- Be conservative in scoring - experimental validation is expensive and time-consuming
"""

    if gene_features:
        feature_text = "\nAdditional gene information:\n"
        for gene, features in gene_features.items():
            if gene in genes:
                feature_text += f"{gene}: {features}\n"
        prompt += feature_text

    return prompt


def make_batch_cluster_analysis_prompt(clusters, gene_features=None):
    """
    Create a prompt for batch analysis of multiple gene clusters with concise output,
    distinguishing between uncharacterized genes and characterized genes with novel roles.
    Args:
        clusters: Dictionary mapping cluster IDs to lists of genes
        gene_features: Optional dict of additional gene features
    Returns:
        prompt: Formatted prompt string
    """
    clusters_text = ""
    for cluster_id, genes in clusters.items():
        gene_list = ", ".join(genes)
        clusters_text += f"Cluster {cluster_id}: {gene_list}\n\n"
    
    prompt = f"""
Analyze the following gene clusters to identify dominant biological functions and prioritize genes in TWO distinct categories:
{clusters_text}
Follow these steps:
1. Identify the dominant biological process, focusing on specific pathways rather than general terms
2. Classify genes into THREE categories using these strict definitions:
   - ESTABLISHED: Well-known members of the identified pathway with clear functional roles in this pathway
   - CHARACTERIZED_OTHER_PATHWAY: Genes with published functional annotation in OTHER pathways, not the dominant one
   - UNCHARACTERIZED: Genes with minimal to no functional annotation in ANY published literature
3. For UNCHARACTERIZED genes, apply these stringent criteria - a gene is only UNCHARACTERIZED if ALL of these are true:
   - Limited or no experimental validation of function in any pathway
   - Few or no publications specifically focused on this gene
   - Unknown molecular function or biological process
   - No characterized protein domains that clearly indicate function
   - Lacks established interaction partners that would suggest function
4. For UNCHARACTERIZED genes, assign a follow-up priority score (1-10) based ONLY on the absence of biological work:
   - 8-10: Highest priority - completely unstudied gene (virtually no publications)
   - 6-7: High priority - extremely limited characterization (1-2 preliminary studies)
   - 4-5: Medium priority - some preliminary characterization but function remains unclear
   - 1-3: Lower priority - more characterized than initially appeared
5. For CHARACTERIZED_OTHER_PATHWAY genes, assign a priority score (1-10) based ONLY on potential biological interplay with ESTABLISHED genes:
   - 8-10: Highest priority - strong evidence for novel interaction with established pathway genes
   - 6-7: High priority - good evidence for interaction with established pathway genes
   - 4-5: Medium priority - some evidence suggesting possible pathway connection
   - 1-3: Lower priority - limited evidence of interaction with established pathway genes
NOTE: Do NOT use the mere presence of a gene in this cluster as evidence for prioritization. Evaluate only based on biological evidence from the literature.
Provide a concise analysis in this exact JSON format for each cluster:
{{
  "cluster_id": 0,  // IMPORTANT: Use only the numeric ID (e.g., 0, 1, 2) not "Cluster 0"
  "dominant_process": "specific pathway name",
  "pathway_confidence": "High/Medium/Low",
  "established_genes": ["GeneA", "GeneB"],
  "uncharacterized_genes": [
    {{
      "gene": "GeneC",
      "priority": 8,
      "rationale": "one-sentence explanation focusing ONLY on its lack of characterization"
    }}
  ],
  "novel_role_genes": [
    {{
      "gene": "GeneD",
      "priority": 7,
      "rationale": "one-sentence explanation focusing ONLY on potential interaction with established pathway genes"
    }}
  ],
  "summary": "one-sentence hypothesis about the most promising genes from both categories"
}}
For clusters with no clear dominant function, use:
{{
  "cluster_id": 0,  // IMPORTANT: Use only the numeric ID (e.g., 0, 1, 2) not "Cluster 0"
  "dominant_process": "Functionally diverse cluster",
  "pathway_confidence": "Low",
  "established_genes": [],
  "uncharacterized_genes": [],
  "novel_role_genes": []
}}
Important:
- The entire response should be a single JSON array containing all cluster analyses
- Be extremely strict about what qualifies as "UNCHARACTERIZED" - any gene with published functional studies should be classified as CHARACTERIZED_OTHER_PATHWAY
- For UNCHARACTERIZED genes, priority should be based SOLELY on how little is known about the gene
- For CHARACTERIZED_OTHER_PATHWAY genes, priority should be based SOLELY on evidence for interaction with the established pathway genes
- Do NOT use co-clustering as evidence for prioritization
- Be conservative in scoring - experimental validation is expensive and time-consuming
- CRITICAL: Use ONLY numeric values for cluster_id (0, 1, 2), not text like "Cluster 0"
"""
    if gene_features:
        feature_text = "\nAdditional gene information:\n"
        for gene, features in gene_features.items():
            feature_text += f"{gene}: {features}\n"
        prompt += feature_text
    return prompt