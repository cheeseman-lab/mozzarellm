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
    Create a prompt for gene cluster analysis with concise JSON output focusing on novel genes.

    Args:
        cluster_id: Identifier for the cluster
        genes: List of gene identifiers in the cluster
        gene_features: Optional dict of additional gene features

    Returns:
        prompt: Formatted prompt string
    """
    gene_list = ", ".join(genes)

    prompt = f"""
Analyze gene cluster {cluster_id} to identify the dominant biological function and prioritize truly UNCHARACTERIZED genes for follow-up:

Genes: {gene_list}

Follow these steps:
1. Identify the dominant biological process, focusing on specific pathways rather than general terms
2. Classify genes into three categories using these strict definitions:
   - ESTABLISHED: Well-known members of the identified pathway with clear functional roles in this pathway
   - CHARACTERIZED: Genes with ANY published functional annotation, even if not associated with this pathway
   - UNCHARACTERIZED: Only genes with minimal to no functional annotation in published literature

Apply these stringent criteria for UNCHARACTERIZED genes - a gene is only UNCHARACTERIZED if ALL of these are true:
- Limited or no experimental validation of function in any pathway
- Few or no publications specifically focused on this gene
- Unknown molecular function or biological process
- No characterized protein domains that clearly indicate function
- Lacks established interaction partners that would suggest function

3. For each truly UNCHARACTERIZED gene, assign a follow-up priority score (1-10):
   - 8-10: Highest priority - completely novel gene with strong evidence for pathway involvement
   - 6-7: High priority - very limited characterization with good evidence for pathway connection
   - 4-5: Medium priority - some preliminary characterization but potential novel pathway role
   - 1-3: Lower priority - more characterized than initially appeared or weak pathway evidence

Provide a concise analysis in this exact JSON format:
{{
  "cluster_id": "{cluster_id}",
  "dominant_process": "specific pathway name",
  "pathway_confidence": "High/Medium/Low",
  "established_genes": ["GeneA", "GeneB"],
  "characterized_genes": ["GeneC", "GeneD"],
  "novel_genes": [
    {{
      "gene": "GeneE",
      "priority": 8,
      "rationale": "one-sentence explanation for why this uncharacterized gene may be a novel pathway member"
    }}
  ],
  "summary": "one-sentence hypothesis about the most promising novel genes"
}}

For clusters with no clear dominant function, use:
{{
  "cluster_id": "{cluster_id}",
  "dominant_process": "Functionally diverse cluster",
  "pathway_confidence": "Low",
  "established_genes": [],
  "characterized_genes": [],
  "novel_genes": []
}}

Important:
- The entire response should be a single JSON array for the cluster
- Be extremely strict about what qualifies as "UNCHARACTERIZED" - any gene with published functional studies should be classified as CHARACTERIZED
- Only assign high priority (8-10) to genes that are truly uncharacterized AND have strong evidence for meaningful pathway association
- When in doubt, classify a gene as CHARACTERIZED rather than UNCHARACTERIZED
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
    Create a prompt for batch analysis of multiple gene clusters with concise output.

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
Analyze the following gene clusters to identify dominant biological functions and prioritize UNCHARACTERIZED genes for follow-up:

{clusters_text}

Follow these steps:
1. Identify the dominant biological process, focusing on specific pathways rather than general terms
2. Classify genes into three categories using these strict definitions:
   - ESTABLISHED: Well-known members of the identified pathway with clear functional roles in this pathway
   - CHARACTERIZED: Genes with ANY published functional annotation, even if not associated with this pathway
   - UNCHARACTERIZED: Only genes with minimal to no functional annotation in published literature

Apply these stringent criteria for UNCHARACTERIZED genes - a gene is only UNCHARACTERIZED if ALL of these are true:
- Limited or no experimental validation of function in any pathway
- Few or no publications specifically focused on this gene
- Unknown molecular function or biological process
- No characterized protein domains that clearly indicate function
- Lacks established interaction partners that would suggest function

3. For each truly UNCHARACTERIZED gene, assign a follow-up priority score (1-10):
   - 8-10: Highest priority - completely novel gene with strong evidence for pathway involvement
   - 6-7: High priority - very limited characterization with good evidence for pathway connection
   - 4-5: Medium priority - some preliminary characterization but potential novel pathway role
   - 1-3: Lower priority - more characterized than initially appeared or weak pathway evidence

Provide a concise analysis in this exact JSON format:
{{
  "cluster_id": "cluster_number",
  "dominant_process": "specific pathway name",
  "pathway_confidence": "High/Medium/Low",
  "established_genes": ["GeneA", "GeneB"],
  "characterized_genes": ["GeneC", "GeneD"],
  "novel_genes": [
    {{
      "gene": "GeneE",
      "priority": 8,
      "rationale": "one-sentence explanation for why this uncharacterized gene may be a novel pathway member"
    }}
  ],
  "summary": "one-sentence hypothesis about the most promising novel genes"
}}

For clusters with no clear dominant function, use:
{{
  "cluster_id": "{cluster_id}",
  "dominant_process": "Functionally diverse cluster",
  "pathway_confidence": "Low",
  "established_genes": [],
  "characterized_genes": [],
  "novel_genes": []
}}

Important:
- The entire response should be a single JSON array containing all cluster analyses
- Be extremely strict about what qualifies as "UNCHARACTERIZED" - any gene with published functional studies should be classified as CHARACTERIZED
- Only assign high priority (8-10) to genes that are truly uncharacterized AND have strong evidence for meaningful pathway association
- When in doubt, classify a gene as CHARACTERIZED rather than UNCHARACTERIZED
- Be conservative in scoring - experimental validation is expensive and time-consuming
"""

    if gene_features:
        feature_text = "\nAdditional gene information:\n"
        for gene, features in gene_features.items():
            feature_text += f"{gene}: {features}\n"
        prompt += feature_text

    return prompt
