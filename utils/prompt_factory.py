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
    Create a prompt for gene cluster analysis with standardized JSON output.
    Works for single cluster analysis.

    Args:
        cluster_id: Identifier for the cluster
        genes: List of gene identifiers in the cluster
        gene_features: Optional dict of additional gene features

    Returns:
        prompt: Formatted prompt string
    """
    gene_list = ", ".join(genes)

    prompt = f"""
Analyze gene cluster {cluster_id} to identify the dominant biological function and prioritize genes for follow-up:

Genes: {gene_list}

Follow these steps:
1. Identify the dominant biological process, focusing on specific pathways rather than general terms
2. Classify each gene as either:
   - ESTABLISHED: Well-known member of the identified pathway with clear functional role
   - CHARACTERIZED: Gene with known functions, but not primarily associated with this pathway
   - UNCHARACTERIZED: Gene with minimal functional annotation or unclear biological role

3. Score each gene on a follow-up priority scale (1-10):
   - 8-10: Highest priority - uncharacterized genes in well-defined pathways OR characterized genes 
     with strong evidence for unexpected associations that challenge current paradigms
   - 6-7: High priority - characterized genes with preliminary evidence for novel pathway associations
     OR uncharacterized genes in less well-defined pathways
   - 4-5: Medium priority - genes with emerging roles in this pathway requiring validation
   - 2-3: Low priority - genes likely involved but with extensive existing literature
   - 1: Not recommended for follow-up - well-established pathway components

Provide your analysis in this EXACT JSON format:
{{
  "cluster_id": "{cluster_id}",
  "dominant_process": "specific pathway name",
  "pathway_confidence": "High (>70% of genes are established pathway members)/Medium (40-70% established)/Low (<40% established)",
  "genes": [
    {{
      "gene": "GeneA",
      "classification": "ESTABLISHED/CHARACTERIZED/UNCHARACTERIZED",
      "follow_up_priority": 1-10,
      "rationale": "one-sentence explanation of classification and priority score"
    }}
  ],
  "summary": "one-sentence hypothesis about unexpected gene-pathway associations"
}}

If no clear dominant function (>50% genes share a common pathway), return:
{{
  "cluster_id": "{cluster_id}",
  "dominant_process": "Functionally diverse cluster",
  "pathway_confidence": "Low",
  "genes": []
}}

Important requirements:
- Output VALID JSON that can be parsed programmatically
- Prioritize poorly characterized or uncharacterized genes with unclear functions
- Give high scores to genes with known functions appearing in unexpected pathway contexts
- Consider genes that might have context-specific roles under these experimental conditions
- Be conservative in your scoring - experimental validation is expensive and time-consuming
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
    Create a prompt for batch analysis of multiple gene clusters.

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
Analyze the following gene clusters to identify dominant biological functions and prioritize genes for follow-up:

{clusters_text}

For each cluster, follow these steps:
1. Identify the dominant biological process, focusing on specific pathways rather than general terms
2. Classify each gene as either:
   - ESTABLISHED: Well-known member of the identified pathway with clear functional role
   - CHARACTERIZED: Gene with known functions, but not primarily associated with this pathway
   - UNCHARACTERIZED: Gene with minimal functional annotation or unclear biological role

3. Score each gene on a follow-up priority scale (1-10):
   - 8-10: Highest priority - uncharacterized genes in well-defined pathways OR characterized genes 
     with strong evidence for unexpected associations that challenge current paradigms
   - 6-7: High priority - characterized genes with preliminary evidence for novel pathway associations
     OR uncharacterized genes in less well-defined pathways
   - 4-5: Medium priority - genes with emerging roles in this pathway requiring validation
   - 2-3: Low priority - genes likely involved but with extensive existing literature
   - 1: Not recommended for follow-up - well-established pathway components

For each cluster, provide an analysis as part of a JSON array with this exact format:
[
  {{
    "cluster_id": "cluster_number",
    "dominant_process": "specific pathway name",
    "pathway_confidence": "High (>70% of genes are established pathway members)/Medium (40-70% established)/Low (<40% established)",
    "genes": [
      {{
        "gene": "GeneA",
        "classification": "ESTABLISHED/CHARACTERIZED/UNCHARACTERIZED",
        "follow_up_priority": 1-10,
        "rationale": "one-sentence explanation of classification and priority score"
      }}
    ],
    "summary": "one-sentence hypothesis about unexpected gene-pathway associations"
  }}
]

For clusters with no clear dominant function (>50% genes share a common pathway), use:
{{
  "cluster_id": "cluster_number",
  "dominant_process": "Functionally diverse cluster",
  "pathway_confidence": "Low",
  "genes": []
}}

Important requirements:
- Output VALID JSON that can be parsed programmatically
- The entire response should be a single JSON array containing all cluster analyses
- Prioritize poorly characterized or uncharacterized genes with unclear functions
- Give high scores to genes with known functions appearing in unexpected pathway contexts
- Consider genes that might have context-specific roles under these experimental conditions
- Be conservative in your scoring - experimental validation is expensive and time-consuming
"""

    if gene_features:
        feature_text = "\nAdditional gene information:\n"
        for gene, features in gene_features.items():
            feature_text += f"{gene}: {features}\n"
        prompt += feature_text

    return prompt
