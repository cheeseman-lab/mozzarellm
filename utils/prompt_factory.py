def make_user_prompt_with_score(genes, gene_features=None):
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
    Create a prompt for gene cluster analysis with focus on pathway discovery.
    
    Args:
        cluster_id: Identifier for the cluster
        genes: List of gene identifiers in the cluster
        gene_features: Optional dict of additional gene features
        
    Returns:
        prompt: Formatted prompt string
    """
    gene_list = ", ".join(genes)
    
    prompt = f"""
Analyze the following gene cluster (Cluster {cluster_id}) to identify the dominant biological function and potential novel pathway members:

Genes: {gene_list}

Follow these specific steps in your analysis:
1. Determine if >50% of these genes share a common biological function or pathway
2. If a dominant process is identified:
   - Name the specific biological process (avoid general terms like "cell signaling")
   - List genes that are established members of this process
   - Identify genes in the cluster that are not yet linked to this process but may be related
   - Provide evidence for why each novel gene might participate in the process

Output your analysis in this exact format:

Cluster ID: {cluster_id}
Dominant Process Name: [specific biological process name]
LLM confidence: (High/Medium/Low)

Known pathway members:
- [Gene]: [Brief description of established role]
[repeat for each known member]

Potential novel members:
- [Gene]: [Specific evidence from protein domains, localization, expression patterns, or interactions]
[repeat for each potential novel member]

Summary hypothesis:
[1-2 sentences proposing how the novel genes might contribute to the pathway]

Important requirements:
- If no clear dominant function (>50% genes), mark as "Functionally diverse cluster"
- Assign confidence based on proportion of known pathway genes and strength of evidence for novel members
- Explicitly state when a gene's potential role is purely speculative
- When evidence is contradictory, acknowledge both supporting and opposing factors
"""

    if gene_features:
        feature_text = "\nAdditional gene information:\n"
        for gene, features in gene_features.items():
            if gene in genes:
                feature_text += f"{gene}: {features}\n"
        prompt += feature_text
    
    return prompt

def make_batch_cluster_prompt(clusters, gene_features=None):
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
        clusters_text += f"Cluster {cluster_id}: {gene_list}\n"
    
    prompt = f"""
Analyze the following gene clusters to identify dominant biological functions and potential novel pathway members:

{clusters_text}

For each cluster, determine if >50% of genes share a common biological function or pathway and follow these steps:
1. If a dominant process is identified:
   - Name the specific biological process (avoid general terms like "cell signaling")
   - List genes that are established members of this process
   - Identify genes not yet linked to this process but may be related
   - Provide evidence for why each novel gene might participate in the process

For each cluster, use this exact format:

Cluster ID: [number]
Dominant Process Name: [specific biological process name]
LLM confidence: (High/Medium/Low)

Known pathway members:
- [Gene]: [Brief description of established role]

Potential novel members:
- [Gene]: [Specific evidence from protein domains, localization, expression patterns, or interactions]

Summary hypothesis:
[1-2 sentences proposing how the novel genes might contribute to the pathway]

Important requirements:
- If no clear dominant function (>50% genes), mark as "Functionally diverse cluster"
- Assign confidence based on proportion of known pathway genes and strength of evidence for novel members
- Explicitly state when a gene's potential role is purely speculative
- When evidence is contradictory, acknowledge both supporting and opposing factors
"""

    if gene_features:
        feature_text = "\nAdditional gene information:\n"
        for gene, features in gene_features.items():
            feature_text += f"{gene}: {features}\n"
        prompt += feature_text
    
    return prompt