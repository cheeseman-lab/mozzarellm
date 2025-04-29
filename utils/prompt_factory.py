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


def make_cluster_analysis_prompt(cluster_id, genes, gene_features=None, screen_info=None):
    """
    Create a prompt for gene cluster analysis with concise JSON output focusing on both
    truly uncharacterized genes and characterized genes with potential novel pathway roles.

    Args:
        cluster_id: Identifier for the cluster
        genes: List of gene identifiers in the cluster
        gene_features: Optional dict of additional gene features
        screen_info: Optional information about the OPS screen and biological context

    Returns:
        prompt: Formatted prompt string
    """
    gene_list = ", ".join(genes)

    context = """
CONTEXT: You are an AI assistant specializing in genomics and systems biology with expertise in pathway analysis. Your task is to analyze gene clusters from optical pooled screening data to identify biological pathways and potential novel pathway members.

You must distinguish between two key discovery types:
1. TRULY UNCHARACTERIZED genes with little to no published functional data
2. CHARACTERIZED genes that may have novel roles in the identified pathway

For each type, use different evaluation criteria:
- For UNCHARACTERIZED genes: prioritize based solely on the absence of biological work, regardless of the cluster context
- For CHARACTERIZED genes with potential novel roles: prioritize based solely on evidence for biological interplay with established pathway genes

Prioritize conservative scoring that emphasizes high-confidence discoveries, as experimental validation requires significant resources. For ambiguous cases, clearly state limitations in your assessment rather than overextending interpretations.

IMPORTANT: Do not use co-clustering as evidence for prioritization - the presence of genes in the same cluster should not influence your scoring.
"""

    screen_context = ""
    if screen_info:
        screen_context = f"""
SCREEN INFORMATION:
{screen_info}

Use this information to better understand the biological context of the screen and inform your assessment of potential novel pathway roles. Genes that might interact with the identified pathway in the specific context of this screen should be prioritized accordingly.
"""

    prompt = f"""
{context}
{screen_context}

Analyze gene cluster {cluster_id} to identify the dominant biological function and prioritize genes in TWO distinct categories:

Genes: {gene_list}

Follow these steps:
1. Identify the dominant biological process, focusing on specific pathways rather than general terms

2. Classify genes into THREE MUTUALLY EXCLUSIVE categories using these strict definitions:
   - ESTABLISHED: Well-known members of the identified pathway with clear functional roles in this pathway
   - CHARACTERIZED_OTHER_PATHWAY: Genes with published functional annotation in OTHER pathways, not the dominant one
   - UNCHARACTERIZED: Genes with minimal to no functional annotation in ANY published literature

3. For UNCHARACTERIZED genes, use these STRICTER criteria - a gene is UNCHARACTERIZED if ANY of these are true:
   - Fewer than 5 publications specifically focused on this gene
   - Function is described with terms like "putative," "predicted," "hypothesized," or "potential"
   - No experimental validation of specific molecular function (even if there are structural predictions)
   - Limited or conflicting data about its biological role
   - Known only by sequence homology or computational predictions

4. For UNCHARACTERIZED genes, assign a follow-up priority score (1-10) based ONLY on the absence of biological work:
   - 8-10: Highest priority - completely unstudied gene (0-2 publications)
   - 6-7: High priority - extremely limited characterization (3-5 publications)
   - 4-5: Medium priority - some preliminary characterization but function remains unclear
   - 1-3: Lower priority - more characterized than initially appeared

5. For CHARACTERIZED_OTHER_PATHWAY genes, evaluate the potential for interaction with the dominant process as follows:
   - Consider molecular evidence for cross-pathway interactions
   - Look for shared regulatory mechanisms
   - Evaluate structural or functional similarities with established pathway components
   - Assess if the gene's known function could complement or regulate the dominant process

   Then assign a priority score (1-10) based SOLELY on evidence for biological connection:
   - 8-10: Highest priority - direct experimental evidence of interaction with established pathway components
   - 6-7: High priority - strong indirect evidence suggesting functional connection to the pathway
   - 4-5: Medium priority - some evidence suggesting possible pathway connection or regulatory role
   - 1-3: Lower priority - weak or speculative connection to the established pathway

6. IMPORTANT: In clusters with Medium or High pathway confidence, EVERY gene MUST be assigned to exactly one category (ESTABLISHED, CHARACTERIZED_OTHER_PATHWAY, or UNCHARACTERIZED). Do not leave any genes uncategorized.

NOTE: Only analyze clusters with a "Medium" or "High" pathway confidence. If the dominant process cannot be determined with at least medium confidence, mark the cluster as "Functionally diverse" with "Low" confidence and do not prioritize any genes.

Provide a concise analysis in this exact JSON format:
{{
  "cluster_id": {cluster_id},
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
      "rationale": "one-sentence explanation focusing ONLY on potential interaction with the dominant process"
    }}
  ],
  "summary": "one-sentence hypothesis about the most promising genes from both categories"
}}

For clusters with no clear dominant function or Low pathway confidence, use:
{{
  "cluster_id": {cluster_id},
  "dominant_process": "Functionally diverse cluster",
  "pathway_confidence": "Low",
  "established_genes": [],
  "uncharacterized_genes": [],
  "novel_role_genes": []
}}

Important:
- The entire response should be a single JSON object for the cluster
- Be SOMEWHAT LIBERAL in classifying genes as "UNCHARACTERIZED" - if there's any doubt, limited information, or contradicting functional roles, classify it as UNCHARACTERIZED
- For UNCHARACTERIZED genes, priority should be based SOLELY on how little is known about the gene
- For CHARACTERIZED_OTHER_PATHWAY genes, priority should be based SOLELY on evidence for interaction with the dominant process
- Do NOT use co-clustering as evidence for prioritization
- If pathway confidence is "Low", do not prioritize ANY genes and use the "Functionally diverse cluster" template
- Be conservative in scoring - experimental validation is expensive and time-consuming
- Count publication numbers carefully when determining if a gene is uncharacterized
"""

    if gene_features:
        feature_text = "\nAdditional gene information:\n"
        for gene, features in gene_features.items():
            if gene in genes:
                feature_text += f"{gene}: {features}\n"
        
        feature_explanation = """
IMPORTANT: The additional gene information provided above should be used to:
1. Better determine if genes are truly UNCHARACTERIZED
2. Evaluate potential pathway connections for CHARACTERIZED_OTHER_PATHWAY genes
3. Identify ESTABLISHED genes for the dominant process

This information contains UniProt annotations for genes, which includes protein domains, cellular localization, and functional descriptions. Pay careful attention to the language used in these annotations - terms like "hypothesized," "predicted," "putative," "potential," or "by similarity" indicate that the function is not experimentally confirmed. Genes with only hypothesized or predicted functions should still be considered relatively uncharacterized. Only treat functions as established when they are described without such qualifying language and have been experimentally validated.
"""
        prompt += feature_text + feature_explanation

    return prompt


def make_batch_cluster_analysis_prompt(clusters, gene_features=None, screen_info=None):
    """
    Create a prompt for batch analysis of multiple gene clusters with concise output,
    distinguishing between uncharacterized genes and characterized genes with novel roles.

    Args:
        clusters: Dictionary mapping cluster IDs to lists of genes
        gene_features: Optional dict of additional gene features
        screen_info: Optional information about the OPS screen and biological context

    Returns:
        prompt: Formatted prompt string
    """
    clusters_text = ""
    for cluster_id, genes in clusters.items():
        gene_list = ", ".join(genes)
        clusters_text += f"Cluster {cluster_id}: {gene_list}\n\n"

    context = """
CONTEXT: You are an AI assistant specializing in genomics and systems biology with expertise in pathway analysis. Your task is to analyze gene clusters from optical pooled screening data to identify biological pathways and potential novel pathway members.

You must distinguish between two key discovery types:
1. TRULY UNCHARACTERIZED genes with little to no published functional data
2. CHARACTERIZED genes that may have novel roles in the identified pathway

For each type, use different evaluation criteria:
- For UNCHARACTERIZED genes: prioritize based solely on the absence of biological work, regardless of the cluster context
- For CHARACTERIZED genes with potential novel roles: prioritize based solely on evidence for biological interplay with established pathway genes

Prioritize conservative scoring that emphasizes high-confidence discoveries, as experimental validation requires significant resources. For ambiguous cases, clearly state limitations in your assessment rather than overextending interpretations.

IMPORTANT: Do not use co-clustering as evidence for prioritization - the presence of genes in the same cluster should not influence your scoring.
"""

    screen_context = ""
    if screen_info:
        screen_context = f"""
SCREEN INFORMATION:
{screen_info}

Use this information to better understand the biological context of the screen and inform your assessment of potential novel pathway roles. Genes that might interact with the identified pathway in the specific context of this screen should be prioritized accordingly.
"""

    prompt = f"""
{context}
{screen_context}

Analyze the following gene clusters to identify dominant biological functions and prioritize genes in TWO distinct categories:

{clusters_text}

Follow these steps:
1. Identify the dominant biological process, focusing on specific pathways rather than general terms

2. Classify genes into THREE MUTUALLY EXCLUSIVE categories using these strict definitions:
   - ESTABLISHED: Well-known members of the identified pathway with clear functional roles in this pathway
   - CHARACTERIZED_OTHER_PATHWAY: Genes with published functional annotation in OTHER pathways, not the dominant one
   - UNCHARACTERIZED: Genes with minimal to no functional annotation in ANY published literature

3. For UNCHARACTERIZED genes, use these STRICTER criteria - a gene is UNCHARACTERIZED if ANY of these are true:
   - Fewer than 5 publications specifically focused on this gene
   - Function is described with terms like "putative," "predicted," "hypothesized," or "potential"
   - No experimental validation of specific molecular function (even if there are structural predictions)
   - Limited or conflicting data about its biological role
   - Known only by sequence homology or computational predictions

4. For UNCHARACTERIZED genes, assign a follow-up priority score (1-10) based ONLY on the absence of biological work:
   - 8-10: Highest priority - completely unstudied gene (0-2 publications)
   - 6-7: High priority - extremely limited characterization (3-5 publications)
   - 4-5: Medium priority - some preliminary characterization but function remains unclear
   - 1-3: Lower priority - more characterized than initially appeared

5. For CHARACTERIZED_OTHER_PATHWAY genes, evaluate the potential for interaction with the dominant process as follows:
   - Consider molecular evidence for cross-pathway interactions
   - Look for shared regulatory mechanisms
   - Evaluate structural or functional similarities with established pathway components
   - Assess if the gene's known function could complement or regulate the dominant process

   Then assign a priority score (1-10) based SOLELY on evidence for biological connection:
   - 8-10: Highest priority - direct experimental evidence of interaction with established pathway components
   - 6-7: High priority - strong indirect evidence suggesting functional connection to the pathway
   - 4-5: Medium priority - some evidence suggesting possible pathway connection or regulatory role
   - 1-3: Lower priority - weak or speculative connection to the established pathway

6. IMPORTANT: In clusters with Medium or High pathway confidence, EVERY gene MUST be assigned to exactly one category (ESTABLISHED, CHARACTERIZED_OTHER_PATHWAY, or UNCHARACTERIZED). Do not leave any genes uncategorized.

NOTE: Only analyze clusters with a "Medium" or "High" pathway confidence. If the dominant process cannot be determined with at least medium confidence, mark the cluster as "Functionally diverse" with "Low" confidence and do not prioritize any genes.

Provide a concise analysis in this exact JSON format for each cluster:
{{
  "cluster_id": 0,  // Use only the numeric ID (e.g., 0, 1, 2) not "Cluster 0"
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
      "rationale": "one-sentence explanation focusing ONLY on potential interaction with the dominant process"
    }}
  ],
  "summary": "one-sentence hypothesis about the most promising genes from both categories"
}}

For clusters with no clear dominant function or Low pathway confidence, use:
{{
  "cluster_id": 0,  // Use only the numeric ID (e.g., 0, 1, 2) not "Cluster 0"
  "dominant_process": "Functionally diverse cluster",
  "pathway_confidence": "Low",
  "established_genes": [],
  "uncharacterized_genes": [],
  "novel_role_genes": []
}}

Important:
- The entire response should be a single JSON array containing all cluster analyses
- Be SOMEWHAT LIBERAL in classifying genes as "UNCHARACTERIZED" - if there's any doubt, limited information, or contradicting functional roles, classify it as UNCHARACTERIZED
- For UNCHARACTERIZED genes, priority should be based SOLELY on how little is known about the gene
- For CHARACTERIZED_OTHER_PATHWAY genes, priority should be based SOLELY on evidence for interaction with the dominant process
- Do NOT use co-clustering as evidence for prioritization
- If pathway confidence is "Low", do not prioritize ANY genes and use the "Functionally diverse cluster" template
- CRITICAL: Use ONLY numeric values for cluster_id (0, 1, 2), not text like "Cluster 0"
- Be conservative in scoring - experimental validation is expensive and time-consuming
- Count publication numbers carefully when determining if a gene is uncharacterized
"""

    if gene_features:
        feature_text = "\nAdditional gene information:\n"
        for gene, features in gene_features.items():
            feature_text += f"{gene}: {features}\n"
        
        feature_explanation = """
IMPORTANT: The additional gene information provided above should be used to:
1. Better determine if genes are truly UNCHARACTERIZED
2. Evaluate potential pathway connections for CHARACTERIZED_OTHER_PATHWAY genes
3. Identify ESTABLISHED genes for the dominant process

This information contains UniProt annotations for genes, which includes protein domains, cellular localization, and functional descriptions. Pay careful attention to the language used in these annotations - terms like "hypothesized," "predicted," "putative," "potential," or "by similarity" indicate that the function is not experimentally confirmed. Genes with only hypothesized or predicted functions should still be considered relatively uncharacterized. Only treat functions as established when they are described without such qualifying language and have been experimentally validated.
"""
        prompt += feature_text + feature_explanation

    return prompt