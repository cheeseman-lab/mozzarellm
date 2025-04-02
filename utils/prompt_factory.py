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