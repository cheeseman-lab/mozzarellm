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


def load_prompt_template(template_path=None, template_string=None, template_type="cluster"):
    """
    Load a prompt template from a file or string.
    
    Args:
        template_path: Path to template file
        template_string: Template string
        template_type: Type of template (cluster, gene_set, batch_cluster)
        
    Returns:
        template: The loaded template string
    """
    if template_path:
        try:
            with open(template_path, 'r') as f:
                template = f.read()
        except Exception as e:
            logging.error(f"Failed to load template from {template_path}: {e}")
            # Fall back to default template
            template = get_default_template(template_type)
    elif template_string:
        template = template_string
    else:
        template = get_default_template(template_type)
    
    # Ensure the template contains the required output format instructions
    output_format = get_output_format_instructions(template_type)
    if output_format not in template:
        # Append output format to template
        template += f"\n\n{output_format}"
        
    return template

def get_default_template(template_type):
    """Get the default template for a given type"""
    if template_type == "gene_set":
        return """
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
    elif template_type == "cluster":
        return """
CONTEXT: You are an AI assistant specializing in genomics and systems biology with expertise in pathway analysis. 

Analyze gene cluster {cluster_id} to identify the dominant biological function and prioritize genes in TWO distinct categories:

Genes: {gene_list}

Follow these steps:
1. Identify the dominant biological process, focusing on specific pathways rather than general terms
2. Classify genes into ESTABLISHED pathway members and potential NOVEL pathway members
3. For potential novel pathway members, assign a priority score (1-10) for follow-up investigation
"""
    elif template_type == "batch_cluster":
        return """
CONTEXT: You are an AI assistant specializing in genomics and systems biology with expertise in pathway analysis. 

Analyze the following gene clusters to identify dominant biological functions and prioritize genes:

{clusters_text}

Follow these steps:
1. Identify the dominant biological process for each cluster
2. Classify genes into ESTABLISHED pathway members and potential NOVEL pathway members
3. For potential novel pathway members, assign a priority score (1-10) for follow-up investigation
"""
    else:
        raise ValueError(f"Unknown template type: {template_type}")

def get_output_format_instructions(template_type):
    """Get standardized output format instructions for a template type"""
    if template_type == "gene_set":
        return """
Format your response as:
FUNCTION NAME: [your concise function name]
CONFIDENCE SCORE: [0.0-1.0 numerical score]
ANALYSIS: [your detailed explanation]
"""
    elif template_type == "cluster":
        return """
Provide a concise analysis in this exact JSON format:
{
  "cluster_id": cluster_id_value,
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
    elif template_type == "batch_cluster":
        return """
Provide analysis for each cluster in this exact JSON array format:
[
  {
    "cluster_id": cluster_id_value,
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
  },
  {
    // Next cluster...
  }
]
"""
    else:
        raise ValueError(f"Unknown template type: {template_type}")

def make_gene_analysis_prompt(genes, gene_features=None, template_path=None):
    """
    Create a prompt for gene set analysis with a custom or default template.
    
    Args:
        genes: List of gene identifiers
        gene_features: Optional dict of additional gene features
        template_path: Path to custom template file
        
    Returns:
        prompt: Formatted prompt string
    """
    gene_list = ", ".join(genes)
    
    # Load template (will fall back to default if needed)
    template = load_prompt_template(template_path=template_path, template_type="gene_set")
    
    # Format the template with gene list
    prompt = template.format(gene_list=gene_list)
    
    # Add gene features if provided
    if gene_features:
        feature_text = "\nAdditional gene information:\n"
        for gene, features in gene_features.items():
            if gene in genes:
                feature_text += f"{gene}: {features}\n"
        prompt += feature_text
    
    return prompt
