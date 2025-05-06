def load_prompt_template(
    template_path=None, template_string=None, template_type="cluster"
):
    """
    Load a prompt template from file, string, or constants.
    """
    import os

    # First attempt to use provided template string
    if template_string:
        print("Using provided template string")
        template = template_string

    # Then try to load from file if provided
    elif template_path:
        print(f"Attempting to load template from: {template_path}")

        if os.path.exists(template_path):
            try:
                with open(template_path, "r") as f:
                    template = f.read()
                print(f"Successfully loaded template ({len(template)} characters)")
            except Exception as e:
                print(f"Failed to load template from {template_path}: {e}")
                # Fall back to constants
                template = get_default_template(template_type)
        else:
            print(f"Could not find prompt template: {template_path}")
            # Fall back to constants
            template = get_default_template(template_type)

    # Otherwise use constants
    else:
        print(f"Using default template for type: {template_type}")
        template = get_default_template(template_type)

    # Add output format if needed
    output_format = get_output_format_instructions(template_type)
    if output_format not in template:
        print("Appending output format instructions to template")
        template += f"\n\n{output_format}"

    return template


def get_default_template(template_type):
    """Get the default template constant for a given type"""
    from mozzarellm.prompts import DEFAULT_CLUSTER_PROMPT, DEFAULT_BATCH_PROMPT

    if template_type == "cluster":
        return DEFAULT_CLUSTER_PROMPT
    elif template_type == "batch_cluster":
        return DEFAULT_BATCH_PROMPT
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
{{
  "cluster_id": "[CLUSTER_ID]",  # IMPORTANT: Use the exact cluster_id provided in the prompt
  "dominant_process": "specific pathway name",
  "pathway_confidence": "High/Medium/Low/No coherent pathway",
  "established_genes": ["GeneA", "GeneB"],
  "uncharacterized_genes": [
    {{
      "gene": "GeneC",
      "priority": 8,
      "rationale": "explanation"
    }}
  ],
  "novel_role_genes": [
    {{
      "gene": "GeneD",
      "priority": 7,
      "rationale": "explanation"
    }}
  ],
  "summary": "key findings summary"
}}
"""
    elif template_type == "batch_cluster":
        return """
Provide analysis for each cluster in this exact JSON array format:

[
  {{
    "cluster_id": "[CLUSTER_ID]",  # IMPORTANT: Use the exact cluster_id as provided in the prompt for each cluster
    "dominant_process": "specific pathway name",
    "pathway_confidence": "High",
    "established_genes": ["GeneA", "GeneB"],
    "uncharacterized_genes": [
      {{
        "gene": "GeneC",
        "priority": 8,
        "rationale": "explanation"
      }}
    ],
    "novel_role_genes": [
      {{
        "gene": "GeneD",
        "priority": 7,
        "rationale": "explanation"
      }}
    ],
    "summary": "key findings summary"
  }}
]

CRITICAL: Your response MUST classify *every gene* into one of the three categories and include all three categories in the output.
CRITICAL: You MUST maintain the exact cluster_id as provided in the prompt for each cluster in your response.
Your response MUST be a valid JSON array starting with '[' and ending with ']'. Do not include explanations outside the JSON.
"""
    else:
        raise ValueError(f"Unknown template type: {template_type}")


def make_cluster_analysis_prompt(
    cluster_id,
    genes,
    gene_annotations_dict=None,
    screen_context=None,
    template_path=None,
):
    """
    Create a prompt for gene cluster analysis with concise JSON output focusing on both
    truly uncharacterized genes and characterized genes with potential novel pathway roles.
    Optimized version that only includes relevant gene features.

    Args:
        cluster_id: Identifier for the cluster
        genes: List of gene identifiers in the cluster
        gene_features: Optional dict of additional gene features
        screen_info: Optional information about the OPS screen and biological context
        template_path: Path to custom template file

    Returns:
        prompt: Formatted prompt string
    """
    gene_list = ", ".join(genes)

    # Load template (will fall back to default if needed)
    template = load_prompt_template(
        template_path=template_path, template_type="cluster"
    )

    # Format template with cluster_id and gene_list (ensure cluster_id is a string)
    prompt = template.format(cluster_id=str(cluster_id), gene_list=gene_list)

    # Add screen information if provided
    if screen_context:
        screen_context = f"""
SCREEN INFORMATION:
{screen_context}

"""
        prompt += screen_context

    # Add gene features if provided - only for genes in this cluster
    if gene_annotations_dict:
        feature_text = "\nAdditional gene information:\n"
        relevant_feature_count = 0

        for gene in genes:
            if gene in gene_annotations_dict:
                feature_text += f"{gene}: {gene_annotations_dict[gene]}\n"
                relevant_feature_count += 1

        # Only add the feature section if we found relevant features
        if relevant_feature_count > 0:
            feature_explanation = """
IMPORTANT: The additional gene information provided above should be used to:
1. Better determine if genes are truly UNCHARACTERIZED
2. Evaluate potential pathway connections for CHARACTERIZED_OTHER_PATHWAY genes
3. Identify ESTABLISHED genes for the dominant process
"""
            prompt += feature_text + feature_explanation
            print(f"Added {relevant_feature_count} gene feature descriptions to prompt")
        else:
            print("No relevant gene features found for this cluster")

    return prompt


def make_batch_cluster_analysis_prompt(
    clusters, gene_annotations_dict=None, screen_context=None, template_path=None
):
    """
    Create a prompt for batch analysis of multiple gene clusters with concise output,
    distinguishing between uncharacterized genes and characterized genes with novel roles.
    Optimized to only include gene features for genes present in the current batch.
    """
    # Create formatted clusters text and collect genes present in this batch
    clusters_text = ""
    batch_genes = set()  # Use a set for efficient lookups

    for cluster_id, genes in clusters.items():
        gene_list = ", ".join(genes)
        clusters_text += f"Cluster {cluster_id}: {gene_list}\n\n"
        # Add all genes from this cluster to our set
        batch_genes.update(genes)

    # Load template (will fall back to default if needed)
    template = load_prompt_template(
        template_path=template_path, template_type="batch_cluster"
    )

    # Format template with clusters_text
    prompt_vars = {"clusters_text": clusters_text}
    try:
        prompt = template.format(**prompt_vars)
    except KeyError:
        # Handle errors by escaping all braces and then restoring only our placeholder
        escaped_template = template.replace("{", "{{").replace("}", "}}")
        escaped_template = escaped_template.replace(
            "{{clusters_text}}", "{clusters_text}"
        )
        prompt = escaped_template.format(**prompt_vars)

    # Add screen information if provided
    if screen_context:
        screen_context = f"""
SCREEN INFORMATION:
{screen_context}

"""
        prompt += screen_context

    # Add gene features if provided - OPTIMIZED to only include genes in this batch
    if gene_annotations_dict:
        feature_text = "\nAdditional gene information:\n"
        relevant_feature_count = 0

        # Only include features for genes in this batch
        for gene, features in gene_annotations_dict.items():
            if gene in batch_genes:
                feature_text += f"{gene}: {features}\n"
                relevant_feature_count += 1

        # Only add the feature section if we found relevant features
        if relevant_feature_count > 0:
            feature_explanation = """
IMPORTANT: The additional gene information provided above should be used to:
1. Better determine if genes are truly UNCHARACTERIZED
2. Evaluate potential pathway connections for CHARACTERIZED_OTHER_PATHWAY genes
3. Identify ESTABLISHED genes for the dominant process
"""
            prompt += feature_text + feature_explanation
            print(f"Added {relevant_feature_count} gene feature descriptions to prompt")
        else:
            print("No relevant gene features found for this batch")

    return prompt
