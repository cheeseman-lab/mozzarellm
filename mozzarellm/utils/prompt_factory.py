import logging
from .config_utils import get_prompt_path


def load_prompt_template(
    template_path=None, template_string=None, template_type="cluster"
):
    """
    Load a prompt template from a file or string.
    """
    # First, find and load the template
    if template_path:
        print(f"Attempting to load template from: {template_path}")

        # Use the utility function to find the prompt template
        resolved_path = get_prompt_path(template_path)
        if resolved_path:
            print(f"Template found at: {resolved_path}")
            try:
                with open(resolved_path, "r") as f:
                    template = f.read()
                print(f"Successfully loaded template ({len(template)} characters)")
            except Exception as e:
                print(f"Failed to load template from {resolved_path}: {e}")
                logging.error(f"Failed to load template from {resolved_path}: {e}")
                # Fall back to default template
                template = get_default_template(template_type)
        else:
            print(f"Could not find prompt template: {template_path}")
            logging.warning(f"Could not find prompt template: {template_path}")
            # Fall back to default template
            template = get_default_template(template_type)
    elif template_string:
        print("Using provided template string")
        template = template_string
    else:
        print(f"Using default template for type: {template_type}")
        template = get_default_template(template_type)

    # Now add the output format instructions if needed
    output_format = get_output_format_instructions(template_type)

    # Fix the issue by escaping curly braces in the output format that aren't meant to be placeholders
    output_format = output_format.replace("{", "{{").replace("}", "}}")
    # But restore any actual placeholders
    output_format = output_format.replace("{{clusters_text}}", "{clusters_text}")

    if output_format not in template:
        print("Appending output format instructions to template")
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
    elif template_type == "batch_cluster":
        return """
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
    cluster_id, genes, gene_features=None, screen_info=None, template_path=None
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
    if screen_info:
        screen_context = f"""
SCREEN INFORMATION:
{screen_info}

Use this information to better understand the biological context of the screen and inform your assessment of potential novel pathway roles.
"""
        prompt += screen_context

    # Add gene features if provided - only for genes in this cluster
    if gene_features:
        feature_text = "\nAdditional gene information:\n"
        relevant_feature_count = 0

        for gene in genes:
            if gene in gene_features:
                feature_text += f"{gene}: {gene_features[gene]}\n"
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
    clusters, gene_features=None, screen_info=None, template_path=None
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
    if screen_info:
        screen_context = f"""
SCREEN INFORMATION:
{screen_info}

Use this information to better understand the biological context of the screen and inform your assessment of potential novel pathway roles.
"""
        prompt += screen_context

    # Add gene features if provided - OPTIMIZED to only include genes in this batch
    if gene_features:
        feature_text = "\nAdditional gene information:\n"
        relevant_feature_count = 0

        # Only include features for genes in this batch
        for gene, features in gene_features.items():
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
