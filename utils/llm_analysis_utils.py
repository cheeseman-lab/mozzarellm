import json
import re
import pandas as pd
import logging
import os
import time
import datetime


def process_analysis(analysis_text):
    """
    Process the raw analysis text from an LLM into structured components.

    Args:
        analysis_text: Raw text response from LLM

    Returns:
        function_name: Extracted function/pathway name
        confidence_score: Extracted confidence score
        detailed_analysis: Extracted detailed analysis
    """
    # Default values in case parsing fails
    function_name = "Unknown"
    confidence_score = "0.0"
    detailed_analysis = analysis_text

    # Try to extract function name
    name_match = re.search(
        r"FUNCTION NAME:?\s*(.*?)(?:\n|$)", analysis_text, re.IGNORECASE
    )
    if name_match:
        function_name = name_match.group(1).strip()

    # Try to extract confidence score
    score_match = re.search(
        r"CONFIDENCE SCORE:?\s*([\d\.]+)", analysis_text, re.IGNORECASE
    )
    if score_match:
        confidence_score = score_match.group(1).strip()

    # Try to extract detailed analysis
    analysis_match = re.search(
        r"ANALYSIS:?\s*([\s\S]*?)(?:$|FUNCTION NAME|CONFIDENCE SCORE)",
        analysis_text,
        re.IGNORECASE,
    )
    if analysis_match:
        detailed_analysis = analysis_match.group(1).strip()

    return function_name, confidence_score, detailed_analysis


def process_cluster_analysis(analysis_text):
    """
    Process the cluster analysis output from an LLM into structured components.

    Args:
        analysis_text: Raw text response from LLM

    Returns:
        result_dict: Dictionary with structured analysis
    """
    # Initialize result dictionary
    result_dict = {
        "cluster_id": None,
        "dominant_process": None,
        "confidence": None,
        "known_members": {},
        "novel_members": {},
        "summary_hypothesis": None,
        "raw_text": analysis_text,
    }

    # Extract cluster ID
    cluster_match = re.search(r"Cluster ID:?\s*(\w+)", analysis_text)
    if cluster_match:
        result_dict["cluster_id"] = cluster_match.group(1).strip()

    # Extract dominant process - updated regex to be more flexible
    process_match = re.search(
        r"Dominant Process Name:?\s*(.*?)(?:\n|$)", analysis_text, re.IGNORECASE
    )
    if process_match:
        result_dict["dominant_process"] = process_match.group(1).strip()

    # Extract confidence level - updated to handle different formats
    confidence_match = re.search(
        r"LLM confidence:?\s*(High|Medium|Low)", analysis_text, re.IGNORECASE
    )
    if confidence_match:
        result_dict["confidence"] = confidence_match.group(1).strip()

    # Improved regex for extracting known pathway members section
    known_section_match = re.search(
        r"Known pathway members:(.*?)(?:Potential novel members:|$)",
        analysis_text,
        re.DOTALL,
    )
    if known_section_match:
        known_section = known_section_match.group(1).strip()
        # Process each line in the known members section
        for line in known_section.split("\n"):
            line = line.strip()
            if line.startswith("-"):
                gene_info_match = re.match(r"-\s*([\w\d]+):\s*(.*)", line)
                if gene_info_match:
                    gene, description = gene_info_match.groups()
                    result_dict["known_members"][gene.strip()] = description.strip()

    # Improved regex for potential novel members section
    novel_section_match = re.search(
        r"Potential novel members:(.*?)(?:Summary hypothesis:|$)",
        analysis_text,
        re.DOTALL,
    )
    if novel_section_match:
        novel_section = novel_section_match.group(1).strip()
        # Process each line in the novel members section
        for line in novel_section.split("\n"):
            line = line.strip()
            if line.startswith("-"):
                gene_info_match = re.match(r"-\s*([\w\d]+):\s*(.*)", line)
                if gene_info_match:
                    gene, evidence = gene_info_match.groups()
                    result_dict["novel_members"][gene.strip()] = evidence.strip()

    # Extract summary hypothesis with improved regex
    summary_match = re.search(
        r"Summary hypothesis:(.*?)(?:$)", analysis_text, re.DOTALL
    )
    if summary_match:
        result_dict["summary_hypothesis"] = summary_match.group(1).strip()

    # Special handling for functionally diverse clusters
    if "Functionally diverse cluster" in analysis_text:
        result_dict["dominant_process"] = "Functionally diverse cluster"
        result_dict["confidence"] = "Low"

    return result_dict


def process_batch_cluster_analysis(analysis_text):
    """
    Process batch cluster analysis by splitting the response into individual cluster analyses.

    Args:
        analysis_text: Raw text response containing multiple cluster analyses

    Returns:
        clusters_dict: Dictionary mapping cluster IDs to their analysis results
    """
    clusters_dict = {}

    # Split the text into individual cluster analyses using a more robust pattern
    # This handles different ways the model might format multiple cluster outputs
    cluster_blocks = re.split(r"\n+Cluster ID:", analysis_text)

    # Process each cluster block
    for i, block in enumerate(cluster_blocks):
        if i == 0 and not block.strip().startswith("Cluster ID:"):
            # If first block doesn't start with "Cluster ID:",
            # it might be an intro or header - check if it contains a cluster ID
            if "Cluster ID:" in block:
                block = block[block.find("Cluster ID:") :]
            else:
                continue  # Skip this block if no cluster ID found

        # Reconstruct the cluster ID prefix if needed (except for first block that already has it)
        if i > 0 or not block.strip().startswith("Cluster ID:"):
            block = "Cluster ID:" + block

        # Process this individual cluster analysis
        result = process_cluster_analysis(block)
        cluster_id = result["cluster_id"]

        if cluster_id:
            clusters_dict[cluster_id] = result

    return clusters_dict


def save_progress(df, analysis_dict, out_file_base):
    """
    Save current progress to both TSV and JSON files.

    Args:
        df: DataFrame with current analysis results
        analysis_dict: Dictionary with full raw responses
        out_file_base: Base filename for output files (without extension)
    """
    # Save dataframe to TSV
    tsv_path = f"{out_file_base}.tsv"
    df.to_csv(tsv_path, sep="\t", index=True)

    # Save raw responses to JSON
    json_path = f"{out_file_base}.json"
    with open(json_path, "w") as f:
        json.dump(analysis_dict, f, indent=2)

    # Log the save
    logging.info(f"Progress saved to {tsv_path} and {json_path}")


def save_cluster_analysis(clusters_dict, out_file_base, include_raw=True):
    """
    Save cluster analysis results to JSON and summary CSV.

    Args:
        clusters_dict: Dictionary with cluster analysis results
        out_file_base: Base filename for output files (without extension)
        include_raw: Whether to include raw text in JSON output
    """
    # Set paths
    json_path = f"{out_file_base}_clusters.json"

    # Check if the JSON file already exists and load previous results
    existing_clusters = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                existing_clusters = json.load(f)
            logging.info(
                f"Loaded {len(existing_clusters)} existing clusters from {json_path}"
            )
        except Exception as e:
            logging.warning(f"Failed to load existing clusters file: {e}")

    # Merge existing clusters with new ones
    combined_clusters = {**existing_clusters, **clusters_dict}

    # Option to exclude raw text to save space
    if not include_raw:
        for cluster_id in combined_clusters:
            if "raw_text" in combined_clusters[cluster_id]:
                combined_clusters[cluster_id].pop("raw_text", None)

    # Add metadata
    output_data = {
        "metadata": {
            "timestamp": time.time(),
            "date": datetime.datetime.now().isoformat(),
            "cluster_count": len(combined_clusters),
        },
        "clusters": combined_clusters,
    }

    # Save full results to JSON
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Create summary DataFrame with additional stats
    summary_data = []
    for cluster_id, analysis in combined_clusters.items():
        summary_row = {
            "cluster_id": cluster_id,
            "dominant_process": analysis.get("dominant_process", "Unknown"),
            "confidence": analysis.get("confidence", "Unknown"),
            "known_members_count": len(analysis.get("known_members", {})),
            "novel_members_count": len(analysis.get("novel_members", {})),
            "known_members": "; ".join(analysis.get("known_members", {}).keys()),
            "novel_members": "; ".join(analysis.get("novel_members", {}).keys()),
            "summary_hypothesis": analysis.get("summary_hypothesis", "None provided"),
        }
        summary_data.append(summary_row)

    # Convert to DataFrame and save
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        csv_path = f"{out_file_base}_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        logging.info(
            f"Cluster analysis saved to {json_path} and {csv_path} with {len(combined_clusters)} total clusters"
        )
    else:
        logging.warning("No cluster data to save to summary CSV")
        logging.info(f"Empty cluster analysis saved to {json_path}")


def log_parsing_issues(analysis_text, logger=None):
    """
    Check for potential parsing issues in cluster analysis text.

    Args:
        analysis_text: Raw text response from LLM
        logger: Optional logger to log issues

    Returns:
        issues: List of potential issues found
    """
    issues = []

    # Check for expected sections
    expected_sections = [
        ("Cluster ID:", "Missing cluster ID"),
        ("Dominant Process Name:", "Missing dominant process"),
        ("LLM confidence:", "Missing confidence level"),
        ("Known pathway members:", "Missing known members section"),
        ("Potential novel members:", "Missing novel members section"),
        ("Summary hypothesis:", "Missing summary hypothesis"),
    ]

    for section, issue in expected_sections:
        if section not in analysis_text:
            issues.append(issue)
            if logger:
                logger.warning(f"Parsing issue: {issue}")

    return issues
