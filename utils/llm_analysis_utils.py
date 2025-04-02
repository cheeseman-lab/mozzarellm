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
    Expects a JSON formatted response.

    Args:
        analysis_text: Raw text response from LLM with JSON content

    Returns:
        result_dict: Dictionary with structured analysis
    """
    # Initialize default result dictionary
    result_dict = {
        "cluster_id": None,
        "dominant_process": "Unknown",
        "pathway_confidence": "Low",
        "genes": [],
        "summary": "",
        "raw_text": analysis_text,
    }

    try:
        # Try to extract JSON from the text
        # First, find JSON object in the text (in case there's other text around it)
        json_start = analysis_text.find("{")
        json_end = analysis_text.rfind("}") + 1

        if json_start >= 0 and json_end > json_start:
            json_text = analysis_text[json_start:json_end]
            # Parse the JSON
            analysis_json = json.loads(json_text)

            # Extract the structured data
            result_dict.update(analysis_json)

            # Ensure cluster_id is a string
            if "cluster_id" in result_dict:
                result_dict["cluster_id"] = str(result_dict["cluster_id"])

            return result_dict
        else:
            logging.warning("No JSON object found in analysis text")
            return result_dict

    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from analysis: {e}")
        logging.debug(f"Problematic text: {analysis_text}")
        return result_dict
    except Exception as e:
        logging.error(f"Error processing cluster analysis: {e}")
        return result_dict


def process_batch_cluster_analysis(analysis_text):
    """
    Process batch cluster analysis by extracting results from a JSON array.

    Args:
        analysis_text: Raw text response containing a JSON array of cluster analyses

    Returns:
        clusters_dict: Dictionary mapping cluster IDs to their analysis results
    """
    clusters_dict = {}

    try:
        # Try to extract JSON array from the text
        # First, find JSON array in the text (in case there's other text around it)
        json_start = analysis_text.find("[")
        json_end = analysis_text.rfind("]") + 1

        if json_start >= 0 and json_end > json_start:
            json_text = analysis_text[json_start:json_end]
            # Parse the JSON array
            analysis_array = json.loads(json_text)

            # Process each cluster in the array
            for cluster_analysis in analysis_array:
                # Ensure each cluster has required fields
                if "cluster_id" in cluster_analysis:
                    cluster_id = str(cluster_analysis["cluster_id"])
                    # Add raw text to each cluster
                    cluster_analysis["raw_text"] = analysis_text
                    clusters_dict[cluster_id] = cluster_analysis

            return clusters_dict

        # If no array found, try to find a single JSON object
        json_start = analysis_text.find("{")
        json_end = analysis_text.rfind("}") + 1

        if json_start >= 0 and json_end > json_start:
            json_text = analysis_text[json_start:json_end]
            # Parse the JSON
            analysis_json = json.loads(json_text)

            # Check if this is a single cluster
            if "cluster_id" in analysis_json:
                cluster_id = str(analysis_json["cluster_id"])
                # Add raw text
                analysis_json["raw_text"] = analysis_text
                clusters_dict[cluster_id] = analysis_json

            return clusters_dict

        logging.warning("No JSON array or object found in batch analysis text")
        return clusters_dict

    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from batch analysis: {e}")
        logging.debug(f"Problematic text: {analysis_text}")
        return clusters_dict
    except Exception as e:
        logging.error(f"Error processing batch cluster analysis: {e}")
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
        clusters_dict: Dictionary with cluster analysis results in JSON format
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
                existing_data = json.load(f)
                if "clusters" in existing_data:
                    existing_clusters = existing_data["clusters"]
                else:
                    existing_clusters = existing_data
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

    # Create summary DataFrame with gene details
    summary_data = []
    for cluster_id, analysis in combined_clusters.items():
        # Basic cluster info
        cluster_info = {
            "cluster_id": cluster_id,
            "dominant_process": analysis.get("dominant_process", "Unknown"),
            "pathway_confidence": analysis.get("pathway_confidence", "Unknown"),
            "gene_count": len(analysis.get("genes", [])),
            "summary": analysis.get("summary", "None provided"),
        }

        # Count genes by classification and priority
        if "genes" in analysis and analysis["genes"]:
            classifications = {
                "ESTABLISHED": 0,
                "CHARACTERIZED": 0,
                "UNCHARACTERIZED": 0,
            }
            high_priority_genes = []

            for gene_info in analysis["genes"]:
                # Count by classification
                classification = gene_info.get("classification", "Unknown")
                if classification in classifications:
                    classifications[classification] += 1

                # Track high priority genes (priority >= 8)
                priority = gene_info.get("follow_up_priority", 0)
                if isinstance(priority, (int, float)) and priority >= 8:
                    high_priority_genes.append(gene_info.get("gene", "Unknown"))

            # Add classification counts to cluster info
            for classification, count in classifications.items():
                cluster_info[f"{classification.lower()}_count"] = count

            # Add high priority genes
            cluster_info["high_priority_genes"] = "; ".join(high_priority_genes)
            cluster_info["high_priority_count"] = len(high_priority_genes)

        summary_data.append(cluster_info)

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
