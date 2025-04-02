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
    Save cluster analysis results to JSON and multiple CSV formats.

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

    # Create summary DataFrame with focus on novel genes
    summary_data = []
    for cluster_id, analysis in combined_clusters.items():
        # Get basic cluster info
        pathway_confidence = analysis.get("pathway_confidence", "Unknown")
        dominant_process = analysis.get("dominant_process", "Unknown")

        # Create base entry for this cluster
        base_entry = {
            "cluster_id": cluster_id,
            "dominant_process": dominant_process,
            "pathway_confidence": pathway_confidence,
            "established_gene_count": len(analysis.get("established_genes", [])),
            "characterized_gene_count": len(analysis.get("characterized_genes", [])),
            "novel_gene_count": len(analysis.get("novel_genes", [])),
            "summary": analysis.get("summary", "None provided"),
            "established_genes": ";".join(analysis.get("established_genes", [])),
            "characterized_genes": ";".join(analysis.get("characterized_genes", [])),
        }

        # If there are no novel genes, add just the base entry
        if not analysis.get("novel_genes"):
            summary_data.append(base_entry)
            continue

        # For clusters with novel genes, create expanded entries
        # First entry contains all the base information
        novel_genes_list = []
        novel_gene_priorities = []

        for novel_gene in analysis.get("novel_genes", []):
            gene_name = novel_gene.get("gene", "Unknown")
            priority = novel_gene.get("priority", 0)
            rationale = novel_gene.get("rationale", "")

            novel_genes_list.append(f"{gene_name}:{priority}")
            novel_gene_priorities.append(priority)

            # Add individual novel gene entries for detailed view
            gene_entry = base_entry.copy()
            gene_entry["novel_gene"] = gene_name
            gene_entry["priority"] = priority
            gene_entry["rationale"] = rationale

            summary_data.append(gene_entry)

        # Update the base entry with novel gene info and add it
        base_entry["novel_genes"] = ";".join(novel_genes_list)
        base_entry["max_priority"] = (
            max(novel_gene_priorities) if novel_gene_priorities else 0
        )
        base_entry["avg_priority"] = (
            sum(novel_gene_priorities) / len(novel_gene_priorities)
            if novel_gene_priorities
            else 0
        )

        # Add base entry (will be duplicate of some info but useful for summary views)
        summary_data.append(base_entry)

    # Convert to DataFrame and save
    if summary_data:
        # Convert to DataFrame
        summary_df = pd.DataFrame(summary_data)

        # Save complete detailed view
        csv_path = f"{out_file_base}_summary.csv"
        summary_df.to_csv(csv_path, index=False)

        # Create and save a prioritized view focusing on novel genes
        try:
            # Extract rows with novel gene information
            novel_gene_df = summary_df.dropna(subset=["novel_gene"]).copy()

            # Sort by pathway confidence (High > Medium > Low) and priority score (descending)
            confidence_order = {"High": 3, "Medium": 2, "Low": 1, "Unknown": 0}
            novel_gene_df["confidence_score"] = novel_gene_df[
                "pathway_confidence"
            ].apply(
                lambda x: confidence_order.get(
                    x.split()[0] if isinstance(x, str) else "Unknown", 0
                )
            )

            # Sort by confidence and priority
            novel_gene_df = novel_gene_df.sort_values(
                ["confidence_score", "priority"], ascending=[False, False]
            )

            # Select and reorder columns for prioritized view
            prioritized_columns = [
                "cluster_id",
                "novel_gene",
                "priority",
                "rationale",
                "dominant_process",
                "pathway_confidence",
                "summary",
            ]
            prioritized_df = novel_gene_df[prioritized_columns]

            # Save prioritized view
            prioritized_path = f"{out_file_base}_prioritized.csv"
            prioritized_df.to_csv(prioritized_path, index=False)

            logging.info(f"Saved prioritized novel genes to {prioritized_path}")
        except Exception as e:
            logging.warning(f"Failed to create prioritized view: {e}")

        # Create and save cluster-level analysis table
        try:
            # Create a DataFrame with one row per cluster
            cluster_data = []

            for cluster_id, analysis in combined_clusters.items():
                pathway_confidence = analysis.get("pathway_confidence", "Unknown")
                dominant_process = analysis.get("dominant_process", "Unknown")

                # Count genes by category
                established_count = len(analysis.get("established_genes", []))
                characterized_count = len(analysis.get("characterized_genes", []))
                novel_count = len(analysis.get("novel_genes", []))
                total_count = established_count + characterized_count + novel_count

                # Calculate pathway confidence score
                confidence_order = {"High": 3, "Medium": 2, "Low": 1, "Unknown": 0}
                confidence_score = confidence_order.get(
                    pathway_confidence.split()[0]
                    if isinstance(pathway_confidence, str)
                    else "Unknown",
                    0,
                )

                # Get novel gene statistics
                novel_genes = []
                novel_gene_priorities = []
                high_priority_genes = []

                for novel_gene in analysis.get("novel_genes", []):
                    gene_name = novel_gene.get("gene", "Unknown")
                    priority = novel_gene.get("priority", 0)

                    novel_genes.append(gene_name)
                    novel_gene_priorities.append(priority)

                    if priority >= 8:
                        high_priority_genes.append(f"{gene_name}:{priority}")

                # Calculate cluster priority score based on:
                # 1. Pathway confidence (3 for High, 2 for Medium, 1 for Low)
                # 2. Number of high-priority novel genes (priority >= 8)
                # 3. Average priority of novel genes
                max_priority = (
                    max(novel_gene_priorities) if novel_gene_priorities else 0
                )
                avg_priority = (
                    sum(novel_gene_priorities) / len(novel_gene_priorities)
                    if novel_gene_priorities
                    else 0
                )
                high_priority_count = len(high_priority_genes)

                # Combined score formula: confidence_score * (1 + high_priority_count/10) * max_priority/10
                # This weights clusters by confidence first, then by novel gene potential
                cluster_priority_score = (
                    confidence_score
                    * (1 + high_priority_count / 10)
                    * (max_priority / 10)
                )

                # Create cluster entry
                cluster_entry = {
                    "cluster_id": cluster_id,
                    "dominant_process": dominant_process,
                    "pathway_confidence": pathway_confidence,
                    "established_count": established_count,
                    "characterized_count": characterized_count,
                    "novel_count": novel_count,
                    "total_genes": total_count,
                    "max_novel_priority": max_priority,
                    "avg_novel_priority": round(avg_priority, 2),
                    "high_priority_count": high_priority_count,
                    "high_priority_genes": ";".join(high_priority_genes),
                    "cluster_priority_score": round(cluster_priority_score, 2),
                    "summary": analysis.get("summary", "None provided"),
                }

                cluster_data.append(cluster_entry)

            # Convert to DataFrame and sort by cluster priority score
            cluster_df = pd.DataFrame(cluster_data)
            cluster_df = cluster_df.sort_values(
                "cluster_priority_score", ascending=False
            )

            # Save cluster analysis
            cluster_path = f"{out_file_base}_clusters_ranked.csv"
            cluster_df.to_csv(cluster_path, index=False)

            logging.info(f"Saved ranked cluster analysis to {cluster_path}")
        except Exception as e:
            logging.warning(f"Failed to create cluster analysis table: {e}")

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
