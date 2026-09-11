import datetime
import json
import logging
import os
import re
import time

import pandas as pd

# Version of the <screen>_clusters.json structure (metadata + clusters{...});
# bump on any breaking change to the keys downstream readers consume.
CLUSTERS_JSON_SCHEMA_VERSION = "1"


def extract_json_from_markdown(text):
    """
    Extracts JSON from text that might be wrapped in markdown code blocks.

    Args:
        text: Raw text that might contain JSON in markdown code blocks

    Returns:
        Extracted JSON string or the original text if no code blocks found
    """
    import re

    # Look for JSON in code blocks (with or without language specifier)
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    matches = re.findall(code_block_pattern, text)

    if matches:
        # Return the largest code block (most likely to be the complete JSON)
        return max(matches, key=len).strip()

    # If no code blocks found, return the original text
    return text


def process_cluster_response(analysis_text):
    """
    Process single cluster analysis output from an LLM.

    Raw response text is persisted by the caller via
    `mozzarellm.utils.trace.save_trace()` — this function returns only the
    parsed/standardized structure.

    Args:
        analysis_text: Raw text response from LLM

    Returns:
        Dictionary with structured analysis for a single cluster
    """
    # Default structure for a single cluster
    default_structure = {
        "cluster_id": None,
        "dominant_process": "Unknown",
        "pathway_confidence": "Low",
        "established_genes": [],
        "uncharacterized_genes": [],
        "novel_role_genes": [],
        "summary": "",
    }

    # First, extract JSON from markdown code blocks if present
    cleaned_text = extract_json_from_markdown(analysis_text)

    try:
        # Try direct JSON parsing first
        try:
            parsed_json = json.loads(cleaned_text)
            return _standardize_cluster_format(parsed_json, analysis_text)

        except json.JSONDecodeError:
            # If direct parsing fails, try more robust methods
            logging.info("Direct JSON parsing failed, trying regex extraction...")

            # Clean up common JSON formatting issues
            cleaned_text = re.sub(r",(\s*[}\]])", r"\1", cleaned_text)  # Remove trailing commas
            cleaned_text = re.sub(
                r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)", r'\1"\2"\3', cleaned_text
            )  # Quote unquoted keys

            # Try to extract JSON using regex patterns
            # Look for a JSON object pattern
            object_pattern = r'\{\s*"cluster_id".*\}'
            json_match = re.search(object_pattern, cleaned_text, re.DOTALL)

            if json_match:
                try:
                    json_str = json_match.group(0)
                    analysis_json = json.loads(json_str)
                    return _standardize_cluster_format(analysis_json, analysis_text)
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse JSON object from regex match: {e}")

            # If regex fails, try another approach - find a complete JSON object
            start_idx = cleaned_text.find("{")
            end_idx = cleaned_text.rfind("}")

            if start_idx >= 0 and end_idx > start_idx:
                try:
                    json_str = cleaned_text[start_idx : end_idx + 1]
                    analysis_json = json.loads(json_str)
                    return _standardize_cluster_format(analysis_json, analysis_text)
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse JSON object using indices: {e}")

        # Last resort - try a completely different approach for particularly problematic responses
        logging.warning("All standard parsing methods failed, attempting final recovery approach")

        # Find all key-value pairs using regex and reconstruct JSON
        try:
            reconstructed_json = {}

            # Extract cluster_id
            cluster_id_match = re.search(r'"cluster_id"\s*:\s*"([^"]+)"', cleaned_text)
            if cluster_id_match:
                reconstructed_json["cluster_id"] = cluster_id_match.group(1)

            # Extract dominant_process
            process_match = re.search(r'"dominant_process"\s*:\s*"([^"]+)"', cleaned_text)
            if process_match:
                reconstructed_json["dominant_process"] = process_match.group(1)

            # Extract pathway_confidence
            confidence_match = re.search(r'"pathway_confidence"\s*:\s*"([^"]+)"', cleaned_text)
            if confidence_match:
                reconstructed_json["pathway_confidence"] = confidence_match.group(1)

            # Extract summary
            summary_match = re.search(r'"summary"\s*:\s*"([^"]+)"', cleaned_text)
            if summary_match:
                reconstructed_json["summary"] = summary_match.group(1)

            if reconstructed_json.get("cluster_id"):
                return _standardize_cluster_format(reconstructed_json, analysis_text)
        except Exception as e:
            logging.error(f"Final recovery approach failed: {e}")

        # If all approaches fail, return the default
        return default_structure

    except Exception as e:
        logging.error(f"Error processing cluster analysis: {e}")
        return default_structure


def _standardize_cluster_format(cluster_data, raw_text):
    """Helper function to standardize cluster data format"""
    # Start with default structure
    standardized = {
        "cluster_id": None,
        "dominant_process": "Unknown",
        "pathway_confidence": "Low",
        "established_genes": [],
        "uncharacterized_genes": [],
        "novel_role_genes": [],
        "summary": "",
    }

    # Update with provided data
    standardized.update(cluster_data)

    # Process cluster_id to ensure it's a string
    if "cluster_id" in cluster_data:
        cluster_id_raw = cluster_data["cluster_id"]

        # Handle cases like "Cluster 0"
        if isinstance(cluster_id_raw, str) and "cluster" in cluster_id_raw.lower():
            digit_match = re.search(r"\d+", cluster_id_raw)
            if digit_match:
                standardized["cluster_id"] = digit_match.group(0)
            else:
                standardized["cluster_id"] = str(cluster_id_raw)
        else:
            standardized["cluster_id"] = str(cluster_id_raw)

    # Handle legacy format conversions
    if "novel_genes" in cluster_data and not standardized.get("uncharacterized_genes"):
        standardized["uncharacterized_genes"] = cluster_data.pop("novel_genes")

    if "characterized_genes" in cluster_data and not standardized.get("novel_role_genes"):
        characterized = cluster_data.pop("characterized_genes")
        if isinstance(characterized, list):
            if characterized and isinstance(characterized[0], str):
                standardized["novel_role_genes"] = [
                    {
                        "gene": gene,
                        "priority": 5,
                        "rationale": "Characterized gene with potential novel pathway role",
                    }
                    for gene in characterized
                ]
            else:
                standardized["novel_role_genes"] = characterized

    # Ensure gene categories are properly structured
    for category in ["uncharacterized_genes", "novel_role_genes"]:
        if standardized[category] and isinstance(standardized[category][0], str):
            standardized[category] = [
                {
                    "gene": gene,
                    "priority": 5,
                    "rationale": f"Default rationale for {category}",
                }
                for gene in standardized[category]
            ]

    return standardized


def _gene_rows(cluster_id, analysis):
    """One row per classified gene: category, evidence-ladder subclass, rationale."""
    process = analysis.get("dominant_process", "")
    confidence = analysis.get("pathway_confidence", "")
    rows = []
    for gene in analysis.get("established_genes", []):
        rows.append(
            {
                "gene": gene,
                "cluster_id": cluster_id,
                "category": "ESTABLISHED",
                "subclass": "",
                "rationale": "",
                "evidence": "",
                "dominant_process": process,
                "pathway_confidence": confidence,
            }
        )
    for category, key in (
        ("NOVEL_ROLE", "novel_role_genes"),
        ("UNCHARACTERIZED", "uncharacterized_genes"),
    ):
        for info in analysis.get(key, []) or []:
            rows.append(
                {
                    "gene": info.get("gene", ""),
                    "cluster_id": cluster_id,
                    "category": category,
                    "subclass": info.get("class", ""),
                    "rationale": info.get("rationale", ""),
                    "evidence": info.get("evidence", ""),
                    "dominant_process": process,
                    "pathway_confidence": confidence,
                }
            )
    return rows


def _cluster_row(cluster_id, analysis):
    """One row per cluster: pathway call, per-category genes/counts, coverage."""
    established = analysis.get("established_genes", []) or []
    novel = [g.get("gene", "") for g in analysis.get("novel_role_genes", []) or []]
    unchar = [g.get("gene", "") for g in analysis.get("uncharacterized_genes", []) or []]
    classified = len(established) + len(novel) + len(unchar)
    missed = analysis.get("missed_genes", []) or []
    total = analysis.get("total_genes_in_cluster", classified)
    return {
        "cluster_id": cluster_id,
        "dominant_process": analysis.get("dominant_process", ""),
        "pathway_confidence": analysis.get("pathway_confidence", ""),
        "summary": analysis.get("summary", ""),
        "n_genes": total,
        "n_classified": classified,
        "n_established": len(established),
        "n_novel_role": len(novel),
        "n_uncharacterized": len(unchar),
        "established_genes": ";".join(established),
        "novel_role_genes": ";".join(novel),
        "uncharacterized_genes": ";".join(unchar),
        "missed_genes": ";".join(missed),
        "classification_completeness": round(
            analysis.get("classification_completeness", 1.0), 3
        ),
    }


def save_cluster_analysis(
    clusters_dict, out_file_base=None, original_df=None, include_raw=True, save_outputs=True
):
    """
    Process and optionally save cluster analysis results to JSON and multiple CSV formats.
    Returns the processed DataFrames regardless of whether they're saved to disk.

    Args:
        clusters_dict: Dictionary with cluster analysis results in JSON format
        out_file_base: Base filename for output files (without extension), required if save_outputs=True
        original_df: Optional original DataFrame with cluster_id and other original data
        include_raw: Whether to include raw text in JSON output
        save_outputs: Whether to write results to disk (default: True)

    Returns:
        dict: Dictionary containing the following keys:
            - 'json_data': The complete JSON data structure
            - 'gene_df': DataFrame with gene-level analysis
            - 'cluster_df': DataFrame with cluster-level analysis
    """
    # Initialize return dictionary
    results = {
        "json_data": None,
        "gene_df": pd.DataFrame(
            columns=[
                "gene", "cluster_id", "category", "subclass", "rationale",
                "evidence", "dominant_process", "pathway_confidence",
            ]
        ),
        "cluster_df": pd.DataFrame(columns=["cluster_id"]),
    }

    # Validate parameters
    if save_outputs and not out_file_base:
        logging.warning("Cannot save outputs without out_file_base parameter")
        save_outputs = False

    # Set paths if saving
    json_path = f"{out_file_base}_clusters.json" if out_file_base else None

    # Check if the JSON file already exists and load previous results
    existing_clusters = {}
    if save_outputs and os.path.exists(json_path):
        try:
            with open(json_path, encoding="utf-8") as f:
                existing_data = json.load(f)
                if "clusters" in existing_data:
                    existing_clusters = existing_data["clusters"]
                else:
                    existing_clusters = existing_data
            logging.info(f"Loaded {len(existing_clusters)} existing clusters from {json_path}")
        except Exception as e:
            logging.warning(f"Failed to load existing clusters file: {e}")

    # Merge existing clusters with new ones
    combined_clusters = {**existing_clusters, **clusters_dict}

    # Option to exclude raw text to save space
    processed_clusters = combined_clusters.copy()
    if not include_raw:
        for cluster_id in processed_clusters:
            if "raw_text" in processed_clusters[cluster_id]:
                processed_clusters[cluster_id].pop("raw_text", None)

    # Add metadata
    output_data = {
        "metadata": {
            "schema_version": CLUSTERS_JSON_SCHEMA_VERSION,
            "timestamp": time.time(),
            "date": datetime.datetime.now().isoformat(),
            "cluster_count": len(processed_clusters),
        },
        "clusters": processed_clusters,
    }

    # Store the JSON data in the results
    results["json_data"] = output_data

    # Save full results to JSON if requested
    if save_outputs and json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Gene-level and cluster-level tables (the user-facing view of the run).
    if combined_clusters:
        gene_rows, cluster_rows = [], []
        for cluster_id, analysis in combined_clusters.items():
            gene_rows.extend(_gene_rows(cluster_id, analysis))
            cluster_rows.append(_cluster_row(cluster_id, analysis))

        gene_columns = [
            "gene", "cluster_id", "category", "subclass", "rationale",
            "evidence", "dominant_process", "pathway_confidence",
        ]
        gene_df = pd.DataFrame(gene_rows, columns=gene_columns)
        cluster_df = pd.DataFrame(cluster_rows)

        # Merge caller-provided per-cluster columns (e.g. the input table's metadata).
        if original_df is not None:
            original = original_df.copy()
            original["cluster_id"] = original["cluster_id"].astype(str)
            for df in (gene_df, cluster_df):
                df["cluster_id"] = df["cluster_id"].astype(str)
            extra = [c for c in original.columns if c != "cluster_id"]
            gene_extra = [c for c in extra if c not in gene_df.columns]
            cluster_extra = [c for c in extra if c not in cluster_df.columns]
            if gene_extra:
                gene_df = gene_df.merge(
                    original[["cluster_id"] + gene_extra], on="cluster_id", how="left"
                )
            if cluster_extra:
                cluster_df = cluster_df.merge(
                    original[["cluster_id"] + cluster_extra], on="cluster_id", how="left"
                )

        cluster_df = (
            cluster_df.assign(_sort=pd.to_numeric(cluster_df["cluster_id"], errors="coerce"))
            .sort_values(["_sort", "cluster_id"], na_position="last", kind="stable")
            .drop(columns="_sort")
            .reset_index(drop=True)
        )
        gene_df = gene_df.sort_values(["cluster_id", "category", "gene"]).reset_index(drop=True)

        results["gene_df"] = gene_df
        results["cluster_df"] = cluster_df

        if save_outputs and out_file_base:
            gene_df.to_csv(f"{out_file_base}_genes.csv", index=False)
            cluster_df.to_csv(f"{out_file_base}_clusters.csv", index=False)
            logging.info(
                f"Saved {len(gene_df)} gene rows and {len(cluster_df)} cluster rows "
                f"to {out_file_base}_genes.csv / _clusters.csv"
            )
    else:
        logging.warning("No cluster data to save")

    return results
