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


def process_cluster_response(analysis_text, is_batch=False):
    """
    Unified function to process cluster analysis output from an LLM.
    Handles both single cluster and batch responses.
    
    Args:
        analysis_text: Raw text response from LLM
        is_batch: Boolean indicating if this is a batch response
        
    Returns:
        If is_batch: Dictionary mapping cluster IDs to their analysis results
        If not is_batch: Dictionary with structured analysis for a single cluster
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
        "raw_text": analysis_text,
    }
    
    try:
        # Find and extract JSON content
        json_array_start = analysis_text.find("[")
        json_object_start = analysis_text.find("{")
        
        if is_batch and json_array_start >= 0:
            # Process batch response (JSON array)
            json_array_end = analysis_text.rfind("]") + 1
            json_text = analysis_text[json_array_start:json_array_end]
            clusters = {}
            
            try:
                analysis_array = json.loads(json_text)
                # Process each cluster in the array
                for cluster_analysis in analysis_array:
                    processed_cluster = _standardize_cluster_format(cluster_analysis, analysis_text)
                    if processed_cluster["cluster_id"]:
                        clusters[processed_cluster["cluster_id"]] = processed_cluster
                return clusters
            except json.JSONDecodeError:
                logging.error("Failed to parse JSON array from batch analysis")
                return {}
                
        elif json_object_start >= 0:
            # Process single cluster response (JSON object)
            json_object_end = analysis_text.rfind("}") + 1
            json_text = analysis_text[json_object_start:json_object_end]
            
            try:
                analysis_json = json.loads(json_text)
                if is_batch and "cluster_id" in analysis_json:
                    # Single cluster in batch mode
                    processed_cluster = _standardize_cluster_format(analysis_json, analysis_text)
                    return {processed_cluster["cluster_id"]: processed_cluster}
                else:
                    # Standard single cluster mode
                    return _standardize_cluster_format(analysis_json, analysis_text)
            except json.JSONDecodeError:
                logging.error("Failed to parse JSON object from analysis")
                return default_structure if not is_batch else {}
        else:
            logging.warning("No JSON content found in analysis text")
            return default_structure if not is_batch else {}
            
    except Exception as e:
        logging.error(f"Error processing cluster analysis: {e}")
        return default_structure if not is_batch else {}

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
        "raw_text": raw_text,
    }
    
    # Update with provided data
    standardized.update(cluster_data)
    
    # Process cluster_id to ensure it's a string
    if "cluster_id" in cluster_data:
        cluster_id_raw = cluster_data["cluster_id"]
        
        # Handle cases like "Cluster 0"
        if isinstance(cluster_id_raw, str) and "cluster" in cluster_id_raw.lower():
            digit_match = re.search(r'\d+', cluster_id_raw)
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
                    {"gene": gene, "priority": 5, "rationale": "Characterized gene with potential novel pathway role"}
                    for gene in characterized
                ]
            else:
                standardized["novel_role_genes"] = characterized
    
    # Ensure gene categories are properly structured
    for category in ["uncharacterized_genes", "novel_role_genes"]:
        if standardized[category] and isinstance(standardized[category][0], str):
            standardized[category] = [
                {"gene": gene, "priority": 5, "rationale": f"Default rationale for {category}"}
                for gene in standardized[category]
            ]
    
    return standardized


def save_cluster_analysis(clusters_dict, out_file_base, original_df=None, include_raw=True):
    """
    Save cluster analysis results to JSON and multiple CSV formats,
    with an option to merge with original data.
    Updated to handle the new gene categories structure.

    Args:
        clusters_dict: Dictionary with cluster analysis results in JSON format
        out_file_base: Base filename for output files (without extension)
        original_df: Optional original DataFrame with cluster_id and other original data
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

    # Create gene-level and cluster-level tables
    if combined_clusters:
        # Create gene-level tables - one for uncharacterized genes and one for novel role genes
        try:
            # Create DataFrames with one row per gene type
            uncharacterized_gene_data = []
            novel_role_gene_data = []
            
            for cluster_id, analysis in combined_clusters.items():
                pathway_confidence = analysis.get("pathway_confidence", "Unknown")
                cluster_score = 0
                summary = analysis.get("summary", "None provided")
                
                # Calculate cluster score
                confidence_order = {"High": 3, "Medium": 2, "Low": 1, "Unknown": 0}
                confidence_score = confidence_order.get(
                    pathway_confidence.split()[0]
                    if isinstance(pathway_confidence, str)
                    else "Unknown",
                    0,
                )
                
                # Get established genes
                established_genes = analysis.get("established_genes", [])
                
                # Get uncharacterized gene statistics
                unchar_priorities = []
                high_unchar_count = 0
                
                for unchar_gene in analysis.get("uncharacterized_genes", []):
                    if unchar_gene.get("priority", 0) >= 8:
                        high_unchar_count += 1
                    unchar_priorities.append(unchar_gene.get("priority", 0))
                
                # Get novel role gene statistics
                novel_role_priorities = []
                high_novel_role_count = 0
                
                for novel_role_gene in analysis.get("novel_role_genes", []):
                    if novel_role_gene.get("priority", 0) >= 8:
                        high_novel_role_count += 1
                    novel_role_priorities.append(novel_role_gene.get("priority", 0))
                
                # Get max priorities
                max_unchar_priority = max(unchar_priorities) if unchar_priorities else 0
                max_novel_role_priority = max(novel_role_priorities) if novel_role_priorities else 0
                
                # Calculate cluster score based on both gene types
                unchar_score = confidence_score * (1 + high_unchar_count / 10) * (max_unchar_priority / 10)
                novel_role_score = confidence_score * (1 + high_novel_role_count / 10) * (max_novel_role_priority / 10)
                cluster_score = round(max(unchar_score, novel_role_score), 2)
                
                # Get all uncharacterized genes
                unchar_genes = [gene.get("gene", "Unknown") for gene in analysis.get("uncharacterized_genes", [])]
                
                # Get all novel role genes
                novel_role_genes = [gene.get("gene", "Unknown") for gene in analysis.get("novel_role_genes", [])]
                
                # Create entries for each uncharacterized gene
                for unchar_gene in analysis.get("uncharacterized_genes", []):
                    gene_name = unchar_gene.get("gene", "Unknown")
                    gene_desc = unchar_gene.get("rationale", "")
                    priority = unchar_gene.get("priority", 0)
                    
                    gene_entry = {
                        "gene_name": gene_name,
                        "gene_description": gene_desc,
                        "gene_importance_score": priority,                 
                        "cluster_id": cluster_id,
                        "cluster_biological_process": analysis.get("dominant_process", "Unknown"), 
                        "pathway_confidence_level": pathway_confidence,    
                        "cluster_importance_score": cluster_score,
                        "follow_up_suggestion": summary,         
                        "established_genes": ";".join(established_genes),
                        "established_gene_count": len(established_genes),  
                        "uncharacterized_genes": ";".join(unchar_genes),
                        "uncharacterized_gene_count": len(unchar_genes),  
                        "novel_role_genes": ";".join(novel_role_genes),
                        "novel_role_gene_count": len(novel_role_genes),
                        "gene_category": "uncharacterized"
                    }
                    
                    uncharacterized_gene_data.append(gene_entry)
                
                # Create entries for each novel role gene
                for novel_role_gene in analysis.get("novel_role_genes", []):
                    gene_name = novel_role_gene.get("gene", "Unknown")
                    gene_desc = novel_role_gene.get("rationale", "")
                    priority = novel_role_gene.get("priority", 0)
                    
                    gene_entry = {
                        "gene_name": gene_name,
                        "gene_description": gene_desc,
                        "gene_importance_score": priority,                 
                        "cluster_id": cluster_id,
                        "cluster_biological_process": analysis.get("dominant_process", "Unknown"), 
                        "pathway_confidence_level": pathway_confidence,    
                        "cluster_importance_score": cluster_score,
                        "follow_up_suggestion": summary,         
                        "established_genes": ";".join(established_genes),
                        "established_gene_count": len(established_genes),  
                        "uncharacterized_genes": ";".join(unchar_genes),
                        "uncharacterized_gene_count": len(unchar_genes),  
                        "novel_role_genes": ";".join(novel_role_genes),
                        "novel_role_gene_count": len(novel_role_genes),
                        "gene_category": "novel_role"
                    }
                    
                    novel_role_gene_data.append(gene_entry)
            
            # Combine all gene data
            all_gene_data = uncharacterized_gene_data + novel_role_gene_data
            
            # Convert to DataFrame and sort by category, then priority
            if all_gene_data:
                gene_df = pd.DataFrame(all_gene_data)
                
                # Merge with original data if provided
                if original_df is not None and not gene_df.empty:
                    # Ensure cluster_id is the same type in both DataFrames
                    gene_df['cluster_id'] = gene_df['cluster_id'].astype(str)
                    original_df_copy = original_df.copy()
                    original_df_copy['cluster_id'] = original_df_copy['cluster_id'].astype(str)
                    
                    # Select only columns from original_df that are not already in gene_df
                    # except for cluster_id which is used for merging
                    original_cols = [col for col in original_df_copy.columns 
                                    if col != 'cluster_id' and col not in gene_df.columns]
                    
                    if original_cols:
                        # Merge the DataFrames
                        gene_df = pd.merge(
                            gene_df,
                            original_df_copy[['cluster_id'] + original_cols],
                            on='cluster_id',
                            how='left'
                        )
                        
                        logging.info(f"Merged gene analysis with {len(original_cols)} columns from original data")
                
                # Sort the data
                gene_df = gene_df.sort_values(["gene_category", "gene_importance_score", "cluster_importance_score"], 
                                             ascending=[True, False, False])
                
                # Save combined gene table
                gene_path = f"{out_file_base}_all_genes.csv"
                gene_df.to_csv(gene_path, index=False)
                
                # Also save separate tables for each gene category
                if uncharacterized_gene_data:
                    unchar_df = gene_df[gene_df['gene_category'] == 'uncharacterized']
                    unchar_path = f"{out_file_base}_uncharacterized_genes.csv"
                    unchar_df.to_csv(unchar_path, index=False)
                    logging.info(f"Saved uncharacterized gene analysis to {unchar_path}")
                
                if novel_role_gene_data:
                    novel_role_df = gene_df[gene_df['gene_category'] == 'novel_role']
                    novel_role_path = f"{out_file_base}_novel_role_genes.csv"
                    novel_role_df.to_csv(novel_role_path, index=False)
                    logging.info(f"Saved novel role gene analysis to {novel_role_path}")
                
                logging.info(f"Saved combined gene analysis to {gene_path}")
            else:
                logging.warning("No gene data to save")
                
        except Exception as e:
            logging.warning(f"Failed to create gene tables: {e}")

        # Create cluster-level analysis table
        try:
            # Create a DataFrame with one row per cluster
            cluster_data = []

            for cluster_id, analysis in combined_clusters.items():
                pathway_confidence = analysis.get("pathway_confidence", "Unknown")
                biological_process = analysis.get("dominant_process", "Unknown")

                # Get all genes by category
                established_genes = analysis.get("established_genes", [])
                uncharacterized_genes_info = analysis.get("uncharacterized_genes", [])
                novel_role_genes_info = analysis.get("novel_role_genes", [])
                
                # Extract gene names
                uncharacterized_genes = [gene.get("gene", "Unknown") for gene in uncharacterized_genes_info]
                novel_role_genes = [gene.get("gene", "Unknown") for gene in novel_role_genes_info]
                
                # Create gene lists
                all_genes = established_genes + uncharacterized_genes + novel_role_genes
                
                # Count genes by category
                established_count = len(established_genes)
                uncharacterized_count = len(uncharacterized_genes)
                novel_role_count = len(novel_role_genes)
                total_count = established_count + uncharacterized_count + novel_role_count

                # Calculate pathway confidence score
                confidence_order = {"High": 3, "Medium": 2, "Low": 1, "Unknown": 0}
                confidence_score = confidence_order.get(
                    pathway_confidence.split()[0]
                    if isinstance(pathway_confidence, str)
                    else "Unknown",
                    0,
                )

                # Get uncharacterized gene statistics
                unchar_priorities = []
                high_unchar_genes = []

                for unchar_gene in uncharacterized_genes_info:
                    gene_name = unchar_gene.get("gene", "Unknown")
                    priority = unchar_gene.get("priority", 0)
                    unchar_priorities.append(priority)

                    if priority >= 8:
                        high_unchar_genes.append(f"{gene_name}:{priority}")

                # Get novel role gene statistics
                novel_role_priorities = []
                high_novel_role_genes = []

                for novel_role_gene in novel_role_genes_info:
                    gene_name = novel_role_gene.get("gene", "Unknown")
                    priority = novel_role_gene.get("priority", 0)
                    novel_role_priorities.append(priority)

                    if priority >= 8:
                        high_novel_role_genes.append(f"{gene_name}:{priority}")

                # Calculate statistics for uncharacterized genes
                max_unchar_priority = max(unchar_priorities) if unchar_priorities else 0
                avg_unchar_priority = (
                    sum(unchar_priorities) / len(unchar_priorities)
                    if unchar_priorities
                    else 0
                )
                high_unchar_count = len(high_unchar_genes)

                # Calculate statistics for novel role genes
                max_novel_role_priority = max(novel_role_priorities) if novel_role_priorities else 0
                avg_novel_role_priority = (
                    sum(novel_role_priorities) / len(novel_role_priorities)
                    if novel_role_priorities
                    else 0
                )
                high_novel_role_count = len(high_novel_role_genes)

                # Combined score formula - take the higher of the two gene type scores
                unchar_score = confidence_score * (1 + high_unchar_count / 10) * (max_unchar_priority / 10)
                novel_role_score = confidence_score * (1 + high_novel_role_count / 10) * (max_novel_role_priority / 10)
                cluster_priority_score = max(unchar_score, novel_role_score)

                # Create cluster entry with added gene information
                cluster_entry = {
                    "cluster_id": cluster_id,
                    "cluster_biological_process": biological_process,
                    "pathway_confidence_level": pathway_confidence,
                    "cluster_importance_score": round(cluster_priority_score, 2),
                    "follow_up_suggestion": analysis.get("summary", "None provided"),
                    "established_genes": ";".join(established_genes),
                    "established_gene_count": established_count,
                    "uncharacterized_genes": ";".join(uncharacterized_genes),
                    "uncharacterized_gene_count": uncharacterized_count,
                    "novel_role_genes": ";".join(novel_role_genes),
                    "novel_role_gene_count": novel_role_count,
                    "total_gene_count": total_count,
                    "highest_unchar_importance": max_unchar_priority,
                    "average_unchar_importance": round(avg_unchar_priority, 2),
                    "high_unchar_genes": ";".join(high_unchar_genes),
                    "high_unchar_gene_count": high_unchar_count,
                    "highest_novel_role_importance": max_novel_role_priority,
                    "average_novel_role_importance": round(avg_novel_role_priority, 2),
                    "high_novel_role_genes": ";".join(high_novel_role_genes),
                    "high_novel_role_gene_count": high_novel_role_count,
                    "all_cluster_genes": ";".join(all_genes),
                }

                cluster_data.append(cluster_entry)

            # Convert to DataFrame
            if cluster_data:
                cluster_df = pd.DataFrame(cluster_data)
                
                # Merge with original data if provided
                if original_df is not None and not cluster_df.empty:
                    # Ensure cluster_id is the same type in both DataFrames
                    cluster_df['cluster_id'] = cluster_df['cluster_id'].astype(str)
                    original_df_copy = original_df.copy()
                    original_df_copy['cluster_id'] = original_df_copy['cluster_id'].astype(str)
                    
                    # Select only columns from original_df that are not already in cluster_df
                    # except for cluster_id which is used for merging
                    original_cols = [col for col in original_df_copy.columns 
                                    if col != 'cluster_id' and col not in cluster_df.columns]
                    
                    if original_cols:
                        # Merge the DataFrames
                        cluster_df = pd.merge(
                            cluster_df,
                            original_df_copy[['cluster_id'] + original_cols],
                            on='cluster_id',
                            how='left'
                        )
                        
                        logging.info(f"Merged cluster analysis with {len(original_cols)} columns from original data")
                
                # Sort by cluster_id, ensuring numeric sorting if possible
                try:
                    # Convert cluster_id to numeric for sorting if possible
                    cluster_df["cluster_id_num"] = pd.to_numeric(cluster_df["cluster_id"], errors="coerce")
                    cluster_df = cluster_df.sort_values("cluster_id_num").drop("cluster_id_num", axis=1)
                except:
                    # Fall back to string sorting if numeric conversion fails
                    cluster_df = cluster_df.sort_values("cluster_id")

                # Save cluster analysis
                cluster_path = f"{out_file_base}_clusters.csv"
                cluster_df.to_csv(cluster_path, index=False)

                logging.info(f"Saved cluster analysis to {cluster_path}")
            else:
                logging.warning("No cluster data to save")
                
        except Exception as e:
            logging.warning(f"Failed to create cluster analysis table: {e}")

        logging.info(
            f"Cluster analysis saved to {json_path} with {len(combined_clusters)} total clusters"
        )
    else:
        logging.warning("No cluster data to save")
        logging.info(f"Empty cluster analysis saved to {json_path}")