import json
import re
import pandas as pd
import logging

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
    name_match = re.search(r"FUNCTION NAME:?\s*(.*?)(?:\n|$)", analysis_text, re.IGNORECASE)
    if name_match:
        function_name = name_match.group(1).strip()
    
    # Try to extract confidence score
    score_match = re.search(r"CONFIDENCE SCORE:?\s*([\d\.]+)", analysis_text, re.IGNORECASE)
    if score_match:
        confidence_score = score_match.group(1).strip()
    
    # Try to extract detailed analysis
    analysis_match = re.search(r"ANALYSIS:?\s*([\s\S]*?)(?:$|FUNCTION NAME|CONFIDENCE SCORE)", analysis_text, re.IGNORECASE)
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
        "raw_text": analysis_text
    }
    
    # Extract cluster ID
    cluster_match = re.search(r"Cluster ID:?\s*(\w+)", analysis_text)
    if cluster_match:
        result_dict["cluster_id"] = cluster_match.group(1).strip()
    
    # Extract dominant process
    process_match = re.search(r"Dominant Process Name:?\s*(.*?)(?:\n|$)", analysis_text)
    if process_match:
        result_dict["dominant_process"] = process_match.group(1).strip()
    
    # Extract confidence level
    confidence_match = re.search(r"LLM confidence:?\s*(High|Medium|Low)", analysis_text)
    if confidence_match:
        result_dict["confidence"] = confidence_match.group(1).strip()
    
    # Extract known pathway members section
    known_section_match = re.search(r"Known pathway members:(.*?)(?:Potential novel members:|$)", analysis_text, re.DOTALL)
    if known_section_match:
        known_section = known_section_match.group(1).strip()
        # Process each line in the known members section
        for line in known_section.split('\n'):
            line = line.strip()
            if line.startswith('-'):
                gene_info_match = re.match(r'-\s*([\w\d]+):\s*(.*)', line)
                if gene_info_match:
                    gene, description = gene_info_match.groups()
                    result_dict["known_members"][gene.strip()] = description.strip()
    
    # Extract potential novel members section
    novel_section_match = re.search(r"Potential novel members:(.*?)(?:Summary hypothesis:|$)", analysis_text, re.DOTALL)
    if novel_section_match:
        novel_section = novel_section_match.group(1).strip()
        # Process each line in the novel members section
        for line in novel_section.split('\n'):
            line = line.strip()
            if line.startswith('-'):
                gene_info_match = re.match(r'-\s*([\w\d]+):\s*(.*)', line)
                if gene_info_match:
                    gene, evidence = gene_info_match.groups()
                    result_dict["novel_members"][gene.strip()] = evidence.strip()
    
    # Extract summary hypothesis
    summary_match = re.search(r"Summary hypothesis:(.*?)(?:\n\n|$)", analysis_text, re.DOTALL)
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
    
    # Split the text into individual cluster analyses
    cluster_blocks = re.split(r'\n\s*Cluster ID:', analysis_text)
    
    # Process each cluster block
    for i, block in enumerate(cluster_blocks):
        if i == 0 and not block.strip():
            continue  # Skip empty first block if it exists
            
        # Reconstruct the cluster ID prefix if needed (except for first block which might be intro)
        if i > 0:
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
    df.to_csv(tsv_path, sep='\t', index=True)
    
    # Save raw responses to JSON
    json_path = f"{out_file_base}.json"
    with open(json_path, 'w') as f:
        json.dump(analysis_dict, f, indent=2)
    
    # Log the save
    logging.info(f"Progress saved to {tsv_path} and {json_path}")

def save_cluster_analysis(clusters_dict, out_file_base):
    """
    Save cluster analysis results to JSON and summary CSV.
    
    Args:
        clusters_dict: Dictionary with cluster analysis results
        out_file_base: Base filename for output files (without extension)
    """
    # Save full results to JSON
    json_path = f"{out_file_base}_clusters.json"
    with open(json_path, 'w') as f:
        json.dump(clusters_dict, f, indent=2)
    
    # Create summary DataFrame
    summary_data = []
    for cluster_id, analysis in clusters_dict.items():
        summary_row = {
            'cluster_id': cluster_id,
            'dominant_process': analysis['dominant_process'],
            'confidence': analysis['confidence'],
            'known_members_count': len(analysis['known_members']),
            'novel_members_count': len(analysis['novel_members']),
            'known_members': '; '.join(analysis['known_members'].keys()),
            'novel_members': '; '.join(analysis['novel_members'].keys()),
            'summary_hypothesis': analysis['summary_hypothesis']
        }
        summary_data.append(summary_row)
    
    # Convert to DataFrame and save
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        csv_path = f"{out_file_base}_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        logging.info(f"Cluster analysis saved to {json_path} and {csv_path}")
    else:
        logging.warning("No cluster data to save to summary CSV")
        logging.info(f"Empty cluster analysis saved to {json_path}")