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


def calculate_stats(df, score_column):
    """
    Calculate statistics for a column of scores.
    
    Args:
        df: DataFrame containing scores
        score_column: Column name for scores
        
    Returns:
        stats: Dictionary of statistics
    """
    # Filter out invalid scores
    valid_scores = df[df[score_column] > 0][score_column]
    
    stats = {
        "count": len(valid_scores),
        "mean": valid_scores.mean(),
        "median": valid_scores.median(),
        "min": valid_scores.min(),
        "max": valid_scores.max(),
        "high_confidence_count": len(df[df[score_column] > 0.85]),
        "medium_confidence_count": len(df[(df[score_column] > 0.80) & (df[score_column] <= 0.85)]),
        "low_confidence_count": len(df[(df[score_column] > 0) & (df[score_column] <= 0.80)]),
        "unscored_count": len(df[df[score_column] <= 0])
    }
    
    return stats


def compare_models(df, model_columns, base_column=None):
    """
    Compare results between different models.
    
    Args:
        df: DataFrame with analysis results
        model_columns: List of columns to compare (prefix for score columns)
        base_column: Optional base model for comparison
        
    Returns:
        comparison: Dictionary with comparison statistics
    """
    comparison = {}
    
    # Calculate agreement between models
    score_columns = [f"{col} Score" for col in model_columns if f"{col} Score" in df.columns]
    name_columns = [f"{col} Name" for col in model_columns if f"{col} Name" in df.columns]
    
    # Prepare scores for comparison (rows with valid scores in all models)
    valid_rows = df.dropna(subset=score_columns)
    
    # Calculate score correlations if we have at least 2 models and enough data
    if len(score_columns) >= 2 and len(valid_rows) > 5:
        score_correlation = valid_rows[score_columns].corr()
        comparison["score_correlation"] = score_correlation.to_dict()
    
    # Calculate name agreement if we have at least 2 models
    if len(name_columns) >= 2:
        for i, col1 in enumerate(name_columns):
            for col2 in name_columns[i+1:]:
                # Calculate percentage of matching names
                match_count = sum(df[col1] == df[col2])
                total_count = sum((df[col1].notna()) & (df[col2].notna()))
                
                if total_count > 0:
                    agreement = match_count / total_count
                    comparison[f"name_agreement_{col1}_vs_{col2}"] = agreement
    
    # Compare with base model if specified
    if base_column and f"{base_column} Name" in df.columns:
        base_name_col = f"{base_column} Name"
        for col in model_columns:
            if col != base_column and f"{col} Name" in df.columns:
                comparison_col = f"{col} Name"
                
                # Calculate agreement with base model
                match_count = sum(df[base_name_col] == df[comparison_col])
                total_count = sum((df[base_name_col].notna()) & (df[comparison_col].notna()))
                
                if total_count > 0:
                    agreement = match_count / total_count
                    comparison[f"{col}_agreement_with_{base_column}"] = agreement
    
    return comparison