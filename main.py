import argparse
import pandas as pd
import json 
import numpy as np
import os
import logging
import re
from tqdm import tqdm
from dotenv import load_dotenv
from collections import defaultdict

# Import utility functions
from utils.openai_query import openai_chat
from utils.anthropic_query import anthropic_chat
from utils.gemini_query import query_genai_model
from utils.perplexity_query import perplexity_chat
from utils.server_model_query import server_model_chat
from utils.prompt_factory import make_user_prompt_with_score, make_cluster_analysis_prompt, make_batch_cluster_prompt
from utils.llm_analysis_utils import (
    process_analysis, 
    process_cluster_analysis, 
    process_batch_cluster_analysis,
    save_progress, 
    save_cluster_analysis
)
from utils.logging_utils import get_model_logger

# Import constants
import constant

# Load environment variables
load_dotenv()

# Add argument parsing
parser = argparse.ArgumentParser(description='Process gene sets or clusters with LLMs.')
parser.add_argument('--config', type=str, required=True, help='Config file for LLM')
parser.add_argument('--mode', type=str, choices=['gene_set', 'cluster'], default='gene_set', 
                    help='Analysis mode: gene_set (original) or cluster (new)')
parser.add_argument('--initialize', action='store_true', help='Initialize the output file with columns')
parser.add_argument('--input', type=str, required=True, help='Path to input csv with gene sets or clusters')
parser.add_argument('--input_sep', type=str, required=True, help='Separator for input csv')
parser.add_argument('--set_index', type=str, default='set_id', help='Column name for gene set/cluster index')
parser.add_argument('--gene_column', type=str, required=True, help='Column name for gene set')
parser.add_argument('--gene_sep', type=str, required=True, help='Separator for gene set')
parser.add_argument('--batch_size', type=int, default=1, help='Number of clusters to analyze in one batch (cluster mode only)')
parser.add_argument('--run_contaminated', action='store_true', help='Run the pipeline for contaminated gene sets')
parser.add_argument('--start', type=int, default=None, help='Start index for gene set/cluster range (default: 0)')
parser.add_argument('--end', type=int, default=None, help='End index for gene set/cluster range (default: process all)')
parser.add_argument('--gene_features', type=str, default=None, help='Path to csv with gene features if needed for prompt')
parser.add_argument('--output_file', type=str, required=True, help='Path to output file (no extension)')

args = parser.parse_args()

config_file = args.config
input_file = args.input
ind_start = args.start
ind_end = args.end
gene_column = args.gene_column
gene_sep = args.gene_sep
set_index = args.set_index
input_sep = args.input_sep
mode = args.mode
batch_size = args.batch_size
gene_features_file = args.gene_features
out_file = args.output_file

# Load configuration
with open(config_file) as json_file:
    config = json.load(json_file)
    
# Handle customized prompt if configured
custom_prompt_file = config.get('CUSTOM_PROMPT_FILE')
customized_prompt = None
if custom_prompt_file and os.path.isfile(custom_prompt_file):
    with open(custom_prompt_file, 'r') as f:
        customized_prompt = f.read()
        assert len(customized_prompt) > 1, "Customized prompt is empty"

# Extract configuration values
context = config['CONTEXT']
model = config['MODEL']
temperature = config['TEMP']
max_tokens = config['MAX_TOKENS']

# Create log file name
log_suffix = f"_{ind_start}_{ind_end}" if ind_start is not None and ind_end is not None else "_all"
LOG_FILE = config['LOG_NAME'] + log_suffix + ".log"

# Get rate from API settings if available
if 'API_SETTINGS' in config:
    for provider, settings in config['API_SETTINGS'].items():
        if any(model.startswith(m.split('-')[0]) for m in settings.get('models', [])):
            rate_per_token = settings.get('rate_per_token', 0.00001)
            break
    else:
        rate_per_token = config.get('RATE_PER_TOKEN', 0.00001)  # Default fallback
else:
    rate_per_token = config.get('RATE_PER_TOKEN', 0.00001)  # Default fallback

DOLLAR_LIMIT = config.get('DOLLAR_LIMIT', 10.0)  # Default fallback

# Set random seed for reproducibility
seed = constant.SEED

# Load gene features if provided
gene_features_dict = None
if gene_features_file:
    try:
        features_df = pd.read_csv(gene_features_file)
        gene_id_column = features_df.columns[0]  # Assume first column is gene ID
        features_column = features_df.columns[1]  # Assume second column is features
        gene_features_dict = dict(zip(features_df[gene_id_column], features_df[features_column]))
        print(f"Loaded features for {len(gene_features_dict)} genes")
    except Exception as e:
        print(f"Error loading gene features: {e}")
        gene_features_dict = None

def process_gene_set(df):
    """
    Process gene sets using the original pipeline.
    
    Args:
        df: DataFrame containing gene sets to analyze
    """
    analysis_dict = {}
    
    # Get logger
    logger = get_model_logger(model, ind_start, ind_end)

    i = 0  # Used for tracking progress and saving the file
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Only process None rows 
        if pd.notna(row.get(f'{column_prefix} Analysis')):
            continue
        
        gene_data = row[gene_column]
        # If gene_data is not a string, then skip
        if type(gene_data) != str:
            logger.warning(f'Gene set {idx} is not a string, skipping')
            continue
        
        genes = gene_data.split(gene_sep)
        
        # Check if gene set is too large
        if len(genes) > constant.MAX_GENES_PER_ANALYSIS:
            logger.warning(f'Gene set {idx} is too big ({len(genes)} genes), skipping')
            continue

        try:
            # Create prompt
            prompt = make_user_prompt_with_score(genes, gene_features_dict)
            finger_print = None
            
            # Call appropriate API based on model name
            if model.startswith('gpt'):
                logger.info("Accessing OpenAI API")
                analysis, finger_print = openai_chat(context, prompt, model, temperature, max_tokens, rate_per_token, LOG_FILE, DOLLAR_LIMIT, seed)
            elif model.startswith('gemini'):
                logger.info("Using Google Gemini API")
                analysis, error_message = query_genai_model(context, prompt, model, temperature, max_tokens, LOG_FILE) 
            elif model.startswith('claude'):
                logger.info("Using Anthropic Claude API")
                analysis, error_message = anthropic_chat(context, prompt, model, temperature, max_tokens, LOG_FILE, seed)
            elif model.startswith('deepseek') or model.startswith('llama') or model.startswith('mistral'):
                logger.info(f"Using Perplexity API with {model}")
                analysis, error_message = perplexity_chat(context, prompt, model, temperature, max_tokens, LOG_FILE)
            else:
                logger.info("Using server model")
                analysis, error_message = server_model_chat(context, prompt, model, temperature, max_tokens, LOG_FILE, seed)
            
            # Process the analysis if we got one
            if analysis:
                # Extract function name, confidence score, and detailed analysis
                llm_name, llm_score, llm_analysis = process_analysis(analysis)
                
                # Clean up the score and return float
                try:
                    llm_score_value = float(re.sub("[^0-9.-]", "", llm_score))
                except ValueError:
                    llm_score_value = float(0)
                
                # Update dataframe with results (using loc method correctly)
                df.loc[idx, f'{column_prefix} Name'] = llm_name
                df.loc[idx, f'{column_prefix} Analysis'] = llm_analysis
                df.loc[idx, f'{column_prefix} Score'] = llm_score_value
                
                # Save raw response
                analysis_dict[f'{idx}_{column_prefix}'] = analysis
                
                # Log success with fingerprint
                logger.info(f'Success for {idx} {column_prefix}.')
                if finger_print:
                    logger.info(f'Model fingerprint for {idx}: {finger_print}')
            else:
                if 'error_message' in locals() and error_message:
                    logger.error(f'Error for query gene set {idx}: {error_message}')
                else:
                    logger.error(f'Error for query gene set {idx}: No analysis returned')
                    
        except Exception as e:
            logger.error(f'Error for {idx}: {e}')
            continue
        
        # Save progress periodically
        i += 1
        if i % 10 == 0:
            # Bin scores into confidence categories
            bins = constant.SCORE_BIN_RANGES
            labels = constant.SCORE_BIN_LABELS
            
            df[f'{column_prefix} Score bins'] = pd.cut(df[f'{column_prefix} Score'], bins=bins, labels=labels)
            save_progress(df, analysis_dict, out_file)
            logger.info(f"Saved progress for {i} gene sets")
    
    # Save the final file
    # Bin scores into confidence categories
    bins = constant.SCORE_BIN_RANGES
    labels = constant.SCORE_BIN_LABELS
    
    df[f'{column_prefix} Score bins'] = pd.cut(df[f'{column_prefix} Score'], bins=bins, labels=labels)
    save_progress(df, analysis_dict, out_file)

def process_clusters(df):
    """
    Process gene clusters using the new pathway discovery pipeline.
    
    Args:
        df: DataFrame containing gene clusters to analyze
    """
    clusters_dict = {}
    
    # Get logger
    logger = get_model_logger(model, ind_start, ind_end)

    # Process clusters one at a time or in batches
    if batch_size <= 1:
        # Process one cluster at a time
        i = 0  # Used for tracking progress and saving the file
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            # Skip if already processed
            if str(idx) in clusters_dict:
                continue
            
            gene_data = row[gene_column]
            # If gene_data is not a string, then skip
            if type(gene_data) != str:
                logger.warning(f'Cluster {idx} genes is not a string, skipping')
                continue
            
            genes = gene_data.split(gene_sep)
            
            # Check if gene set is too large
            if len(genes) > constant.MAX_GENES_PER_ANALYSIS:
                logger.warning(f'Cluster {idx} is too big ({len(genes)} genes), skipping')
                continue

            try:
                # Create prompt
                prompt = make_cluster_analysis_prompt(idx, genes, gene_features_dict)
                
                # Call appropriate API based on model name
                if model.startswith('gpt'):
                    logger.info("Accessing OpenAI API")
                    analysis, finger_print = openai_chat(context, prompt, model, temperature, max_tokens, rate_per_token, LOG_FILE, DOLLAR_LIMIT, seed)
                elif model.startswith('gemini'):
                    logger.info("Using Google Gemini API")
                    analysis, error_message = query_genai_model(context, prompt, model, temperature, max_tokens, LOG_FILE) 
                elif model.startswith('claude'):
                    logger.info("Using Anthropic Claude API")
                    analysis, error_message = anthropic_chat(context, prompt, model, temperature, max_tokens, LOG_FILE, seed)
                elif model.startswith('deepseek') or model.startswith('llama') or model.startswith('mistral'):
                    logger.info(f"Using Perplexity API with {model}")
                    analysis, error_message = perplexity_chat(context, prompt, model, temperature, max_tokens, LOG_FILE)
                else:
                    logger.info("Using server model")
                    analysis, error_message = server_model_chat(context, prompt, model, temperature, max_tokens, LOG_FILE, seed)
                
                # Process the analysis if we got one
                if analysis:
                    # Parse the structured output
                    cluster_result = process_cluster_analysis(analysis)
                    
                    # Store the result
                    clusters_dict[str(idx)] = cluster_result
                    
                    # Log success
                    logger.info(f'Success for cluster {idx}')
                    if 'finger_print' in locals() and finger_print:
                        logger.info(f'Model fingerprint for {idx}: {finger_print}')
                else:
                    if 'error_message' in locals() and error_message:
                        logger.error(f'Error for cluster {idx}: {error_message}')
                    else:
                        logger.error(f'Error for cluster {idx}: No analysis returned')
                
            except Exception as e:
                logger.error(f'Error for cluster {idx}: {e}')
                continue
            
            # Save progress periodically
            i += 1
            if i % 5 == 0:
                save_cluster_analysis(clusters_dict, out_file)
                logger.info(f"Saved progress for {i} clusters")
    
    else:
        # Process clusters in batches
        batch_clusters = {}
        batch_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            # Skip if already processed
            if str(idx) in clusters_dict:
                continue
            
            gene_data = row[gene_column]
            # If gene_data is not a string, then skip
            if type(gene_data) != str:
                logger.warning(f'Cluster {idx} genes is not a string, skipping')
                continue
            
            genes = gene_data.split(gene_sep)
            
            # Check if gene set is too large
            if len(genes) > constant.MAX_GENES_PER_ANALYSIS:
                logger.warning(f'Cluster {idx} is too big ({len(genes)} genes), skipping')
                continue
            
            # Add to current batch
            batch_clusters[str(idx)] = genes
            batch_count += 1
            
            # Process batch if it's full or we're at the end
            if batch_count >= batch_size or idx == df.index[-1]:
                try:
                    # Create batch prompt
                    prompt = make_batch_cluster_prompt(batch_clusters, gene_features_dict)
                    
                    # Call appropriate API
                    if model.startswith('gpt'):
                        logger.info("Accessing OpenAI API")
                        analysis, finger_print = openai_chat(context, prompt, model, temperature, max_tokens, rate_per_token, LOG_FILE, DOLLAR_LIMIT, seed)
                    elif model.startswith('gemini'):
                        logger.info("Using Google Gemini API")
                        analysis, error_message = query_genai_model(context, prompt, model, temperature, max_tokens, LOG_FILE) 
                    elif model.startswith('claude'):
                        logger.info("Using Anthropic Claude API")
                        analysis, error_message = anthropic_chat(context, prompt, model, temperature, max_tokens, LOG_FILE, seed)
                    elif model.startswith('deepseek') or model.startswith('llama') or model.startswith('mistral'):
                        logger.info(f"Using Perplexity API with {model}")
                        analysis, error_message = perplexity_chat(context, prompt, model, temperature, max_tokens, LOG_FILE)
                    else:
                        logger.info("Using server model")
                        analysis, error_message = server_model_chat(context, prompt, model, temperature, max_tokens, LOG_FILE, seed)
                    
                    # Process the batch analysis if we got one
                    if analysis:
                        # Parse the structured output - returns dict of cluster_id -> analysis
                        batch_results = process_batch_cluster_analysis(analysis)
                        
                        # Merge with main results
                        clusters_dict.update(batch_results)
                        
                        # Log success
                        logger.info(f'Successfully processed batch of {len(batch_clusters)} clusters')
                    else:
                        if 'error_message' in locals() and error_message:
                            logger.error(f'Error for batch: {error_message}')
                        else:
                            logger.error(f'Error for batch: No analysis returned')
                
                except Exception as e:
                    logger.error(f'Error processing batch: {e}')
                
                # Reset batch for next round
                batch_clusters = {}
                batch_count = 0
                
                # Save progress
                save_cluster_analysis(clusters_dict, out_file)
                logger.info(f"Saved progress with {len(clusters_dict)} clusters processed so far")
    
    # Save final results
    save_cluster_analysis(clusters_dict, out_file)
    logger.info(f"Completed analysis for {len(clusters_dict)} clusters")

if __name__ == "__main__":
    # Handle tab separator
    if input_sep == '\\t':
        input_sep = '\t'
    
    # Load the data
    raw_df = pd.read_csv(input_file, sep=input_sep, index_col=set_index)
    print(f"Loaded data with columns: {raw_df.columns}")

    # Set start and end indices if not provided
    ind_start = args.start if args.start is not None else 0
    ind_end = args.end if args.end is not None else len(raw_df)

    print(f"Processing rows from index {ind_start} to {ind_end-1}")

    # Only process the specified range of entries
    df = raw_df.iloc[ind_start:ind_end].copy()
    
    # Create column prefix based on model name
    if '-' in model:
        name_fix = '_'.join(model.split('-')[:2])
    else:
        name_fix = model.replace(':', '_')
    
    # Choose which pipeline to run based on mode
    if mode == 'gene_set':
        # Original gene set analysis mode
        column_prefix = name_fix + '_default'
        
        if args.initialize:
            # Initialize the input file with llm names, analysis and score to None
            df[f'{column_prefix} Name'] = None
            df[f'{column_prefix} Analysis'] = None
            df[f'{column_prefix} Score'] = -np.inf
        
        print(f"Found {df[f'{column_prefix} Analysis'].isna().sum()} gene sets to analyze")
        process_gene_set(df)
        
        # If run_contaminated is true, then run the pipeline for contaminated gene sets
        if args.run_contaminated:
            # Run the pipeline for contaminated gene sets 
            contaminated_columns = [col for col in df.columns if col.endswith('contaminated_Genes')]
            
            for col in contaminated_columns:
                gene_column = col  # Note need to change the gene_column to the contaminated column
                contam_prefix = '_'.join(col.split('_')[0:2])
                
                column_prefix = name_fix + '_' + contam_prefix
                print(column_prefix)
                
                if args.initialize:
                    # Initialize the input file with llm names, analysis and score to None
                    df[f'{column_prefix} Name'] = None
                    df[f'{column_prefix} Analysis'] = None
                    df[f'{column_prefix} Score'] = -np.inf
                
                print(f"Found {df[f'{column_prefix} Analysis'].isna().sum()} contaminated gene sets to analyze")
                process_gene_set(df)
    
    elif mode == 'cluster':
        # New cluster analysis mode
        print(f"Processing {df.shape[0]} clusters in range {ind_start}-{ind_end}")
        process_clusters(df)
    
    print("Analysis completed successfully")