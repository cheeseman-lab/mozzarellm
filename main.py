import argparse
import pandas as pd
import json 
import numpy as np
import os
import logging
import re
from tqdm import tqdm
from dotenv import load_dotenv

# Import utility functions
from utils.openai_query import openai_chat
from utils.anthropic_query import anthropic_chat
from utils.gemini_query import query_genai_model
from utils.perplexity_query import perplexity_chat
from utils.server_model_query import server_model_chat
from utils.prompt_factory import make_user_prompt_with_score
from utils.llm_analysis_utils import process_analysis, save_progress
from utils.logging_utils import get_model_logger

# Import constants
import constant

# Load environment variables
load_dotenv()

# Add argument parsing
parser = argparse.ArgumentParser(description='Process range of gene sets with LLMs.')
parser.add_argument('--config', type=str, required=True, help='Config file for LLM')
parser.add_argument('--initialize', action='store_true', help='If provided, initializes the input file with llm names, analysis and score to None. By default, this is not done.')
parser.add_argument('--input', type=str, required=True, help='Path to input csv with gene sets')
parser.add_argument('--input_sep', type=str, required=True, help='Separator for input csv')
parser.add_argument('--set_index', type=str, default='0', help='Column name for gene set index, default would be first column')
parser.add_argument('--gene_column', type=str, required=True, help='Column name for gene set')
parser.add_argument('--gene_sep', type=str, required=True, help='Separator for gene set')
parser.add_argument('--run_contaminated', action='store_true', help='If provided, runs the pipeline for contaminated gene sets. By default, this is not done.')
parser.add_argument('--start', type=int, required=True, help='Start index for gene set range')
parser.add_argument('--end', type=int, required=True, help='End index for gene set range')
parser.add_argument('--gene_features', type=str, default=None, help='Path to csv with gene features if need to be included in prompt')
parser.add_argument('--direct', action='store_true', default=None, help='Whether to use direct prompt or not, default is None')
parser.add_argument('--customized_prompt', type=str, default=None, help='If using customized prompt then use the path to the customized prompt, default is None')
parser.add_argument('--output_file', type=str, required=True, help='Path to output with LLM analysis, no need to include file extension, will be saved as .tsv and .json')

args = parser.parse_args()

config_file = args.config
input_file = args.input
ind_start = args.start
ind_end = args.end
gene_column = args.gene_column
gene_sep = args.gene_sep
set_index = args.set_index
input_sep = args.input_sep
gene_features = args.gene_features
direct = args.direct
customized_prompt = args.customized_prompt
out_file = args.output_file


# Load configuration
with open(config_file) as json_file:
    config = json.load(json_file)
    
# Handle customized prompt if provided
if args.customized_prompt:
    # Check if the config file has a CUSTOM_PROMPT_FILE key
    custom_prompt_file = config.get('CUSTOM_PROMPT_FILE', args.customized_prompt)
    
    # Make sure the file exists
    if os.path.isfile(custom_prompt_file):
        with open(custom_prompt_file, 'r') as f:
            customized_prompt = f.read()
            assert len(customized_prompt) > 1, "Customized prompt is empty"
    else:
        print(f"Customized prompt file {custom_prompt_file} does not exist")
        customized_prompt = None
else:
    customized_prompt = None

# Extract configuration values
context = config['CONTEXT']
model = config['MODEL']
temperature = config['TEMP']
max_tokens = config['MAX_TOKENS']
LOG_FILE = config['LOG_NAME'] + f'_{ind_start}_{ind_end}.log'

# For OpenAI models, get additional parameters
if model.startswith('gpt'):
    rate_per_token = config['RATE_PER_TOKEN']
    DOLLAR_LIMIT = config['DOLLAR_LIMIT']
else:
    # Get rate from API settings if available
    if 'API_SETTINGS' in config:
        for provider, settings in config['API_SETTINGS'].items():
            if any(model.startswith(m.split('-')[0]) for m in settings.get('models', [])):
                rate_per_token = settings.get('rate_per_token', 0.00001)
                break
        else:
            rate_per_token = 0.00001  # Default fallback
    else:
        rate_per_token = 0.00001  # Default fallback
    
    DOLLAR_LIMIT = config.get('DOLLAR_LIMIT', 10.0)  # Default fallback

# Set random seed for reproducibility
seed = constant.SEED

def main(df):
    """
    Main function to process gene sets and get LLM analysis.
    
    Args:
        df: DataFrame containing gene sets to analyze
    """
    analysis_dict = {}
    
    # Get logger
    logger = get_model_logger(model, ind_start, ind_end)

    i = 0  # Used for tracking progress and saving the file
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Only process None rows 
        if pd.notna(row[f'{column_prefix} Analysis']):
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
            prompt = make_user_prompt_with_score(genes, gene_features)
            finger_print = None
            
            # Select which API to use based on model name
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
                
                # Update dataframe with results
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


if __name__ == "__main__":
    # Handle tab separator
    if input_sep == '\\t':
        input_sep = '\t'
    
    # Load the data
    raw_df = pd.read_csv(input_file, sep=input_sep, index_col=set_index)
    print(raw_df.columns)

    # Only process the specified range of genes
    df = raw_df.iloc[ind_start:ind_end]
    
    # Create column prefix based on model name
    if '-' in model:
        name_fix = '_'.join(model.split('-')[:2])
    else:
        name_fix = model.replace(':', '_')
    column_prefix = name_fix + '_default'  # Start with default gene set
    
    if args.initialize:
        # Initialize the input file with llm names, analysis and score to None
        df[f'{column_prefix} Name'] = None
        df[f'{column_prefix} Analysis'] = None
        df[f'{column_prefix} Score'] = -np.inf
    
    print(f"Found {df[f'{column_prefix} Analysis'].isna().sum()} gene sets to analyze")
    main(df)  # Run with the real set 
    
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
            main(df)

print("Analysis completed successfully")