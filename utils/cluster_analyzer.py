"""
Utility module for gene cluster analysis with LLMs.
This module provides functions extracted from main.py to enable reuse
in different contexts (CLI, notebook, etc.)
"""

import pandas as pd
import json
import logging
import time
from datetime import datetime
from tqdm import tqdm
import os

# Import utility functions
from .openai_query import openai_chat
from .anthropic_query import anthropic_chat
from .gemini_query import query_genai_model
from .server_model_query import server_model_chat
from .prompt_factory import (
    make_cluster_analysis_prompt,
    make_batch_cluster_analysis_prompt,
)
from .llm_analysis_utils import (
    process_cluster_response,
    save_cluster_analysis,
)
from .logging_utils import get_model_logger, setup_logger

# Import constants
import constant


def load_config(config_file=None, model_override=None):
    """
    Load configuration with optional model override.

    Args:
        config_file: Path to config file (optional)
        model_override: Model to use, overriding config (optional)

    Returns:
        config: Configuration dictionary
    """
    # Default configuration
    default_config = {
        "MODEL": "",
        "CONTEXT": "You are an AI assistant specializing in genomics and systems biology.",
        "TEMP": 0.0,
        "MAX_TOKENS": 4000,
        "RATE_PER_TOKEN": 0.00001,
        "DOLLAR_LIMIT": 10.0,
        "LOG_NAME": "analysis",
        "API_SETTINGS": {
            "openai": {
                "models": ["gpt-4o", "gpt-4.5", "gpt-3.5-turbo"],
                "rate_per_token": 0.00001,
            },
            "anthropic": {
                "models": [
                    "claude-3-7-sonnet-20250219",
                    "claude-3.5-sonnet",
                    "claude-3-opus-20240229",
                ],
                "rate_per_token": 0.000015,
            },
            "gemini": {
                "models": ["gemini-2.5-pro-exp-03-25", "gemini-2.0-flash"],
                "rate_per_token": 0.000005,
            },
        },
    }

    # If config file provided, load and merge with defaults
    if config_file:
        try:
            with open(config_file) as json_file:
                user_config = json.load(json_file)
                # Update default config with user settings
                for key, value in user_config.items():
                    if (
                        isinstance(value, dict)
                        and key in default_config
                        and isinstance(default_config[key], dict)
                    ):
                        # Deep merge for nested dictionaries
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
        except Exception as e:
            logging.error(f"Error loading config file: {e}")

    # Override model if specified
    if model_override:
        default_config["MODEL"] = model_override

    # Determine rate_per_token based on model if not explicitly set
    model = default_config["MODEL"]
    if model and "API_SETTINGS" in default_config:
        for provider, settings in default_config["API_SETTINGS"].items():
            if any(
                model.startswith(m.split("-")[0]) for m in settings.get("models", [])
            ):
                rate_per_token = settings.get(
                    "rate_per_token", default_config.get("RATE_PER_TOKEN", 0.00001)
                )
                default_config["RATE_PER_TOKEN"] = rate_per_token
                break

    # Add lowercase aliases for consistency
    default_config["context"] = default_config["CONTEXT"]
    default_config["model"] = default_config["MODEL"]
    default_config["temperature"] = default_config["TEMP"]
    default_config["max_tokens"] = default_config["MAX_TOKENS"]
    default_config["rate_per_token"] = default_config["RATE_PER_TOKEN"]
    default_config["dollar_limit"] = default_config["DOLLAR_LIMIT"]
    default_config["log_name"] = default_config["LOG_NAME"]

    return default_config


def load_gene_features(gene_features_file):
    """Load gene features from a file if provided."""
    if not gene_features_file:
        return None

    try:
        features_df = pd.read_csv(gene_features_file)
        gene_id_column = features_df.columns[0]  # Assume first column is gene ID
        features_column = features_df.columns[1]  # Assume second column is features
        gene_features_dict = dict(
            zip(features_df[gene_id_column], features_df[features_column])
        )
        print(f"Loaded features for {len(gene_features_dict)} genes")
        return gene_features_dict
    except Exception as e:
        print(f"Error loading gene features: {e}")
        return None


def load_screen_info(screen_info_file):
    """Load screen information from a file if provided."""
    if not screen_info_file:
        return None

    try:
        with open(screen_info_file, "r") as f:
            screen_info = f.read().strip()
        print(f"Loaded screen information: {len(screen_info)} characters")
        return screen_info
    except Exception as e:
        print(f"Error loading screen information: {e}")
        return None


def query_llm(
    context,
    prompt,
    model,
    temperature,
    max_tokens,
    rate_per_token,
    log_file,
    dollar_limit,
    seed=None,
):
    """Send a query to the appropriate LLM based on model name."""
    logger = logging.getLogger(log_file)

    try:
        # Call appropriate API based on model name
        if model.startswith("gpt") or model.startswith("o4") or model.startswith("o3"):
            logger.info("Accessing OpenAI API")
            return openai_chat(
                context,
                prompt,
                model,
                temperature,
                max_tokens,
                rate_per_token,
                log_file,
                dollar_limit,
                seed,
            )
        elif model.startswith("gemini"):
            logger.info("Using Google Gemini API")
            analysis, error_message = query_genai_model(
                context, prompt, model, temperature, max_tokens, log_file
            )
            return analysis, None if analysis else error_message
        elif model.startswith("claude"):
            logger.info("Using Anthropic Claude API")
            analysis, error_message = anthropic_chat(
                context, prompt, model, temperature, max_tokens, log_file, seed
            )
            return analysis, None if analysis else error_message
        else:
            logger.info("Using server model")
            analysis, error_message = server_model_chat(
                context, prompt, model, temperature, max_tokens, log_file, seed
            )
            return analysis, None if analysis else error_message
    except Exception as e:
        logger.error(f"Error querying LLM: {str(e)}")
        return None, f"Error: {str(e)}"


def process_clusters(
    df, config, gene_column, gene_sep, out_file, custom_prompt_path=None,
    gene_features_path=None, screen_info_path=None, batch_size=1, 
    start_idx=0, end_idx=None, log_name=None, use_tqdm=True
):
    """
    Process gene clusters to identify pathways and novel members.
    
    Args:
        df: DataFrame with clusters to analyze
        config: Configuration dictionary
        gene_column: Column name containing gene symbols
        gene_sep: Separator for genes in the column
        out_file: Output file path (without extension)
        custom_prompt_path: Path to custom prompt template (optional)
        gene_features_path: Path to gene features file (optional)
        screen_info_path: Path to screen info file (optional)
        batch_size: Number of clusters to analyze in one batch
        start_idx: Start index in DataFrame (default: 0)
        end_idx: End index in DataFrame (default: all)
        log_name: Custom log name (default: from config)
        use_tqdm: Whether to use tqdm progress bar
        
    Returns:
        clusters_dict: Dictionary of cluster analysis results
    """
    # Set up logging
    if not log_name:
        log_name = config.get("log_name", "cluster_analysis")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_name}_{timestamp}.log"
    logger = setup_logger(log_file)
    
    # Load gene features and screen info if paths provided
    gene_features_dict = load_gene_features(gene_features_path)
    screen_info = load_screen_info(screen_info_path)
    
    # Extract config values
    context = config["context"]
    model = config["model"]
    temperature = config["temperature"]
    max_tokens = config["max_tokens"]
    rate_per_token = config["rate_per_token"]
    dollar_limit = config["dollar_limit"]
    
    # Apply range limits
    if end_idx is None:
        end_idx = len(df)
    df = df.iloc[start_idx:end_idx].copy()
    
    logger.info(f"Processing {len(df)} clusters with model {model}")
    
    # Dictionary to store results
    clusters_dict = {}
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    # Use tqdm for progress tracking if requested
    if use_tqdm:
        row_iterator = tqdm(df.iterrows(), total=len(df), desc="Processing clusters")
    else:
        row_iterator = df.iterrows()
    
    if batch_size <= 1:
        # Process one cluster at a time
        for idx, row in row_iterator:
            # Skip if already processed
            if str(idx) in clusters_dict:
                continue

            # Get genes for this cluster
            gene_data = row[gene_column]
            if not isinstance(gene_data, str):
                logger.warning(f"Cluster {idx} genes is not a string, skipping")
                continue

            genes = gene_data.split(gene_sep)

            # Check if gene set is too large
            if len(genes) > constant.MAX_GENES_PER_ANALYSIS:
                logger.warning(
                    f"Cluster {idx} is too big ({len(genes)} genes), skipping"
                )
                continue

            # Create prompt and query LLM
            prompt = make_cluster_analysis_prompt(
                idx,
                genes,
                gene_features_dict,
                screen_info,
                template_path=custom_prompt_path,
            )
            
            analysis, error = query_llm(
                context,
                prompt,
                model,
                temperature,
                max_tokens,
                rate_per_token,
                log_file,
                dollar_limit,
                constant.SEED,
            )

            # Process the analysis if we got one
            if analysis:
                # Parse the structured output
                cluster_result = process_cluster_response(analysis, is_batch=False)

                # Store the result
                clusters_dict[str(idx)] = cluster_result
                logger.info(f"Success for cluster {idx}")
            else:
                logger.error(f"Error for cluster {idx}: {error}")

            # Save progress periodically
            if len(clusters_dict) % 5 == 0:
                save_cluster_analysis(clusters_dict, out_file)
                logger.info(f"Saved progress for {len(clusters_dict)} clusters")
    
    else:
        # Process clusters in batches
        batch_clusters = {}
        
        for i, (idx, row) in enumerate(row_iterator):
            is_last_cluster = i == len(df) - 1

            # Skip if already processed
            if str(idx) in clusters_dict:
                continue

            # Get genes for this cluster
            gene_data = row[gene_column]
            if not isinstance(gene_data, str):
                logger.warning(f"Cluster {idx} genes is not a string, skipping")
                continue

            genes = gene_data.split(gene_sep)

            # Check if gene set is too large
            if len(genes) > constant.MAX_GENES_PER_ANALYSIS:
                logger.warning(
                    f"Cluster {idx} is too big ({len(genes)} genes), skipping"
                )
                continue

            # Add to current batch
            batch_clusters[str(idx)] = genes

            # Process batch if it's full or we're at the end
            if len(batch_clusters) >= batch_size or is_last_cluster:
                batch_size_actual = len(batch_clusters)

                try:
                    # Create batch prompt
                    prompt = make_batch_cluster_analysis_prompt(
                        batch_clusters,
                        gene_features_dict,
                        screen_info,
                        template_path=custom_prompt_path,
                    )

                    # Query LLM
                    analysis, error = query_llm(
                        context,
                        prompt,
                        model,
                        temperature,
                        max_tokens,
                        rate_per_token,
                        log_file,
                        dollar_limit,
                        constant.SEED,
                    )

                    # Process the batch analysis if we got one
                    if analysis:
                        # Parse the structured output - returns dict of cluster_id -> analysis
                        batch_results = process_cluster_response(
                            analysis, is_batch=True
                        )

                        # Check if parsing was successful
                        if batch_results and len(batch_results) > 0:
                            # Merge with main results
                            clusters_dict.update(batch_results)

                            # Log success
                            logger.info(
                                f"Successfully processed batch of {len(batch_clusters)} clusters, extracted {len(batch_results)} results"
                            )

                            # Check if we missed any clusters
                            missing_clusters = [
                                cid
                                for cid in batch_clusters.keys()
                                if cid not in batch_results
                            ]
                            if missing_clusters:
                                logger.warning(
                                    f"Failed to extract results for clusters: {', '.join(missing_clusters)}"
                                )
                        else:
                            logger.error(
                                "Failed to extract any cluster results from batch response"
                            )
                    else:
                        logger.error(f"Error processing batch: {error}")

                except Exception as e:
                    logger.error(f"Error processing batch: {e}")

                # Reset batch for next round
                batch_clusters = {}

                # Save progress
                save_cluster_analysis(clusters_dict, out_file)
                logger.info(f"Saved progress with {len(clusters_dict)} clusters processed so far")

    # Save final results
    save_cluster_analysis(clusters_dict, out_file)
    logger.info(f"Completed analysis for {len(clusters_dict)} clusters")
    
    return clusters_dict


def analyze_gene_clusters(
    input_file, 
    output_file, 
    config_path=None, 
    model_name=None, 
    custom_prompt_path=None, 
    gene_features_path=None, 
    screen_info_path=None, 
    input_sep=",", 
    gene_column="genes", 
    gene_sep=";", 
    batch_size=1, 
    start_idx=0, 
    end_idx=None,
    log_name=None
):
    """
    High-level function to analyze gene clusters from a file.
    
    Args:
        input_file: Path to input CSV/TSV with gene clusters
        output_file: Path to output file (without extension)
        config_path: Path to configuration JSON file
        model_name: Override model specified in config
        custom_prompt_path: Path to custom prompt template
        gene_features_path: Path to gene features CSV
        screen_info_path: Path to screen info text file
        input_sep: Separator for input CSV (default: comma)
        gene_column: Column name containing gene symbols
        gene_sep: Separator for genes within a set
        batch_size: Number of clusters to analyze in one batch
        start_idx: Start index for processing
        end_idx: End index for processing
        log_name: Custom log name
        
    Returns:
        clusters_dict: Dictionary of cluster analysis results
    """
    # Load configuration
    config = load_config(config_path, model_override=model_name)
    
    # Handle tab separator conversion
    if input_sep == "\\t":
        input_sep = "\t"
    
    # Load the data
    try:
        df = pd.read_csv(input_file, sep=input_sep)
        print(
            f"Loaded data with {len(df)} rows and columns: {list(df.columns)}"
        )
    except Exception as e:
        print(f"Error loading input file: {e}")
        return None
    
    # Process clusters
    return process_clusters(
        df=df,
        config=config,
        gene_column=gene_column,
        gene_sep=gene_sep,
        out_file=output_file,
        custom_prompt_path=custom_prompt_path,
        gene_features_path=gene_features_path,
        screen_info_path=screen_info_path,
        batch_size=batch_size,
        start_idx=start_idx,
        end_idx=end_idx,
        log_name=log_name
    )