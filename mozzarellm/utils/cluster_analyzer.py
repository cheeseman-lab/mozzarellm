"""
Utility module for gene cluster analysis with LLMs.
This module provides functions extracted from main.py to enable reuse
in different contexts (CLI, notebook, etc.)
"""

import pandas as pd
import logging
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
from .logging_utils import setup_logger

# Import configuration and prompt constants
from mozzarellm.configs import (
    DEFAULT_CONFIG,
    DEFAULT_OPENAI_CONFIG,
    DEFAULT_ANTHROPIC_CONFIG,
    DEFAULT_GEMINI_CONFIG,
)

# Import constants
from mozzarellm import constant


def analyze_gene_clusters(
    # Input data options
    input_file=None,
    input_df=None,
    input_sep=",",
    gene_column="genes",
    gene_sep=";",
    cluster_id_column="cluster_id",
    model_name=None,
    config_path=None,
    config_dict=None,
    screen_context_path=None,
    screen_context=None,
    cluster_analysis_prompt_path=None,
    cluster_analysis_prompt=None,
    gene_annotations_path=None,
    gene_annotations_df=None,
    batch_size=1,
    start_idx=0,
    end_idx=None,
    output_file=None,
    save_outputs=True,
    outputs_to_generate=["json", "clusters", "flagged_genes"],
):
    """
    Analyze gene clusters to identify biological pathways and potential novel gene functions.

    Parameters:
    -----------
    # Input data options
    input_file : str, optional
        Path to a CSV/TSV file containing gene clusters
    input_df : pandas.DataFrame, optional
        DataFrame containing gene clusters (alternative to input_file)
    input_sep : str, default=","
        Separator used in the input file

    # Data structure parameters
    gene_column : str, default="genes"
        Column name containing semicolon-separated gene lists
    gene_sep : str, default=";"
        Separator used between genes in the gene column
    cluster_id_column : str, default="cluster_id"
        Column name containing cluster identifiers

    # Model and configuration
    model_name : str, optional
        LLM to use for analysis (e.g., "gpt-4o", "claude-3-7-sonnet-20250219")
    config_path : str, optional
        Path to a JSON configuration file
    config_dict : dict, optional
        Configuration dictionary (alternative to config_path)

    # Analysis context
    screen_context_path : str, optional
        Path to a file containing information about the experiment/screen
    screen_context : str, optional
        String containing information about the experiment/screen

    # Prompts
    cluster_analysis_prompt_path : str, optional
        Path to a custom prompt template for pathway identification
    cluster_analysis_prompt : str, optional
        Custom prompt string for pathway identification

    # Gene information
    gene_annotations_path : str, optional
        Path to a CSV file with gene annotations/features, with 
        column 0 containing the gene, and column 1 containing the annotation/features
    gene_annotations_df : pandas.Dataframe
        Dataframe mapping gene IDs to their annotations/features, with 
        column 0 containing the gene, and column 1 containing the annotation/features

    # Processing options
    batch_size : int, default=1
        Number of clusters to analyze in each API call
    start_idx : int, default=0
        Starting index in the input data
    end_idx : int, optional
        Ending index in the input data

    # Output options
    output_file : str, optional
        Base path for output files (without extension)
    save_outputs : bool, default=True
        Whether to save results to disk
    outputs_to_generate : list, default=["json", "clusters", "flagged_genes"]
        Which output files to generate

    Returns:
    --------
    dict
        Dictionary containing analysis results and generated DataFrames
    """
    # Load configuration
    config = None
    if config_dict:
        config = config_dict
    else:
        config = load_config(config_path, model_override=model_name)

    # Automatically select config based on model name if not provided
    if not config and model_name:
        if any(model_name.startswith(prefix) for prefix in ["gpt", "o4", "o3"]):
            config = DEFAULT_OPENAI_CONFIG
        elif model_name.startswith("claude"):
            config = DEFAULT_ANTHROPIC_CONFIG
        elif model_name.startswith("gemini"):
            config = DEFAULT_GEMINI_CONFIG
        else:
            config = DEFAULT_CONFIG

    # Load data
    df = None
    if input_df is not None:
        df = input_df
    elif input_file is not None:
        if input_sep == "\\t":
            input_sep = "\t"
        df = pd.read_csv(input_file, sep=input_sep)
    else:
        raise ValueError("Either input_file or input_df must be provided")

    print(f"Loaded data with {len(df)} rows and columns: {list(df.columns)}")

    # Ensure we have a cluster ID column
    if cluster_id_column not in df.columns:
        print(
            f"Warning: Cluster ID column '{cluster_id_column}' not found. Using DataFrame index."
        )
        df[cluster_id_column] = df.index.astype(str)

    # Load gene annotations
    annotations = None
    if gene_annotations_df is not None:
        # Process the dataframe to create the annotations dictionary
        try:
            # Get column names for gene ID and features
            gene_id_column = gene_annotations_df.columns[0]  # First column is gene ID
            features_column = gene_annotations_df.columns[1]  # Second column is features
            
            # Create dictionary mapping gene IDs to their annotations
            annotations = dict(
                zip(gene_annotations_df[gene_id_column], gene_annotations_df[features_column])
            )
            print(f"Created annotations dictionary with {len(annotations)} entries from DataFrame")
        except Exception as e:
            print(f"Error processing gene annotations DataFrame: {e}")
            # Fall back to file-based loading if available
            if gene_annotations_path is not None:
                annotations = load_gene_annotations(gene_annotations_path)
            
    elif gene_annotations_path is not None:
        annotations = load_gene_annotations(gene_annotations_path)
        
    # Load screen context
    context = None
    if screen_context is not None:
        context = screen_context
    elif screen_context_path is not None:
        context = load_screen_context(screen_context_path)

    # Process clusters
    results = process_clusters(
        df=df,
        config=config,
        gene_column=gene_column,
        gene_sep=gene_sep,
        out_file=output_file if save_outputs else None,
        cluster_analysis_prompt_path=cluster_analysis_prompt_path,
        cluster_analysis_prompt=cluster_analysis_prompt,
        gene_annotations_dict=annotations,
        screen_context=context,
        batch_size=batch_size,
        start_idx=start_idx,
        end_idx=end_idx,
        cluster_id_column=cluster_id_column,
        save_outputs=save_outputs,
        outputs_to_generate=outputs_to_generate,
    )

    return results


def process_clusters(
    df,
    config,
    gene_column,
    gene_sep,
    out_file,
    cluster_analysis_prompt_path=None,
    cluster_analysis_prompt=None,
    gene_annotations_dict=None,
    screen_context=None,
    batch_size=1,
    start_idx=0,
    end_idx=None,
    log_name=None,
    use_tqdm=True,
    cluster_id_column="cluster_id",
    save_outputs=True,
    outputs_to_generate=["json", "clusters", "flagged_genes"],
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
        cluster_id_column: Column name for cluster ID (default: "cluster_id")

    Returns:
        clusters_dict: Dictionary of cluster analysis results
    """

    # Set up logging
    if not log_name:
        log_name = config.get("log_name", "cluster_analysis")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_name}_{timestamp}.log"
    logger = setup_logger(log_file)

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

    # Ensure cluster_id_column exists in the DataFrame
    if cluster_id_column not in df.columns:
        logger.warning(
            f"Cluster ID column '{cluster_id_column}' not found in DataFrame. Using index as cluster ID."
        )
        df[cluster_id_column] = df.index.astype(str)

    if batch_size <= 1:
        # Process one cluster at a time
        for idx, row in row_iterator:
            # Get actual cluster ID from the data
            cluster_id = str(row[cluster_id_column])

            # Skip if already processed
            if cluster_id in clusters_dict:
                continue

            # Get genes for this cluster
            gene_data = row[gene_column]
            if not isinstance(gene_data, str):
                logger.warning(f"Cluster {cluster_id} genes is not a string, skipping")
                continue

            genes = gene_data.split(gene_sep)

            # Check if gene set is too large
            if len(genes) > constant.MAX_GENES_PER_ANALYSIS:
                logger.warning(
                    f"Cluster {cluster_id} is too big ({len(genes)} genes), skipping"
                )
                continue

            # Create prompt and query LLM
            prompt = make_cluster_analysis_prompt(
                cluster_id,
                genes,
                gene_annotations_dict,
                screen_context,
                template_path=cluster_analysis_prompt_path,
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

                # Ensure the cluster_id in the result matches our actual cluster_id
                if cluster_result.get("cluster_id") != cluster_id:
                    logger.warning(
                        f"Cluster ID mismatch. Expected {cluster_id}, got {cluster_result.get('cluster_id')}. Fixing."
                    )
                    cluster_result["cluster_id"] = cluster_id

                # Store the result
                clusters_dict[cluster_id] = cluster_result
                logger.info(f"Success for cluster {cluster_id}")
            else:
                logger.error(f"Error for cluster {cluster_id}: {error}")

            # Save progress periodically
            if len(clusters_dict) % 5 == 0:
                save_cluster_analysis(clusters_dict, out_file)
                logger.info(f"Saved progress for {len(clusters_dict)} clusters")

    else:
        # Process clusters in batches
        batch_clusters = {}

        for i, (idx, row) in enumerate(row_iterator):
            is_last_cluster = i == len(df) - 1

            # Get actual cluster ID from the data
            cluster_id = str(row[cluster_id_column])

            # Skip if already processed
            if cluster_id in clusters_dict:
                continue

            # Get genes for this cluster
            gene_data = row[gene_column]
            if not isinstance(gene_data, str):
                logger.warning(f"Cluster {cluster_id} genes is not a string, skipping")
                continue

            genes = gene_data.split(gene_sep)

            # Check if gene set is too large
            if len(genes) > constant.MAX_GENES_PER_ANALYSIS:
                logger.warning(
                    f"Cluster {cluster_id} is too big ({len(genes)} genes), skipping"
                )
                continue

            # Add to current batch
            batch_clusters[cluster_id] = genes  # Use actual cluster_id from data

            # Process batch if it's full or we're at the end
            if len(batch_clusters) >= batch_size or is_last_cluster:
                try:
                    prompt = make_batch_cluster_analysis_prompt(
                        batch_clusters,
                        gene_annotations_dict,
                        screen_context,
                        template_path=cluster_analysis_prompt_path,
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
                            # Ensure cluster IDs match our actual cluster IDs
                            for result_id, result in list(batch_results.items()):
                                if result_id not in batch_clusters:
                                    # Try to match based on sequence or find closest match
                                    correct_id = None
                                    for real_id in batch_clusters.keys():
                                        if real_id not in batch_results:
                                            correct_id = real_id
                                            break

                                    if correct_id:
                                        logger.warning(
                                            f"Cluster ID mismatch. Got {result_id}, using {correct_id} instead."
                                        )
                                        batch_results[correct_id] = result
                                        batch_results[correct_id]["cluster_id"] = (
                                            correct_id
                                        )
                                        del batch_results[result_id]
                                else:
                                    # Ensure the result has the correct ID
                                    if result.get("cluster_id") != result_id:
                                        result["cluster_id"] = result_id

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
                logger.info(
                    f"Saved progress with {len(clusters_dict)} clusters processed so far"
                )

    # Save final results
    save_cluster_analysis(clusters_dict, out_file)
    logger.info(f"Completed analysis for {len(clusters_dict)} clusters")

    return clusters_dict


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


def load_config(config_file=None, model_override=None):
    """
    Load configuration with optional model override.
    Args:
        config_file: Path or name of config file (optional)
        model_override: Model to use, overriding config (optional)
    Returns:
        config: Configuration dictionary
    """
    from mozzarellm.configs import (
        DEFAULT_CONFIG, 
        DEFAULT_OPENAI_CONFIG,
        DEFAULT_OPENAI_REASONING_CONFIG,
        DEFAULT_ANTHROPIC_CONFIG,
        DEFAULT_GEMINI_CONFIG,
        REASONING_OPENAI_MODELS
    )
    
    # Select the appropriate base config
    if model_override:
        # Check if this is a REASONING OpenAI model
        if any(model_override.startswith(prefix) for prefix in REASONING_OPENAI_MODELS):
            config = DEFAULT_OPENAI_REASONING_CONFIG.copy()
        # Check for standard OpenAI models
        elif any(model_override.startswith(prefix) for prefix in ["gpt", "o4", "o3"]):
            config = DEFAULT_OPENAI_CONFIG.copy()
        # Check for Anthropic models
        elif model_override.startswith("claude"):
            config = DEFAULT_ANTHROPIC_CONFIG.copy()
        # Check for Gemini models
        elif model_override.startswith("gemini"):
            config = DEFAULT_GEMINI_CONFIG.copy()
        else:
            config = DEFAULT_CONFIG.copy()
        
        # Set the specified model
        config["MODEL"] = model_override
    else:
        # Start with default config if no model specified
        config = DEFAULT_CONFIG.copy()

    # Load custom config if provided
    if config_file:
        try:
            import json
            import os

            if os.path.exists(config_file):
                with open(config_file) as f:
                    custom_config = json.load(f)
                    # Update config with custom settings
                    for key, value in custom_config.items():
                        if (
                            isinstance(value, dict)
                            and key in config
                            and isinstance(config[key], dict)
                        ):
                            # Deep merge for nested dictionaries
                            config[key].update(value)
                        else:
                            config[key] = value
                logging.info(f"Loaded configuration from {config_file}")
            else:
                logging.warning(f"Config file not found: {config_file}")
        except Exception as e:
            logging.error(f"Error loading config file: {e}")

    # Add lowercase aliases for consistency
    config["context"] = config["CONTEXT"]
    config["model"] = config["MODEL"]
    config["temperature"] = config["TEMP"]
    config["max_tokens"] = config["MAX_TOKENS"]
    config["rate_per_token"] = config["RATE_PER_TOKEN"]
    config["dollar_limit"] = config["DOLLAR_LIMIT"]
    config["log_name"] = config["LOG_NAME"]

    return config


def load_screen_context(context_path):
    """
    Load screen context from a file.

    Args:
        context_path: Path to a file containing screen context information

    Returns:
        String containing the screen context or None if file not found
    """
    import os

    if not context_path:
        return None

    try:
        if os.path.exists(context_path):
            with open(context_path, "r") as f:
                screen_context = f.read().strip()
            print(f"Loaded screen context: {len(screen_context)} characters")
            return screen_context
        else:
            print(f"Screen context file not found: {context_path}")
            return None
    except Exception as e:
        print(f"Error loading screen context: {e}")
        return None


def load_gene_annotations(annotations_path):
    """
    Load gene annotations from a file.

    Args:
        annotations_path: Path to a CSV file with gene annotations

    Returns:
        Dictionary mapping gene IDs to their annotations or None if file not found
    """
    import os
    import pandas as pd

    if not annotations_path:
        return None

    try:
        if os.path.exists(annotations_path):
            annotations_df = pd.read_csv(annotations_path)
            gene_id_column = annotations_df.columns[0]  # Assume first column is gene ID
            features_column = annotations_df.columns[
                1
            ]  # Assume second column is features
            gene_annotations_dict = dict(
                zip(annotations_df[gene_id_column], annotations_df[features_column])
            )
            print(f"Loaded annotations for {len(gene_annotations_dict)} genes")
            return gene_annotations_dict
        else:
            print(f"Gene annotations file not found: {annotations_path}")
            return None
    except Exception as e:
        print(f"Error loading gene annotations: {e}")
        return None
