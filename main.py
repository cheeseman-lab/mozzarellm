import argparse
import pandas as pd
import json
import os
import logging
import re
import time
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

# Import utility functions
from utils.openai_query import openai_chat
from utils.anthropic_query import anthropic_chat
from utils.gemini_query import query_genai_model
from utils.perplexity_query import perplexity_chat
from utils.server_model_query import server_model_chat
from utils.prompt_factory import (
    make_gene_analysis_prompt,
    make_cluster_analysis_prompt,
    make_batch_cluster_analysis_prompt,
)
from utils.llm_analysis_utils import (
    process_analysis,
    process_cluster_analysis,
    process_batch_cluster_analysis,
    save_progress,
    save_cluster_analysis,
)
from utils.logging_utils import get_model_logger

# Import constants
import constant

# Load environment variables
load_dotenv()


def setup_argument_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="Process gene clusters with LLMs.")
    parser.add_argument("--config", type=str, required=True, help="Config file for LLM")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["gene_set", "cluster"],
        default="cluster",
        help="Analysis mode: gene_set (original) or cluster (default)",
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input csv with gene clusters"
    )
    parser.add_argument(
        "--input_sep", type=str, required=True, help="Separator for input csv"
    )
    parser.add_argument(
        "--set_index",
        type=str,
        default="cluster_id",
        help="Column name for cluster index",
    )
    parser.add_argument(
        "--gene_column", type=str, required=True, help="Column name for gene set"
    )
    parser.add_argument(
        "--gene_sep", type=str, required=True, help="Separator for gene set"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of clusters to analyze in one batch (cluster mode only)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start index for cluster range (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index for cluster range (default: process all)",
    )
    parser.add_argument(
        "--gene_features",
        type=str,
        default=None,
        help="Path to csv with gene features if needed for prompt",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output file (no extension)",
    )

    # Legacy arguments for gene_set mode
    parser.add_argument(
        "--initialize",
        action="store_true",
        help="Initialize the output file with columns (gene_set mode)",
    )
    parser.add_argument(
        "--run_contaminated",
        action="store_true",
        help="Run the pipeline for contaminated gene sets (gene_set mode)",
    )

    return parser


def load_config(config_file):
    """Load and validate configuration from a JSON file."""
    with open(config_file) as json_file:
        config = json.load(json_file)

    # Extract required configuration values
    context = config["CONTEXT"]
    model = config["MODEL"]
    temperature = config["TEMP"]
    max_tokens = config["MAX_TOKENS"]

    # Get rate from API settings if available
    if "API_SETTINGS" in config:
        for provider, settings in config["API_SETTINGS"].items():
            if any(
                model.startswith(m.split("-")[0]) for m in settings.get("models", [])
            ):
                rate_per_token = settings.get("rate_per_token", 0.00001)
                break
        else:
            rate_per_token = config.get("RATE_PER_TOKEN", 0.00001)  # Default fallback
    else:
        rate_per_token = config.get("RATE_PER_TOKEN", 0.00001)  # Default fallback

    dollar_limit = config.get("DOLLAR_LIMIT", 10.0)  # Default fallback
    log_name = config.get("LOG_NAME", "analysis")

    # Check for custom prompt
    custom_prompt_file = config.get("CUSTOM_PROMPT_FILE")
    customized_prompt = None
    if custom_prompt_file and os.path.isfile(custom_prompt_file):
        with open(custom_prompt_file, "r") as f:
            customized_prompt = f.read()
            if not customized_prompt or len(customized_prompt) < 2:
                logging.warning(
                    "Customized prompt file exists but is empty or too short"
                )
                customized_prompt = None

    return {
        "context": context,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "rate_per_token": rate_per_token,
        "dollar_limit": dollar_limit,
        "log_name": log_name,
        "customized_prompt": customized_prompt,
    }


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
        if model.startswith("gpt"):
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
        elif (
            model.startswith("deepseek")
            or model.startswith("llama")
            or model.startswith("mistral")
        ):
            logger.info(f"Using Perplexity API with {model}")
            analysis, error_message = perplexity_chat(
                context, prompt, model, temperature, max_tokens, log_file
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


def process_gene_set(df, config, args, logger, gene_features_dict=None):
    """
    Process gene sets using the original pipeline.

    Args:
        df: DataFrame containing gene sets to analyze
        config: Configuration dictionary
        args: Command line arguments
        logger: Logger instance
        gene_features_dict: Optional dictionary with gene features

    Returns:
        analysis_dict: Dictionary containing analysis results
    """
    analysis_dict = {}

    # Extract needed config values
    context = config["context"]
    model = config["model"]
    temperature = config["temperature"]
    max_tokens = config["max_tokens"]
    rate_per_token = config["rate_per_token"]
    dollar_limit = config["dollar_limit"]

    # Set up paths and column names
    gene_column = args.gene_column
    gene_sep = args.gene_sep
    out_file = args.output_file

    # Create column prefix based on model name
    if "-" in model:
        column_prefix = "_".join(model.split("-")[:2])
    else:
        column_prefix = model.replace(":", "_")

    # Set random seed for reproducibility
    seed = constant.SEED

    # Set up logging file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{config['log_name']}_{timestamp}.log"

    i = 0  # Used for tracking progress and saving the file
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing gene sets"):
        # Only process None rows
        if pd.notna(row.get(f"{column_prefix} Analysis")):
            continue

        gene_data = row[gene_column]
        # If gene_data is not a string, then skip
        if not isinstance(gene_data, str):
            logger.warning(f"Gene set {idx} is not a string, skipping")
            continue

        genes = gene_data.split(gene_sep)

        # Check if gene set is too large
        if len(genes) > constant.MAX_GENES_PER_ANALYSIS:
            logger.warning(f"Gene set {idx} is too big ({len(genes)} genes), skipping")
            continue

        try:
            # Create prompt
            prompt = make_gene_analysis_prompt(genes, gene_features_dict)

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
                seed,
            )

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
                df.loc[idx, f"{column_prefix} Name"] = llm_name
                df.loc[idx, f"{column_prefix} Analysis"] = llm_analysis
                df.loc[idx, f"{column_prefix} Score"] = llm_score_value

                # Save raw response
                analysis_dict[f"{idx}_{column_prefix}"] = analysis

                # Log success
                logger.info(f"Success for {idx} {column_prefix}.")
                if isinstance(error, str) and "fingerprint" in error:
                    logger.info(f"Model fingerprint for {idx}: {error}")
            else:
                logger.error(f"Error for query gene set {idx}: {error}")

        except Exception as e:
            logger.error(f"Error for {idx}: {e}")
            continue

        # Save progress periodically
        i += 1
        if i % 10 == 0:
            # Bin scores into confidence categories
            bins = constant.SCORE_BIN_RANGES
            labels = constant.SCORE_BIN_LABELS

            df[f"{column_prefix} Score bins"] = pd.cut(
                df[f"{column_prefix} Score"], bins=bins, labels=labels
            )
            save_progress(df, analysis_dict, out_file)
            logger.info(f"Saved progress for {i} gene sets")

    # Save the final file
    # Bin scores into confidence categories
    bins = constant.SCORE_BIN_RANGES
    labels = constant.SCORE_BIN_LABELS

    df[f"{column_prefix} Score bins"] = pd.cut(
        df[f"{column_prefix} Score"], bins=bins, labels=labels
    )
    save_progress(df, analysis_dict, out_file)

    return analysis_dict


def process_clusters(df, config, args, logger, gene_features_dict=None):
    """Process gene clusters to identify pathways and novel members."""
    clusters_dict = {}

    # Extract needed config values
    context = config["context"]
    model = config["model"]
    temperature = config["temperature"]
    max_tokens = config["max_tokens"]
    rate_per_token = config["rate_per_token"]
    dollar_limit = config["dollar_limit"]

    # Set up logging file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{config['log_name']}_{timestamp}.log"

    # Determine whether to process in batches
    batch_size = args.batch_size
    gene_column = args.gene_column
    gene_sep = args.gene_sep
    out_file = args.output_file

    if batch_size <= 1:
        # Process one cluster at a time
        for idx, row in tqdm(
            df.iterrows(), total=df.shape[0], desc="Processing clusters"
        ):
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
            prompt = make_cluster_analysis_prompt(idx, genes, gene_features_dict)
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
                cluster_result = process_cluster_analysis(analysis)

                # Store the result
                clusters_dict[str(idx)] = cluster_result
                logger.info(f"Success for cluster {idx}")
            else:
                logger.error(f"Error for cluster {idx}: {error}")

            # Save progress every 5 clusters
            if len(clusters_dict) % 5 == 0:
                save_cluster_analysis(clusters_dict, out_file)
                logger.info(f"Saved progress for {len(clusters_dict)} clusters")

    else:
        # Process clusters in batches
        batch_clusters = {}
        clusters_processed = 0

        # Use enumerate to safely track the last iteration
        total_clusters = len(df)
        for i, (idx, row) in enumerate(
            tqdm(df.iterrows(), total=total_clusters, desc="Processing cluster batches")
        ):
            is_last_cluster = i == total_clusters - 1

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
                        batch_clusters, gene_features_dict
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
                        batch_results = process_batch_cluster_analysis(analysis)

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
                                    f'Failed to extract results for clusters: {", ".join(missing_clusters)}'
                                )
                        else:
                            logger.error(
                                "Failed to extract any cluster results from batch response"
                            )
                            # Optionally save the raw response for debugging
                            with open(f"failed_batch_{time.time()}.txt", "w") as f:
                                f.write(analysis)

                        # If batch processing failed and batch size > 2, consider processing clusters individually
                        if batch_size_actual > 2:
                            logger.info(
                                "Batch processing failed, processing clusters individually"
                            )
                            for cluster_id, genes_list in batch_clusters.items():
                                individual_prompt = make_cluster_analysis_prompt(
                                    cluster_id, genes_list, gene_features_dict
                                )
                                individual_analysis, individual_error = query_llm(
                                    context,
                                    individual_prompt,
                                    model,
                                    temperature,
                                    max_tokens,
                                    rate_per_token,
                                    log_file,
                                    dollar_limit,
                                    constant.SEED,
                                )

                                if individual_analysis:
                                    cluster_result = process_cluster_analysis(
                                        individual_analysis
                                    )
                                    clusters_dict[cluster_id] = cluster_result
                                    clusters_processed += 1
                                    logger.info(
                                        f"Successfully processed individual cluster {cluster_id}"
                                    )
                                else:
                                    logger.error(
                                        f"Error processing individual cluster {cluster_id}: {individual_error}"
                                    )

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


def main():
    """Main function to process gene sets or clusters."""
    # Parse command line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create logger
    logger = get_model_logger(config["model"], args.start, args.end)

    # Handle tab separator conversion
    input_sep = "\t" if args.input_sep == "\\t" else args.input_sep

    # Load the data
    try:
        raw_df = pd.read_csv(args.input, sep=input_sep, index_col=args.set_index)
        print(
            f"Loaded data with {len(raw_df)} rows and columns: {list(raw_df.columns)}"
        )
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        return

    # Handle start and end indices
    start_idx = args.start if args.start is not None else 0
    end_idx = args.end if args.end is not None else len(raw_df)

    if start_idx < 0 or start_idx >= len(raw_df):
        logger.error(
            f"Invalid start index {start_idx}, must be between 0 and {len(raw_df)-1}"
        )
        return

    if end_idx <= start_idx or end_idx > len(raw_df):
        logger.error(
            f"Invalid end index {end_idx}, must be between {start_idx+1} and {len(raw_df)}"
        )
        return

    print(f"Processing rows from index {start_idx} to {end_idx-1}")

    # Select the range of entries to process
    try:
        df = raw_df.iloc[start_idx:end_idx].copy()
    except Exception as e:
        logger.error(f"Error selecting data range: {e}")
        return

    # Load gene features if provided
    gene_features_dict = load_gene_features(args.gene_features)

    # Run the appropriate analysis mode
    if args.mode == "cluster":
        # Process gene clusters
        print(f"Processing {len(df)} clusters in range {start_idx}-{end_idx}")
        results = process_clusters(df, config, args, logger, gene_features_dict)
        
        # Reset index to make cluster_id a column for merging
        original_df = raw_df.reset_index()
        
        # Save the results with the original dataframe
        save_cluster_analysis(results, args.output_file, original_df=original_df)
        
        print(f"Analysis completed for {len(results)} clusters")
    else:
        # Gene set analysis mode remains unchanged
        print(f"Processing {len(df)} gene sets in range {start_idx}-{end_idx}")

        # Create column prefix for this model
        if "-" in config["model"]:
            column_prefix = "_".join(config["model"].split("-")[:2])
        else:
            column_prefix = config["model"].replace(":", "_")

        # Initialize columns if needed
        if args.initialize:
            df[f"{column_prefix} Name"] = None
            df[f"{column_prefix} Analysis"] = None
            df[f"{column_prefix} Score"] = float("-inf")
            print(f"Initialized output columns with prefix '{column_prefix}'")

        # Check how many entries need processing
        print(
            f"Found {df[f'{column_prefix} Analysis'].isna().sum()} gene sets to analyze"
        )

        # Process gene sets
        process_gene_set(df, config, args, logger, gene_features_dict)

    print("Analysis completed successfully")
    

if __name__ == "__main__":
    main()
