import argparse
import pandas as pd
import json
import logging
import time
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

# Import utility functions
from utils.openai_query import openai_chat
from utils.anthropic_query import anthropic_chat
from utils.gemini_query import query_genai_model
from utils.server_model_query import server_model_chat
from utils.prompt_factory import (
    make_cluster_analysis_prompt,
    make_batch_cluster_analysis_prompt,
)
from utils.llm_analysis_utils import (
    process_cluster_response,
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
        "--model",
        type=str,
        default=None,
        help="Override model specified in config file",
    )
    parser.add_argument(
        "--custom_prompt",
        type=str,
        default=None,
        help="Path to custom prompt template file",
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input csv with gene clusters"
    )
    parser.add_argument(
        "--input_sep", type=str, required=True, help="Separator for input csv"
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
    parser.add_argument(
        "--screen_info",
        type=str,
        default=None,
        help="Path to file containing information about the OPS screen context",
    )

    return parser


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
    df, config, args, logger, gene_features_dict=None, screen_info=None
):
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
            prompt = make_cluster_analysis_prompt(
                idx,
                genes,
                gene_features_dict,
                screen_info,
                template_path=args.custom_prompt,
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
                    # Create batch prompt - now with screen_info
                    prompt = make_batch_cluster_analysis_prompt(
                        batch_clusters,
                        gene_features_dict,
                        screen_info,
                        template_path=args.custom_prompt,
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
                                    cluster_id,
                                    genes_list,
                                    gene_features_dict,
                                    screen_info,
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
                                    cluster_result = process_cluster_response(
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
    config = load_config(args.config, args.model)

    # Create logger
    logger = get_model_logger(args.model, args.start, args.end)

    # Handle tab separator conversion
    input_sep = "\t" if args.input_sep == "\\t" else args.input_sep

    # Load the data
    try:
        raw_df = pd.read_csv(args.input, sep=input_sep)
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

    # Load screen information if provided
    screen_info = load_screen_info(args.screen_info)

    # Run the appropriate analysis mode
    if args.mode == "cluster":
        # Process gene clusters
        print(f"Processing {len(df)} clusters in range {start_idx}-{end_idx}")
        results = process_clusters(
            df, config, args, logger, gene_features_dict, screen_info
        )

        # Reset index to make cluster_id a column for merging
        original_df = raw_df.reset_index()

        # Save the results with the original dataframe
        save_cluster_analysis(results, args.output_file, original_df=original_df)

        print(f"Analysis completed for {len(results)} clusters")

    print("Analysis completed successfully")


if __name__ == "__main__":
    main()
