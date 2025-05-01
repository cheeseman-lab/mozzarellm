import argparse
from utils.cluster_analyzer import analyze_gene_clusters
from utils.cluster_utils import reshape_to_clusters, parse_additional_cols


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

    # Add reshape options
    parser.add_argument(
        "--reshape",
        action="store_true",
        help="Whether to reshape gene-level data to cluster-level before analysis",
    )
    parser.add_argument(
        "--reshape_input",
        type=str,
        default=None,
        help="Path to input gene-level data for reshaping (if different from --input)",
    )
    parser.add_argument(
        "--reshape_output",
        type=str,
        default=None,
        help="Path to output cluster-level data after reshaping (if different from --input)",
    )
    parser.add_argument(
        "--gene_col",
        type=str,
        default="gene_symbol",
        help="Column name for gene identifiers in the reshape step",
    )
    parser.add_argument(
        "--cluster_col",
        type=str,
        default="cluster",
        help="Column name for cluster assignments in the reshape step",
    )
    parser.add_argument(
        "--additional_cols",
        type=str,
        default=None,
        help="Comma-separated list of additional columns to include in the reshape step",
    )

    return parser


def main():
    """Main function to process gene sets or clusters."""
    # Parse command line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Handle reshaping if requested
    input_file = args.input
    if args.reshape:
        reshape_input = args.reshape_input or args.input
        reshape_output = args.reshape_output or args.input

        print(f"Reshaping gene-level data from {reshape_input} to {reshape_output}")

        # Handle tab separator conversion
        sep = "\t" if args.input_sep == "\\t" else args.input_sep

        # Parse additional columns
        additional_cols = parse_additional_cols(args.additional_cols)

        # Perform the reshaping
        reshape_to_clusters(
            input_file=reshape_input,
            output_file=reshape_output,
            sep=sep,
            gene_col=args.gene_col,
            cluster_col=args.cluster_col,
            gene_sep=args.gene_sep,
            additional_cols=additional_cols,
        )

        # Update input file for subsequent analysis
        input_file = reshape_output

    # Use the refactored function for cluster analysis
    if args.mode == "cluster":
        # Process gene clusters
        print(f"Processing clusters in range {args.start or 0} to {args.end or 'end'}")
        results = analyze_gene_clusters(
            input_file=input_file,
            output_file=args.output_file,
            config_path=args.config,
            model_name=args.model,
            custom_prompt_path=args.custom_prompt,
            gene_features_path=args.gene_features,
            screen_info_path=args.screen_info,
            input_sep=args.input_sep,
            gene_column=args.gene_column,
            gene_sep=args.gene_sep,
            batch_size=args.batch_size,
            start_idx=args.start if args.start is not None else 0,
            end_idx=args.end,
        )
        print(f"Analysis completed for {len(results) if results else 0} clusters")

    print("Analysis completed successfully")


if __name__ == "__main__":
    main()
