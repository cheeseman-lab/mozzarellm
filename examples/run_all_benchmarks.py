"""Run all benchmarks across multiple models with CSV validation outputs.

This script runs the OPS, DepMap, and Proteomics benchmarks for one or more models,
creates timestamped result directories, and aggregates validation CSVs.

Usage:
    python run_all_benchmarks.py                    # Run with default models
    python run_all_benchmarks.py --models claude-sonnet-4-5-20250929 gpt-4o

Note: For RAG comparison (baseline vs enhanced vs concise), use examples/rag/run_benchmark.py
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Default models to benchmark (latest/most advanced models - Oct 2025)
# One from each provider for comprehensive comparison
DEFAULT_MODELS = [
    "claude-sonnet-4-5-20250929",  # Anthropic - Latest Claude Sonnet (Sept 2025)
    "gemini-2.5-pro",  # Google - Stable Gemini Pro (June 2025)
    "gpt-4o",  # OpenAI - Flagship model
]

# Benchmark configurations (three benchmark datasets)
BENCHMARKS = [
    {"name": "OPS", "dir": "ops", "script": "run_benchmark.py"},
    {"name": "DepMap", "dir": "depmap", "script": "run_benchmark.py"},
    {"name": "Proteomics", "dir": "proteomics", "script": "run_benchmark.py"},
]


def run_benchmark(
    benchmark_dir: str, benchmark_name: str, model: str, output_dir: Path
) -> dict:
    """Run a single benchmark with a specific model using CLI arguments.

    Args:
        benchmark_dir: Directory containing the benchmark script
        benchmark_name: Display name for the benchmark
        model: Model identifier to use
        output_dir: Output directory for this benchmark's results

    Returns:
        Dictionary with success status and paths
    """
    print(f"  Running {benchmark_name}...")

    script_path = Path(benchmark_dir) / "run_benchmark.py"

    if not script_path.exists():
        error_msg = f"Benchmark script not found: {script_path}"
        print(f"    [FAIL] {error_msg}")
        return {
            "model": model,
            "benchmark": benchmark_name,
            "success": False,
            "error": error_msg,
        }

    try:
        # Run the benchmark with CLI args
        result = subprocess.run(
            [
                sys.executable,
                "run_benchmark.py",
                "--model",
                model,
                "--output-dir",
                str(output_dir),
            ],
            cwd=benchmark_dir,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            print("    [FAIL]")
            if result.stderr:
                print(f"    Error: {result.stderr[:200]}")
            return {
                "model": model,
                "benchmark": benchmark_name,
                "success": False,
                "error": result.stderr if result.stderr else "Non-zero return code",
                "output_dir": str(output_dir),
            }
        else:
            print("    [PASS]")
            return {
                "model": model,
                "benchmark": benchmark_name,
                "success": True,
                "output_dir": str(output_dir),
                "quick_csv": str(output_dir / "quick_validation.csv"),
                "detailed_csv": str(output_dir / "detailed_analysis.csv"),
            }

    except subprocess.TimeoutExpired:
        print("    [FAIL] Timeout after 10 minutes")
        return {
            "model": model,
            "benchmark": benchmark_name,
            "success": False,
            "error": "Timeout after 10 minutes",
            "output_dir": str(output_dir),
        }
    except Exception as e:
        print(f"    [FAIL] Error: {e}")
        return {
            "model": model,
            "benchmark": benchmark_name,
            "success": False,
            "error": str(e),
            "output_dir": str(output_dir),
        }


def aggregate_quick_validations(all_results: list, output_path: Path):
    """Aggregate all quick validation CSVs into a master CSV.

    Args:
        all_results: List of result dictionaries
        output_path: Path to save the aggregated CSV

    Returns:
        DataFrame with aggregated results or None if no successful results
    """
    all_dfs = []

    for result in all_results:
        if result["success"] and "quick_csv" in result:
            csv_path = Path(result["quick_csv"])
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                all_dfs.append(df)

    if all_dfs:
        master_df = pd.concat(all_dfs, ignore_index=True)
        # Sort by cluster for side-by-side model comparison
        if "cluster_id" in master_df.columns:
            master_df = master_df.sort_values(by=["cluster_id", "model"])
        master_df.to_csv(output_path, index=False)
        return master_df
    return None


def print_summary(all_results: list, master_csv_path: Path = None):
    """Print a summary of all benchmark results.

    Args:
        all_results: List of result dictionaries
        master_csv_path: Optional path to master CSV for detailed summary
    """
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Print basic status
    for result in all_results:
        status = "[PASS]" if result["success"] else "[FAIL]"
        print(f"{status} {result['benchmark']:<15} {result['model']:<40}")

    # Print error details if any failures
    failed_results = [r for r in all_results if not r["success"]]
    if failed_results:
        print("\nFAILURE DETAILS:")
        print("=" * 80)
        for result in failed_results:
            print(f"\n{result['benchmark']} - {result['model']}")
            print(f"  Error: {result.get('error', 'Unknown error')}")
        print("=" * 80)

    # Print summary from master CSV if available
    if master_csv_path and master_csv_path.exists():
        print("\nVALIDATION SUMMARY:")
        print("=" * 80)
        df = pd.read_csv(master_csv_path)
        # Group by dataset and model, show function match rates
        for (dataset, model), group in df.groupby(["dataset", "model"]):
            matches = group["function_match"].sum()
            total = len(group)
            print(f"{dataset:<15} {model:<40} {matches}/{total} functions matched")
        print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run all benchmarks across multiple models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Models to benchmark (default: {', '.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        help="Base directory for outputs (default: benchmark_results/run_TIMESTAMP)",
    )

    args = parser.parse_args()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_base:
        base_output_dir = Path(args.output_base)
    else:
        base_output_dir = (
            Path(__file__).parent / "benchmark_results" / f"run_{timestamp}"
        )

    base_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(
        f"Running benchmarks for {len(args.models)} model(s) across {len(BENCHMARKS)} benchmark(s)"
    )
    print(f"Models: {', '.join(args.models)}")
    print(f"Output directory: {base_output_dir}")
    print(f"{'=' * 80}")

    all_results = []

    # Run each benchmark for each model
    for i, model in enumerate(args.models):
        print(f"\n{'#' * 80}")
        print(f"MODEL {i + 1}/{len(args.models)}: {model}")
        print(f"{'#' * 80}")

        for j, benchmark in enumerate(BENCHMARKS):
            print(f"\n>>> Benchmark {j + 1}/{len(BENCHMARKS)}: {benchmark['name']}")

            # Create benchmark-specific output directory
            model_safe = model.replace("/", "_")
            benchmark_output_dir = base_output_dir / f"{benchmark['name']}_{model_safe}"
            benchmark_output_dir.mkdir(parents=True, exist_ok=True)

            # Run the benchmark
            result = run_benchmark(
                benchmark["dir"], benchmark["name"], model, benchmark_output_dir
            )
            all_results.append(result)

    # Aggregate validation CSVs
    print("\n" + "=" * 80)
    print("AGGREGATING RESULTS")
    print("=" * 80)

    # Aggregate quick validations
    master_csv_path = base_output_dir / "master_validation.csv"
    master_df = aggregate_quick_validations(all_results, master_csv_path)

    if master_df is not None:
        print(f"Master validation CSV saved to: {master_csv_path}")
    else:
        print("WARNING: No successful results to aggregate")
        master_csv_path = None

    # Aggregate detailed analysis CSVs
    all_detailed_dfs = []
    for result in all_results:
        if result["success"] and "detailed_csv" in result:
            detailed_csv_path = Path(result["detailed_csv"])
            if detailed_csv_path.exists():
                df = pd.read_csv(detailed_csv_path)
                all_detailed_dfs.append(df)

    if all_detailed_dfs:
        master_detailed_df = pd.concat(all_detailed_dfs, ignore_index=True)
        # Sort by cluster and gene for side-by-side model comparison
        if (
            "cluster_id" in master_detailed_df.columns
            and "gene" in master_detailed_df.columns
        ):
            master_detailed_df = master_detailed_df.sort_values(
                by=["cluster_id", "gene", "model"]
            )
        master_detailed_path = base_output_dir / "master_detailed_analysis.csv"
        master_detailed_df.to_csv(master_detailed_path, index=False)
        print(f"Master detailed analysis CSV saved to: {master_detailed_path}")

    # Save detailed results JSON
    results_json_path = base_output_dir / "results.json"
    with open(results_json_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "models": args.models,
                "benchmarks": [b["name"] for b in BENCHMARKS],
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"Detailed results saved to: {results_json_path}")

    # Print summary
    print_summary(all_results, master_csv_path)

    # Exit with error if any benchmark failed
    if any(not r["success"] for r in all_results):
        print("\nWARNING: Some benchmarks failed")
        sys.exit(1)
    else:
        print("\nAll benchmarks completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
