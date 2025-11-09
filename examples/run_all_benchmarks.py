"""Run all benchmarks across multiple models.

This script runs the OPS, DepMap, and Proteomics benchmarks for one or more models
and aggregates the validation results for easy comparison.

Usage:
    python run_all_benchmarks.py                    # Run with default models
    python run_all_benchmarks.py --models o1 claude-3-7-sonnet-20250219
    python run_all_benchmarks.py --output custom_results.json
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Default models to benchmark (latest/most advanced models)
DEFAULT_MODELS = [
    "o1",  # OpenAI's reasoning model
    "claude-3-7-sonnet-20250219",  # Latest Claude Sonnet
    "gemini-2.5-pro-preview-03-25",  # Latest Gemini Pro
    "gpt-4o",  # OpenAI's flagship model
]

# Benchmark configurations
BENCHMARKS = [
    {"name": "OPS", "dir": "ops", "script": "run_benchmark.py"},
    {"name": "DepMap", "dir": "depmap", "script": "run_benchmark.py"},
    {"name": "Proteomics", "dir": "proteomics", "script": "run_benchmark.py"},
]


def run_benchmark(benchmark_dir: str, model: str) -> dict:
    """Run a single benchmark with a specific model.

    Args:
        benchmark_dir: Directory containing the benchmark script
        model: Model identifier to use

    Returns:
        Dictionary with validation results
    """
    print(f"\n{'=' * 70}")
    print(f"Running {benchmark_dir} benchmark with {model}")
    print(f"{'=' * 70}\n")

    # Modify the benchmark script to use the specified model
    script_path = Path(benchmark_dir) / "run_benchmark.py"
    script_content = script_path.read_text()

    # Replace MODEL line
    original_model_line = None
    for line in script_content.split("\n"):
        if line.startswith('MODEL = "'):
            original_model_line = line
            break

    if original_model_line:
        modified_content = script_content.replace(
            original_model_line, f'MODEL = "{model}"  # Modified by run_all_benchmarks.py'
        )
        script_path.write_text(modified_content)

    try:
        # Run the benchmark
        result = subprocess.run(
            [sys.executable, "run_benchmark.py"],
            cwd=benchmark_dir,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        # Restore original model line
        if original_model_line:
            script_path.write_text(script_content)

        # Parse validation results from output
        output = result.stdout
        validation_results = parse_validation_output(output)

        # Add model and benchmark info
        validation_results["model"] = model
        validation_results["benchmark"] = benchmark_dir
        validation_results["success"] = result.returncode == 0

        if result.returncode != 0:
            validation_results["error"] = result.stderr

        return validation_results

    except subprocess.TimeoutExpired:
        print("WARNING: Benchmark timed out after 10 minutes")
        return {
            "model": model,
            "benchmark": benchmark_dir,
            "success": False,
            "error": "Timeout after 10 minutes",
        }
    except Exception as e:
        print(f"WARNING: Error running benchmark: {e}")
        return {
            "model": model,
            "benchmark": benchmark_dir,
            "success": False,
            "error": str(e),
        }
    finally:
        # Ensure original content is restored
        if original_model_line:
            script_path.write_text(script_content)


def parse_validation_output(output: str) -> dict:
    """Parse validation results from benchmark output.

    Args:
        output: Stdout from benchmark script

    Returns:
        Dictionary with function_matches and genes_classified metrics
    """
    results = {}

    for line in output.split("\n"):
        if "Function matches:" in line:
            # Extract: "Function matches: 6/6 (100.0%)"
            parts = line.split("Function matches:")[1].strip()
            count_part = parts.split("(")[0].strip()
            results["function_matches"] = count_part

        elif "Genes classified:" in line:
            # Extract: "Genes classified: 7/7 (100.0%)"
            parts = line.split("Genes classified:")[1].strip()
            count_part = parts.split("(")[0].strip()
            results["genes_classified"] = count_part

    return results


def print_summary(all_results: list):
    """Print a summary table of all results.

    Args:
        all_results: List of result dictionaries
    """
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Model':<40} {'Benchmark':<15} {'Functions':<12} {'Genes':<12} {'Status'}")
    print("-" * 80)

    for result in all_results:
        model = result["model"]
        benchmark = result["benchmark"].upper()
        status = "PASS" if result["success"] else "FAIL"

        if result["success"]:
            func = result.get("function_matches", "N/A")
            genes = result.get("genes_classified", "N/A")
        else:
            func = "N/A"
            genes = "N/A"

        print(f"{model:<40} {benchmark:<15} {func:<12} {genes:<12} {status}")

    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run all benchmarks across multiple models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Models to benchmark (default: {', '.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save JSON results (default: benchmark_results/results_TIMESTAMP.json)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file",
    )

    args = parser.parse_args()

    print(f"\n{'=' * 80}")
    print(
        f"Running benchmarks for {len(args.models)} model(s) across {len(BENCHMARKS)} benchmark(s)"
    )
    print(f"Models: {', '.join(args.models)}")
    print(f"{'=' * 80}")

    all_results = []

    # Run each benchmark for each model
    for model in args.models:
        for benchmark in BENCHMARKS:
            result = run_benchmark(benchmark["dir"], model)
            all_results.append(result)

    # Print summary
    print_summary(all_results)

    # Save results unless --no-save is specified
    if not args.no_save:
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            # Default: save to benchmark_results/ with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = Path(__file__).parent / "benchmark_results"
            results_dir.mkdir(exist_ok=True)
            output_path = results_dir / f"results_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    # Exit with error if any benchmark failed
    if any(not r["success"] for r in all_results):
        print("\nWARNING: Some benchmarks failed")
        sys.exit(1)
    else:
        print("\nAll benchmarks completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
