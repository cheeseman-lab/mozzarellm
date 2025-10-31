"""Run all benchmarks across multiple models.

This script runs the OPS, DepMap, and Proteomics benchmarks for one or more models
and aggregates the validation results for easy comparison.

Usage:
    python run_all_benchmarks.py                    # Run with default models
    python run_all_benchmarks.py --models o4-mini claude-sonnet-4-5-20250929
    python run_all_benchmarks.py --output custom_results.json
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Default models to benchmark (latest/most advanced models - Oct 2025)
# One from each provider for comprehensive comparison
DEFAULT_MODELS = [
    "claude-sonnet-4-5-20250929",  # Anthropic - Latest Claude Sonnet (Sept 2025)
    "gemini-2.5-pro",  # Google - Stable Gemini Pro (June 2025)
    "gpt-4o",  # OpenAI - Flagship model
]

# Benchmark configurations
BENCHMARKS = [
    {"name": "OPS", "dir": "ops", "script": "run_benchmark.py"},
    {"name": "DepMap", "dir": "depmap", "script": "run_benchmark.py"},
    {"name": "Proteomics", "dir": "proteomics", "script": "run_benchmark.py"},
]


def run_benchmark(benchmark_dir: str, model: str, verbose: bool = False) -> dict:
    """Run a single benchmark with a specific model.

    Args:
        benchmark_dir: Directory containing the benchmark script
        model: Model identifier to use
        verbose: If True, print detailed output including stdout/stderr

    Returns:
        Dictionary with validation results
    """
    print(f"\n{'=' * 70}")
    print(f"Running {benchmark_dir} benchmark with {model}")
    print(f"{'=' * 70}\n")

    # Modify the benchmark script to use the specified model
    script_path = Path(benchmark_dir) / "run_benchmark.py"

    if not script_path.exists():
        error_msg = f"Benchmark script not found: {script_path}"
        print(f"✗ {error_msg}")
        return {
            "model": model,
            "benchmark": benchmark_dir,
            "success": False,
            "error": error_msg,
        }

    print(f"📄 Reading benchmark script: {script_path}")
    script_content = script_path.read_text()

    # Replace MODEL line
    original_model_line = None
    for line in script_content.split("\n"):
        if line.startswith('MODEL = "'):
            original_model_line = line
            break

    if original_model_line:
        print(f"🔧 Modifying MODEL from: {original_model_line.strip()}")
        print(f"   to: MODEL = \"{model}\"")
        modified_content = script_content.replace(
            original_model_line, f'MODEL = "{model}"  # Modified by run_all_benchmarks.py'
        )
        script_path.write_text(modified_content)
    else:
        print("⚠️  Warning: Could not find MODEL line in script")

    try:
        # Run the benchmark
        print("🚀 Executing benchmark...")
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
            print("✓ Restored original script content")

        # Parse validation results from output
        output = result.stdout
        stderr = result.stderr

        print(f"\n📊 Benchmark completed with return code: {result.returncode}")

        if verbose or result.returncode != 0:
            print("\n" + "-" * 70)
            print("STDOUT:")
            print("-" * 70)
            print(output if output else "(no output)")

        if stderr:
            print("\n" + "-" * 70)
            print("STDERR:")
            print("-" * 70)
            print(stderr)
            print("-" * 70)

        validation_results = parse_validation_output(output)

        # Add model and benchmark info
        validation_results["model"] = model
        validation_results["benchmark"] = benchmark_dir
        validation_results["success"] = result.returncode == 0
        validation_results["stdout"] = output
        validation_results["stderr"] = stderr

        if result.returncode != 0:
            validation_results["error"] = stderr if stderr else "Non-zero return code"
            print("✗ Benchmark FAILED")
        else:
            print("✓ Benchmark SUCCEEDED")
            print(f"   Function matches: {validation_results.get('function_matches', 'N/A')}")
            print(f"   Genes classified: {validation_results.get('genes_classified', 'N/A')}")

        return validation_results

    except subprocess.TimeoutExpired:
        print("⚠️  Benchmark timed out after 10 minutes")
        if original_model_line:
            script_path.write_text(script_content)
        return {
            "model": model,
            "benchmark": benchmark_dir,
            "success": False,
            "error": "Timeout after 10 minutes",
        }
    except Exception as e:
        print(f"⚠️  Error running benchmark: {e}")
        if original_model_line:
            script_path.write_text(script_content)
        return {
            "model": model,
            "benchmark": benchmark_dir,
            "success": False,
            "error": str(e),
        }
    finally:
        # Ensure original content is restored
        if original_model_line:
            try:
                script_path.write_text(script_content)
            except Exception as e:
                print(f"⚠️  Warning: Could not restore original script: {e}")


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
        status = "✓" if result["success"] else "✗"

        if result["success"]:
            func = result.get("function_matches", "N/A")
            genes = result.get("genes_classified", "N/A")
        else:
            func = "N/A"
            genes = "N/A"

        print(f"{model:<40} {benchmark:<15} {func:<12} {genes:<12} {status}")

    print("=" * 80)

    # Print error details if any failures
    failed_results = [r for r in all_results if not r["success"]]
    if failed_results:
        print("\nFAILURE DETAILS:")
        print("=" * 80)
        for result in failed_results:
            print(f"\n{result['benchmark'].upper()} - {result['model']}")
            print(f"  Error: {result.get('error', 'Unknown error')}")
            if result.get('stderr'):
                print(f"  Stderr preview: {result['stderr'][:200]}...")
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
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output including full stdout from benchmarks",
    )

    args = parser.parse_args()

    print(f"\n{'=' * 80}")
    print(
        f"Running benchmarks for {len(args.models)} model(s) across {len(BENCHMARKS)} benchmark(s)"
    )
    print(f"Models: {', '.join(args.models)}")
    if args.verbose:
        print("Verbose mode: ENABLED")
    print(f"{'=' * 80}")

    all_results = []

    # Run each benchmark for each model
    for i, model in enumerate(args.models):
        print(f"\n{'#' * 80}")
        print(f"MODEL {i+1}/{len(args.models)}: {model}")
        print(f"{'#' * 80}")
        for j, benchmark in enumerate(BENCHMARKS):
            print(f"\n>>> Benchmark {j+1}/{len(BENCHMARKS)}: {benchmark['name']}")
            result = run_benchmark(benchmark["dir"], model, verbose=args.verbose)
            all_results.append(result)

            # Print immediate status
            status_emoji = "✓" if result["success"] else "✗"
            print(f"{status_emoji} {benchmark['name']} with {model}: {'SUCCESS' if result['success'] else 'FAILED'}")
            if not result["success"]:
                print(f"   Error: {result.get('error', 'Unknown error')}")

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
        print(f"\n✓ Results saved to: {output_path}")

    # Exit with error if any benchmark failed
    if any(not r["success"] for r in all_results):
        print("\n⚠️  Some benchmarks failed")
        sys.exit(1)
    else:
        print("\n✓ All benchmarks completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
