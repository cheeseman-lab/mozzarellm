"""Report generation for architecture benchmarking.

Produces a human-readable Markdown report comparing architectures across all
benchmark runs, plus CSV/JSON aggregate summaries.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def generate_report(
    records: list[dict[str, Any]],
    config_dict: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Generate a Markdown report comparing architectures.

    Args:
        records: List of per-run record dicts.
        config_dict: The config snapshot dict for reference.
        output_dir: Directory to write report.md and aggregate files.

    Returns:
        Path to the generated report.md.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group records by route
    by_route: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_route[rec["route"]].append(rec)

    # Compute aggregate statistics per route
    route_summaries = {}
    for route_name, route_records in sorted(by_route.items()):
        route_summaries[route_name] = _summarize_route(route_records)

    # Write aggregate_summary.json
    agg_json_path = output_dir / "aggregate_summary.json"
    agg_json_path.write_text(json.dumps(route_summaries, indent=2, default=str), encoding="utf-8")

    # Write aggregate_summary.csv
    agg_csv_path = output_dir / "aggregate_summary.csv"
    _write_aggregate_csv(route_summaries, agg_csv_path)

    # Write report.md
    report_path = output_dir / "report.md"
    report_lines = _build_report_markdown(route_summaries, records, config_dict)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return report_path


def _summarize_route(records: list[dict]) -> dict[str, Any]:
    """Compute aggregate stats for a single route's records."""
    n = len(records)
    if n == 0:
        return {}

    metrics_list = [r.get("metrics", {}) for r in records]

    def _mean(key):
        vals = [m.get(key) for m in metrics_list if m.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    def _rate(key):
        vals = [m.get(key) for m in metrics_list if m.get(key) is not None]
        return sum(1 for v in vals if v) / len(vals) if vals else None

    errors = [r for r in records if r.get("error")]

    # Timing aggregates
    timing_list = [r.get("timing", {}) for r in records]

    def _timing_vals(key):
        return [t.get(key) for t in timing_list if t.get(key) is not None]

    full_run_times = _timing_vals("full_run_time_seconds")
    model_lats = _timing_vals("model_latency_seconds")
    n_calls_vals = _timing_vals("n_api_calls")

    # Derived: seconds per schema-compliant / gene-complete output
    schema_ok_count = sum(1 for m in metrics_list if m.get("schema_compliance"))
    gene_ok_count = sum(1 for m in metrics_list if (m.get("gene_completeness") or 0) >= 1.0)
    total_full = sum(full_run_times) if full_run_times else 0

    summary = {
        "n_runs": n,
        "n_errors": len(errors),
        "error_rate": len(errors) / n,
        "json_parse_success_rate": _rate("json_parse_success"),
        "schema_compliance_rate": _rate("schema_compliance"),
        "cluster_id_match_rate": _rate("cluster_id_exact_match"),
        "gene_completeness_mean": _mean("gene_completeness"),
        "valid_confidence_rate": _rate("valid_confidence_value"),
        "valid_subclass_rate": _rate("valid_subclass_values"),
        "no_duplicate_genes_rate": _rate("no_duplicate_genes"),
        "categories_exclusive_rate": _rate("categories_mutually_exclusive"),
        "mean_latency_s": _mean("latency_seconds"),
        "mean_input_tokens": _mean("input_tokens"),
        "mean_output_tokens": _mean("output_tokens"),
        "mean_cost_usd": _mean("estimated_cost_usd"),
        "total_cost_usd": sum(
            m.get("estimated_cost_usd", 0) or 0 for m in metrics_list
        ),
        # Timing
        "mean_full_run_time_seconds": statistics.mean(full_run_times) if full_run_times else None,
        "median_full_run_time_seconds": statistics.median(full_run_times) if full_run_times else None,
        "p95_full_run_time_seconds": (
            sorted(full_run_times)[int(len(full_run_times) * 0.95)] if len(full_run_times) >= 2 else
            (full_run_times[0] if full_run_times else None)
        ),
        "mean_model_latency_seconds": statistics.mean(model_lats) if model_lats else None,
        "mean_n_api_calls": statistics.mean(n_calls_vals) if n_calls_vals else None,
        "seconds_per_schema_compliant_output": (
            total_full / schema_ok_count if schema_ok_count else None
        ),
        "seconds_per_gene_complete_output": (
            total_full / gene_ok_count if gene_ok_count else None
        ),
    }
    return summary


def _write_aggregate_csv(route_summaries: dict, path: Path) -> None:
    """Write route-level aggregate summaries to CSV."""
    if not route_summaries:
        return

    fieldnames = ["route"] + list(next(iter(route_summaries.values())).keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for route_name, summary in sorted(route_summaries.items()):
            row = {"route": route_name, **summary}
            writer.writerow(row)


def _build_report_markdown(
    route_summaries: dict[str, dict],
    records: list[dict],
    config_dict: dict,
) -> list[str]:
    """Build the Markdown report lines."""
    lines = []
    lines.append(f"# Architecture Benchmark Report")
    lines.append("")
    lines.append(f"**Experiment:** `{config_dict.get('experiment_id', 'unknown')}`")
    lines.append(f"**Model:** `{config_dict.get('model', {}).get('model_name', 'unknown')}`")
    lines.append(f"**Replicates:** {config_dict.get('run', {}).get('num_replicates', '?')}")
    lines.append(f"**Total LLM Analysis Runs:** {len(records)}")
    lines.append(f"**Dry run:** {config_dict.get('run', {}).get('dry_run', False)}")
    lines.append("")
    lines.append("> **Note:** Total LLM Analysis Runs = routes \u00d7 clusters \u00d7 replicates. ")
    lines.append("> Per-screen totals follow the same formula scoped to that screen's clusters.")
    lines.append("")

    # Summary table
    lines.append("## Route Comparison")
    lines.append("")
    lines.append(
        "| Route | Runs | Errors | Parse% | Schema% | Gene Compl. | "
        "Latency (s) | Cost ($) |"
    )
    lines.append(
        "|-------|------|--------|--------|---------|-------------|"
        "-------------|----------|"
    )
    for route_name, s in sorted(route_summaries.items()):
        lines.append(
            f"| {route_name} "
            f"| {s.get('n_runs', 0)} "
            f"| {s.get('n_errors', 0)} "
            f"| {_pct(s.get('json_parse_success_rate'))} "
            f"| {_pct(s.get('schema_compliance_rate'))} "
            f"| {_pct(s.get('gene_completeness_mean'))} "
            f"| {_fmt(s.get('mean_latency_s'), '.1f')} "
            f"| {_fmt(s.get('mean_cost_usd'), '.4f')} |"
        )
    lines.append("")

    # Structural quality
    lines.append("## Structural Quality")
    lines.append("")
    lines.append(
        "| Route | ClusterID Match | Confidence Valid | Subclass Valid | No Duplicates | Exclusive |"
    )
    lines.append(
        "|-------|----------------|-----------------|---------------|--------------|-----------|"
    )
    for route_name, s in sorted(route_summaries.items()):
        lines.append(
            f"| {route_name} "
            f"| {_pct(s.get('cluster_id_match_rate'))} "
            f"| {_pct(s.get('valid_confidence_rate'))} "
            f"| {_pct(s.get('valid_subclass_rate'))} "
            f"| {_pct(s.get('no_duplicate_genes_rate'))} "
            f"| {_pct(s.get('categories_exclusive_rate'))} |"
        )
    lines.append("")

    # Efficiency
    lines.append("## Efficiency")
    lines.append("")
    lines.append("| Route | Mean In Tokens | Mean Out Tokens | Mean Latency (s) | Total Cost ($) |")
    lines.append("|-------|---------------|----------------|-------------------|----------------|")
    for route_name, s in sorted(route_summaries.items()):
        lines.append(
            f"| {route_name} "
            f"| {_fmt(s.get('mean_input_tokens'), '.0f')} "
            f"| {_fmt(s.get('mean_output_tokens'), '.0f')} "
            f"| {_fmt(s.get('mean_latency_s'), '.1f')} "
            f"| {_fmt(s.get('total_cost_usd'), '.4f')} |"
        )
    lines.append("")

    # Timing
    lines.append("## Timing")
    lines.append("")
    lines.append(
        "| Route | Full Run (mean) | Full Run (med) | Full Run (p95) | Model (mean) "
        "| API Calls | s/schema OK | s/gene OK |"
    )
    lines.append(
        "|-------|-----------------|----------------|----------------|------------- "
        "|-----------|-------------|-----------|"
    )
    for route_name, s in sorted(route_summaries.items()):
        lines.append(
            f"| {route_name} "
            f"| {_fmt(s.get('mean_full_run_time_seconds'), '.1f')} "
            f"| {_fmt(s.get('median_full_run_time_seconds'), '.1f')} "
            f"| {_fmt(s.get('p95_full_run_time_seconds'), '.1f')} "
            f"| {_fmt(s.get('mean_model_latency_seconds'), '.1f')} "
            f"| {_fmt(s.get('mean_n_api_calls'), '.1f')} "
            f"| {_fmt(s.get('seconds_per_schema_compliant_output'), '.1f')} "
            f"| {_fmt(s.get('seconds_per_gene_complete_output'), '.1f')} |"
        )
    lines.append("")

    # Overhead comparison
    _route_full = {rn: s.get("mean_full_run_time_seconds") for rn, s in route_summaries.items()}
    baseline_full = _route_full.get("3a")
    if baseline_full is not None:
        lines.append("### Overhead vs Baseline (3a)")
        lines.append("")
        lines.append("| Route | Mean Full Run (s) | Overhead (s) | Overhead (%) |")
        lines.append("|-------|-------------------|-------------|-------------|")
        for route_name in sorted(route_summaries.keys()):
            w = _route_full.get(route_name)
            if w is not None:
                delta = w - baseline_full
                pct = (delta / baseline_full * 100) if baseline_full > 0 else 0
                lines.append(
                    f"| {route_name} "
                    f"| {w:.1f} "
                    f"| {'+' if delta >= 0 else ''}{delta:.1f} "
                    f"| {'+' if pct >= 0 else ''}{pct:.0f}% |"
                )
            else:
                lines.append(f"| {route_name} | -- | -- | -- |")
        lines.append("")

    # Order variant comparison (Phase 2)
    order_records = [r for r in records if r.get("order_variant", "none") != "none"]
    if order_records:
        lines.extend(_build_order_variant_section(route_summaries, records))

    # Per-screen breakdown
    lines.append("## Per-Screen Breakdown")
    lines.append("")
    by_screen: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_screen[rec.get("screen_name", "unknown")].append(rec)

    for screen_name, screen_records in sorted(by_screen.items()):
        lines.append(f"### {screen_name}")
        lines.append(f"- Clusters: {len(set(r['cluster_id'] for r in screen_records))}")
        lines.append(f"- Total LLM Analysis Runs: {len(screen_records)}")
        errors = [r for r in screen_records if r.get("error")]
        lines.append(f"- Errors: {len(errors)}")
        lines.append("")

    # Errors section
    error_records = [r for r in records if r.get("error")]
    if error_records:
        lines.append("## Errors")
        lines.append("")
        for rec in error_records[:20]:  # cap at 20
            lines.append(
                f"- `{rec['run_id']}`: {rec['error'][:200]}"
            )
        if len(error_records) > 20:
            lines.append(f"- ... and {len(error_records) - 20} more")
        lines.append("")

    return lines


def _build_order_variant_section(
    route_summaries: dict[str, dict],
    records: list[dict],
) -> list[str]:
    """Build Markdown section comparing order variants (Phase 2)."""
    lines: list[str] = []
    lines.append("## Order Variant Comparison (Phase 2)")
    lines.append("")

    # Group records by (base_route, order_variant)
    by_base_variant: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        base = rec.get("base_route", rec.get("route", "unknown"))
        variant = rec.get("order_variant", "none")
        by_base_variant[base][variant].append(rec)

    # Comparison table per base route
    for base_route in sorted(by_base_variant.keys()):
        variants = by_base_variant[base_route]
        lines.append(f"### Base Route: {base_route}")
        lines.append("")
        lines.append(
            "| Variant | Hypothesis | Runs | Parse% | Schema% | Gene Compl. | Latency (s) | Cost ($) |"
        )
        lines.append(
            "|---------|-----------|------|--------|---------|-------------|-------------|----------|"
        )

        for variant_name in sorted(variants.keys()):
            vrecs = variants[variant_name]
            hypothesis = vrecs[0].get("order_hypothesis") or "--"
            n = len(vrecs)
            metrics_list = [r.get("metrics", {}) for r in vrecs]

            def _rate(key):
                vals = [m.get(key) for m in metrics_list if m.get(key) is not None]
                return sum(1 for v in vals if v) / len(vals) if vals else None

            def _mean(key):
                vals = [m.get(key) for m in metrics_list if m.get(key) is not None]
                return sum(vals) / len(vals) if vals else None

            lines.append(
                f"| {variant_name} "
                f"| {hypothesis} "
                f"| {n} "
                f"| {_pct(_rate('json_parse_success'))} "
                f"| {_pct(_rate('schema_compliance'))} "
                f"| {_pct(_mean('gene_completeness'))} "
                f"| {_fmt(_mean('latency_seconds'), '.1f')} "
                f"| {_fmt(_mean('estimated_cost_usd'), '.4f')} |"
            )
        lines.append("")

    # Hypothesis summary
    lines.append("### Hypothesis Summary")
    lines.append("")
    hypotheses_seen: dict[str, str] = {}
    for rec in records:
        h = rec.get("order_hypothesis")
        v = rec.get("order_variant", "none")
        if h and h != "--" and h not in hypotheses_seen:
            hypotheses_seen[h] = v

    if hypotheses_seen:
        lines.append("| Hypothesis | Variant | Interpretation |")
        lines.append("|-----------|---------|----------------|")
        for h, v in sorted(hypotheses_seen.items()):
            lines.append(f"| {h} | {v} | _(compare canonical vs {v} above)_ |")
        lines.append("")
    else:
        lines.append("_No hypothesis metadata found in records._")
        lines.append("")

    return lines


def _pct(val: float | None) -> str:
    """Format a 0-1 rate as percentage string."""
    if val is None:
        return "--"
    return f"{val * 100:.0f}%"


def _fmt(val: float | None, fmt: str) -> str:
    """Format a numeric value."""
    if val is None:
        return "--"
    return f"{val:{fmt}}"
