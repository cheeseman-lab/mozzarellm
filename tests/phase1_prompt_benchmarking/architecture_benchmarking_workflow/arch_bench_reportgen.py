"""Report generation for architecture benchmarking.

Produces a human-readable Markdown report comparing architectures across all
benchmark runs, plus CSV/JSON aggregate summaries.
"""

from __future__ import annotations

import csv
import json
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

    return {
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
    }


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
    lines.append(f"**Total runs:** {len(records)}")
    lines.append(f"**Dry run:** {config_dict.get('run', {}).get('dry_run', False)}")
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

    # Per-screen breakdown
    lines.append("## Per-Screen Breakdown")
    lines.append("")
    by_screen: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_screen[rec.get("screen_name", "unknown")].append(rec)

    for screen_name, screen_records in sorted(by_screen.items()):
        lines.append(f"### {screen_name}")
        lines.append(f"- Clusters: {len(set(r['cluster_id'] for r in screen_records))}")
        lines.append(f"- Total runs: {len(screen_records)}")
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


def _pct(val: float | None) -> str:
    """Format a 0-1 rate as percentage string."""
    if val is None:
        return "—"
    return f"{val * 100:.0f}%"


def _fmt(val: float | None, fmt: str) -> str:
    """Format a numeric value."""
    if val is None:
        return "—"
    return f"{val:{fmt}}"
