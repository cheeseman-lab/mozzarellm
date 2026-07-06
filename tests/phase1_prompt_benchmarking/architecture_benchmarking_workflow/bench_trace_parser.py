"""
Trace-to-CSV Parser
--------------------
Reads parsed_output from benchmark trace JSONs and produces gene-level
prediction CSVs aligned with the benchmark_clusters.csv row structure.

One CSV per route per experiment. Filename:
  {experiment_id}_{route}.csv              (if overwrite_outputs is True)
  {experiment_id}_{route}_{YYYYMMDD}.csv   (if overwrite_outputs is False)

Usage:
    python -m tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_trace_parser \
        --traces-dir <path_to_traces_dir> \
        --output-dir <path_to_output_dir> \
        [--overwrite]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Filename parser
# ---------------------------------------------------------------------------

# Trace filenames use double-underscore separators:
#   {experiment_id}__{route}__{screen_name}__cluster_{cluster_id}__rep_{replicate}.json
# Route names with underscores (e.g. 3a_mcp) must be matched greedily before
# shorter variants to avoid ambiguity.
TRACE_FILENAME_RE = re.compile(
    r"^(?P<experiment_id>.+?)__"
    r"(?P<route>3[abc](?:(?!__).)*)"
    r"__(?P<screen_name>.+?)__"
    r"cluster_(?P<cluster_id>.+?)__"
    r"rep_(?P<replicate>\d+)\.json$"
)


def parse_trace_filename(path: Path) -> dict:
    """Extract metadata from a trace filename.

    Returns a dict with keys: experiment_id, route, screen_name, cluster_id,
    replicate, filename_parse_status.
    """
    match = TRACE_FILENAME_RE.match(path.name)
    if not match:
        return {
            "experiment_id": None,
            "route": None,
            "screen_name": None,
            "cluster_id": None,
            "replicate": None,
            "filename_parse_status": "failed",
        }

    data = match.groupdict()
    data["replicate"] = int(data["replicate"])
    data["filename_parse_status"] = "success"
    return data


# ---------------------------------------------------------------------------
# Parsed-output -> gene rows
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "screen_name",
    "cluster_id",
    "gene_symbol",
    "route",
    "replicate",
    "run_id",
    "predicted_class",
    "predicted_subclass",
    "rationale",
    "evidence",
    "pathway",
    "pathway_confidence",
    "source_trace_path",
    # Phase 2 order benchmarking metadata
    "base_route",
    "order_variant",
    "order_hypothesis",
    "component_order",
]


def _extract_gene_rows(
    parsed_output: dict[str, Any],
    meta: dict[str, Any],
    source_path: str,
) -> list[dict]:
    """Convert a single trace's parsed_output into flat gene-level rows."""
    if not parsed_output:
        return []

    pathway = parsed_output.get("dominant_process", "")
    pathway_confidence = parsed_output.get("pathway_confidence", "")
    cluster_id = meta["cluster_id"]
    screen_name = meta["screen_name"]
    route = meta["route"]
    replicate = meta["replicate"]
    run_id = meta["run_id"]

    rows: list[dict] = []

    # --- Established genes (simple string list) ---
    for gene in parsed_output.get("established_genes", []):
        rows.append(
            {
                "screen_name": screen_name,
                "cluster_id": cluster_id,
                "gene_symbol": gene,
                "route": route,
                "replicate": replicate,
                "run_id": run_id,
                "predicted_class": "ESTABLISHED",
                "predicted_subclass": "",
                "rationale": "",
                "evidence": "",
                "pathway": pathway,
                "pathway_confidence": pathway_confidence,
                "source_trace_path": source_path,
                "base_route": meta.get("base_route", ""),
                "order_variant": meta.get("order_variant", ""),
                "order_hypothesis": meta.get("order_hypothesis", ""),
                "component_order": meta.get("component_order", ""),
            }
        )

    # --- Uncharacterized genes (list of dicts) ---
    for gene_obj in parsed_output.get("uncharacterized_genes", []):
        rows.append(
            {
                "screen_name": screen_name,
                "cluster_id": cluster_id,
                "gene_symbol": gene_obj.get("gene", ""),
                "route": route,
                "replicate": replicate,
                "run_id": run_id,
                "predicted_class": "UNCHARACTERIZED",
                "predicted_subclass": gene_obj.get("class", ""),
                "rationale": gene_obj.get("rationale", ""),
                "evidence": gene_obj.get("evidence", ""),
                "pathway": pathway,
                "pathway_confidence": pathway_confidence,
                "source_trace_path": source_path,
                "base_route": meta.get("base_route", ""),
                "order_variant": meta.get("order_variant", ""),
                "order_hypothesis": meta.get("order_hypothesis", ""),
                "component_order": meta.get("component_order", ""),
            }
        )

    # --- Novel-role genes (list of dicts) ---
    for gene_obj in parsed_output.get("novel_role_genes", []):
        rows.append(
            {
                "screen_name": screen_name,
                "cluster_id": cluster_id,
                "gene_symbol": gene_obj.get("gene", ""),
                "route": route,
                "replicate": replicate,
                "run_id": run_id,
                "predicted_class": "NOVEL_ROLE",
                "predicted_subclass": gene_obj.get("class", ""),
                "rationale": gene_obj.get("rationale", ""),
                "evidence": gene_obj.get("evidence", ""),
                "pathway": pathway,
                "pathway_confidence": pathway_confidence,
                "source_trace_path": source_path,
                "base_route": meta.get("base_route", ""),
                "order_variant": meta.get("order_variant", ""),
                "order_hypothesis": meta.get("order_hypothesis", ""),
                "component_order": meta.get("component_order", ""),
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Main extraction logic
# ---------------------------------------------------------------------------


def extract_predictions_from_traces(traces_dir: Path) -> dict[str, list[dict]]:
    """Read all trace JSONs and return gene rows grouped by route.

    Returns: {route_name: [row_dict, ...]}
    """
    if not traces_dir.is_dir():
        raise FileNotFoundError(f"Traces directory not found: {traces_dir}")

    route_rows: dict[str, list[dict]] = {}
    parse_failures: list[str] = []

    for trace_path in sorted(traces_dir.glob("*.json")):
        # Parse metadata from filename
        file_meta = parse_trace_filename(trace_path)

        # Load trace JSON
        try:
            trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            parse_failures.append(f"{trace_path.name}: JSON read error -- {e}")
            continue

        # Resolve metadata: prefer JSON fields, fall back to filename-parsed values
        meta = {
            "experiment_id": trace_data.get("run_id", "").split("__")[0]
            or file_meta.get("experiment_id"),
            "route": trace_data.get("route") or file_meta.get("route"),
            "screen_name": file_meta.get("screen_name"),  # not in trace JSON directly
            "cluster_id": file_meta.get("cluster_id")
            or str(trace_data.get("parsed_output", {}).get("cluster_id", "")),
            "replicate": file_meta.get("replicate") or 1,
            "run_id": trace_data.get("run_id", trace_path.stem),
            # Phase 2 order metadata (from trace JSON, empty for Phase 1)
            "base_route": trace_data.get("base_route", ""),
            "order_variant": trace_data.get("order_variant", ""),
            "order_hypothesis": trace_data.get("order_hypothesis", ""),
            "component_order": (
                ",".join(trace_data["component_order"])
                if isinstance(trace_data.get("component_order"), list)
                else str(trace_data.get("component_order", ""))
            ),
        }

        if not meta["route"]:
            parse_failures.append(f"{trace_path.name}: could not determine route")
            continue

        parsed_output = trace_data.get("parsed_output")
        if not parsed_output:
            parse_failures.append(f"{trace_path.name}: no parsed_output field")
            continue

        # Extract gene rows
        rows = _extract_gene_rows(
            parsed_output=parsed_output,
            meta=meta,
            source_path=trace_path.name,
        )

        route_rows.setdefault(meta["route"], []).extend(rows)

    # Report failures
    if parse_failures:
        print(f"\n  [trace_parser] {len(parse_failures)} trace(s) skipped:")
        for msg in parse_failures:
            print(f"    - {msg}")

    return route_rows


def write_prediction_csvs(
    route_rows: dict[str, list[dict]],
    output_dir: Path,
    experiment_id: str,
    overwrite: bool = True,
) -> list[Path]:
    """Write one CSV per route. Returns list of written paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for route, rows in sorted(route_rows.items()):
        if not rows:
            continue

        if overwrite:
            fname = f"{experiment_id}_{route}.csv"
        else:
            date_str = datetime.now().strftime("%Y%m%d")
            fname = f"{experiment_id}_{route}_{date_str}.csv"

        out_path = output_dir / fname
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        written.append(out_path)
        print(f"  [trace_parser] Wrote {len(rows)} gene rows -> {out_path.name}")

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Parse benchmark trace JSONs into gene-level prediction CSVs."
    )
    parser.add_argument(
        "--traces-dir",
        type=Path,
        required=True,
        help="Path to the traces/ directory containing run JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write prediction CSVs. Defaults to parent of traces-dir.",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Experiment ID for output filenames. Inferred from traces if omitted.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Use stable filenames (no date suffix) that overwrite on re-run.",
    )
    args = parser.parse_args()

    traces_dir: Path = args.traces_dir.resolve()
    output_dir: Path = (args.output_dir or traces_dir.parent).resolve()

    print(f"\n  Parsing traces from: {traces_dir}")
    print(f"  Output directory:    {output_dir}")

    # Extract
    route_rows = extract_predictions_from_traces(traces_dir)

    if not route_rows:
        print("  No valid traces found. Exiting.")
        sys.exit(1)

    # Infer experiment_id from first trace if not provided
    experiment_id = args.experiment_id
    if not experiment_id:
        # Use the first row's run_id prefix
        first_rows = next(iter(route_rows.values()))
        if first_rows:
            experiment_id = first_rows[0]["run_id"].split("__")[0]
        else:
            experiment_id = "unknown_experiment"

    # Write CSVs
    written = write_prediction_csvs(
        route_rows=route_rows,
        output_dir=output_dir,
        experiment_id=experiment_id,
        overwrite=args.overwrite,
    )

    print(f"\n  Done. {len(written)} CSV(s) written.\n")


if __name__ == "__main__":
    main()
