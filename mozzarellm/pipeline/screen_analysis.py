"""One-call screen analysis: prompt -> per-cluster LLM calls -> traces -> tables.

``analyze_screen`` is the library layer under the interface notebook: it builds
the system prompt for the chosen mode, walks every cluster's evidence bundle
through ``client.analyze``, writes a per-cluster trace JSON, and produces the
run's outputs -- the raw per-cluster JSON plus the user-facing gene and cluster
tables (``<screen>_genes.csv`` / ``<screen>_clusters.csv``).
"""

from __future__ import annotations

import logging
from pathlib import Path

from mozzarellm.prompt_components import CANONICAL_FEATURE_INTERP_COT_ORDER
from mozzarellm.utils.llm_analysis_utils import save_cluster_analysis
from mozzarellm.utils.prompt_factory import (
    make_cluster_analysis_system_prompt,
    make_single_cluster_analysis_user_prompt,
)
from mozzarellm.utils.trace import save_trace


def analyze_screen(
    *,
    screen_name: str,
    cluster_to_bundle_map: dict,
    client,
    run_dir: str | Path,
    screen_context_path: str | Path | None = None,
    mode: str = "cot",
    mcp: bool = False,
    include_features: bool = False,
    original_df=None,
) -> dict:
    """Analyze every cluster in a screen and write the run's outputs.

    Args:
        screen_name: Label for the screen; prefixes the output files.
        cluster_to_bundle_map: {cluster_id: evidence bundle path}, e.g. from
            ``build_cluster_id_to_bundle_path``.
        client: An LLM client from ``create_client``.
        run_dir: Directory the run writes into (traces/ + JSON + CSVs).
        screen_context_path: The screen's context JSON (assay, imaging,
            clustering); embedded in the system prompt.
        mode: "standard" | "cot" | "stepwise" -- prompt delivery format.
        mcp: Attach PubMed literature-validation tools.
        include_features: Feed each gene's phenotypic feature columns to the
            model and add the feature-interpretation reasoning steps.
            Currently supported for mode="cot" without MCP.
        original_df: Optional per-cluster metadata table (must carry
            ``cluster_id``); its columns merge into the output tables.

    Returns:
        dict with ``results`` (per-cluster parsed JSON), ``gene_df`` /
        ``cluster_df`` (the tabular view, also written as CSVs),
        ``total_cost_usd``, and ``errors`` ({cluster_id: message}).
    """
    if include_features and (mode != "cot" or mcp):
        raise ValueError(
            "include_features is currently supported for mode='cot' without MCP "
            "(the feature-interpretation reasoning steps are CoT components)"
        )

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    mode_label = f"{mode}_mcp" if mcp else mode
    if include_features:
        mode_label += "_feat"

    system_prompt = make_cluster_analysis_system_prompt(
        screen_name=screen_name,
        screen_context_path=screen_context_path,
        mode=mode,
        mcp=mcp,
        component_order=(list(CANONICAL_FEATURE_INTERP_COT_ORDER) if include_features else None),
    )

    results: dict = {}
    errors: dict = {}
    total_cost = 0.0
    for cluster_id in cluster_to_bundle_map:
        cluster_id = str(cluster_id)
        user_prompt = make_single_cluster_analysis_user_prompt(
            cluster_id,
            screen_name,
            cluster_to_bundle_map,
            include_features=include_features,
        )
        try:
            parsed, raw_outputs = client.analyze(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                mode=mode,
                mcp=mcp,
            )
        except Exception as e:  # noqa: BLE001 -- one bad cluster must not kill the run
            errors[cluster_id] = str(e)
            save_trace(
                run_dir,
                cluster_id,
                model=client.model,
                mode=mode_label,
                raw_response="",
                error=str(e),
            )
            logging.warning(f"Cluster {cluster_id} failed: {e}")
            continue

        save_trace(
            run_dir,
            cluster_id,
            model=client.model,
            mode=mode_label,
            raw_response=raw_outputs.get("response_text", ""),
            tool_calls=raw_outputs.get("tool_calls", []),
            elapsed_s=raw_outputs.get("elapsed_s"),
            input_tokens=raw_outputs.get("input_tokens"),
            output_tokens=raw_outputs.get("output_tokens"),
            cost_usd=raw_outputs.get("cost_usd"),
            pricing_warning=raw_outputs.get("pricing_warning"),
            schema_warnings=raw_outputs.get("schema_warnings"),
            error=raw_outputs.get("error"),
            steps=raw_outputs.get("steps"),
        )
        total_cost += raw_outputs.get("cost_usd") or 0.0
        if raw_outputs.get("error"):
            errors[cluster_id] = raw_outputs["error"]
        if parsed is not None:
            results[cluster_id] = parsed

    tables = save_cluster_analysis(
        results,
        out_file_base=str(run_dir / screen_name),
        original_df=original_df,
    )
    return {
        "results": results,
        "gene_df": tables["gene_df"],
        "cluster_df": tables["cluster_df"],
        "run_dir": run_dir,
        "total_cost_usd": round(total_cost, 4),
        "errors": errors,
    }
