"""One-call screen analysis: prompt -> per-cluster LLM calls -> traces -> tables.

``analyze_screen`` is the library layer under the interface notebook: it builds
the system prompt for the chosen mode, walks every cluster's evidence bundle
through ``client.analyze``, writes a per-cluster trace JSON, and produces the
run's outputs -- the raw per-cluster JSON plus the user-facing gene and cluster
tables (``<screen>_genes.csv`` / ``<screen>_clusters.csv``).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from mozzarellm.pipeline.bundle_builder import (
    build_evidence_bundles,
    get_or_append_stable_accession,
)
from mozzarellm.prompt_components import build_cot_component_order
from mozzarellm.utils.cluster_utils import (
    STRENGTH_RANK_COL,
    attach_strength_ranks,
    build_cluster_id_to_bundle_path,
)
from mozzarellm.utils.io import load_table
from mozzarellm.utils.llm_analysis_utils import save_cluster_analysis
from mozzarellm.utils.prompt_factory import (
    compose_stepwise_user_turns,
    make_cluster_analysis_system_prompt,
    make_single_cluster_analysis_user_prompt,
)
from mozzarellm.utils.trace import save_trace


def prepare_screen_bundles(
    *,
    screen_name: str,
    cluster_table,
    output_dir: str | Path,
    gene_column: str = "gene_symbol",
    cluster_id_column: str = "cluster",
    feature_columns: list[str] | None = None,
    strength_column: str | None = None,
    strength_higher_is_stronger: bool = True,
    organism_id: int = 9606,
    control_prefix: str = "nontargeting_",
    rebuild: bool = False,
) -> dict:
    """Cluster table -> stable accessions -> evidence bundles -> {cluster_id: path}.

    Bundles are cached under ``<output_dir>/<screen_name>_analysis/``; an
    existing bundle directory is reused unless ``rebuild=True``.

    Args:
        cluster_table: DataFrame or path to a CSV/TSV/XLSX with one row per
            gene, carrying ``gene_column`` and ``cluster_id_column`` (plus any
            ``feature_columns`` to embed in the bundles).
        strength_column: Optional per-gene perturbation-strength column — any
            metric (AUC, e-distance, ...). The raw values never enter the
            bundles; each gene gets a scale-free ``phenotype_strength_rank``
            of the form ``"N/M"`` (rank among the table's M scored genes,
            1 = strongest). A column already holding ``"N/M"`` strings passes
            through unchanged. Genes with missing strength carry no rank.
        strength_higher_is_stronger: Direction of the raw metric — True when
            larger values mean a stronger phenotype (e.g. AUC, e-distance).
        control_prefix: Gene symbols with this prefix are treated as
            non-targeting controls (mapped to ``NON_TARGETING_CONTROL``
            instead of a UniProt lookup).
    """
    output_dir = Path(output_dir)
    cluster_df = (
        cluster_table if hasattr(cluster_table, "columns") else load_table(cluster_table)
    )
    if strength_column is not None:
        cluster_df = attach_strength_ranks(
            cluster_df, strength_column, higher_is_stronger=strength_higher_is_stronger
        )
    bundle_dir = output_dir / f"{screen_name}_analysis" / f"{screen_name}_evidence_bundles"

    if rebuild or not bundle_dir.exists():
        acc_cluster_df = get_or_append_stable_accession(
            screen_name=screen_name,
            cluster_df=cluster_df,
            gene_column=gene_column,
            organism_id=organism_id,
            warn_on_fallback=False,
            control_prefix=control_prefix,
            output_dir=output_dir,
        )
        build_evidence_bundles(
            screen_name=screen_name,
            acc_cluster_df=acc_cluster_df,
            gene_column=gene_column,
            cluster_id_column=cluster_id_column,
            stable_accession_col="accession",
            feature_columns=feature_columns or None,
            output_dir=output_dir,
        )
    else:
        logging.info(f"Using cached bundles at {bundle_dir}")

    return build_cluster_id_to_bundle_path(bundle_dir, screen_name=screen_name)


_NO_PATHWAY = "no coherent biological pathway"


def _add_coverage(parsed: dict, bundle_path) -> dict:
    """Ground the cluster's coverage in the bundle's actual gene list.

    The model's own totals are not trusted: total_genes_in_cluster,
    missed_genes, and classification_completeness are computed against the
    evidence bundle, so an incomplete response cannot report itself complete.
    """
    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    genes = [
        g.get("gene_symbol")
        for g in bundle.get("cluster_genes") or []
        if isinstance(g, dict) and g.get("gene_symbol")
    ]
    classified = set(parsed.get("established_genes") or [])
    for key in ("novel_role_genes", "uncharacterized_genes"):
        classified |= {g.get("gene") for g in parsed.get(key) or [] if isinstance(g, dict)}
    parsed["total_genes_in_cluster"] = len(genes)
    parsed["missed_genes"] = sorted(set(genes) - classified)
    parsed["classification_completeness"] = (
        len(classified & set(genes)) / len(genes) if genes else 1.0
    )
    return parsed


def _bundle_has_features(bundle: dict) -> bool:
    return "feature_coherence" in bundle or any(
        isinstance(g, dict) and (g.get("up_features") or g.get("down_features"))
        for g in bundle.get("cluster_genes", [])
    )


def _bundle_has_strength(bundle: dict) -> bool:
    return any(
        isinstance(g, dict) and g.get(STRENGTH_RANK_COL)
        for g in bundle.get("cluster_genes", [])
    )


def _resolve_phenotype_flag(requested, name: str, cluster_to_bundle_map: dict, probe) -> bool:
    """Resolve an "auto"/True/False phenotype flag against the actual bundles.

    The corresponding prompt steps enter the chain iff the data exists: "auto"
    detects it, True demands it (error when absent), False strips it.
    """
    if requested not in ("auto", True, False):
        raise ValueError(f"{name} must be True, False, or 'auto'; got {requested!r}")
    present = any(
        probe(json.loads(Path(p).read_text(encoding="utf-8")))
        for p in cluster_to_bundle_map.values()
    )
    if requested is True and not present:
        raise ValueError(f"{name}=True but no bundle carries the corresponding data")
    return present if requested == "auto" else requested


def analyze_screen(
    *,
    screen_name: str,
    cluster_to_bundle_map: dict,
    client,
    run_dir: str | Path,
    screen_context_path: str | Path | None = None,
    screen_context: dict | None = None,
    mode: str = "cot",
    mcp: bool = False,
    include_features: bool | str = "auto",
    include_strength: bool | str = "auto",
    component_overrides: dict[str, str] | None = None,
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
        screen_context: The same context as an in-memory dict (validated
            through the same schema); use instead of writing a JSON file.
        mode: "standard" | "cot" | "stepwise" -- prompt delivery format.
        mcp: Attach PubMed literature-validation tools.
        include_features: Feed each gene's up/down feature lists to the model
            and add the feature-interpretation reasoning steps (cFC, cPC).
            "auto" (default) includes them iff the bundles carry feature data;
            True requires it (error when absent); False strips it. The steps
            enter the prompt only when the data does. Supported for
            mode="cot" (with or without MCP).
        include_strength: Same contract for the phenotype-strength step (cPS)
            and the per-gene ``phenotype_strength_rank`` field (bundles built
            with ``strength_column``).
        component_overrides: {component_key: text} replacements for individual
            prompt components (see mozzarellm.prompt_components
            COMPONENT_REGISTRY) -- run your own wording for any reasoning step
            without editing the package.
        original_df: Optional per-cluster metadata table (must carry
            ``cluster_id``); its columns merge into the output tables.

    Returns:
        dict with ``results`` (per-cluster parsed JSON), ``gene_df`` /
        ``cluster_df`` (the tabular view, also written as CSVs),
        ``total_cost_usd``, and ``errors`` ({cluster_id: message}).
    """
    if mode != "cot":
        if include_features is True or include_strength is True:
            raise ValueError(
                "include_features/include_strength are supported for mode='cot' "
                "(the phenotype reasoning steps are CoT components)"
            )
        features = strength = False  # "auto" resolves off where the steps don't exist
    else:
        features = _resolve_phenotype_flag(
            include_features, "include_features", cluster_to_bundle_map, _bundle_has_features
        )
        strength = _resolve_phenotype_flag(
            include_strength, "include_strength", cluster_to_bundle_map, _bundle_has_strength
        )

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    mode_label = f"{mode}_mcp" if mcp else mode
    if features:
        mode_label += "_feat"
    if strength:
        mode_label += "_strength"

    component_order = (
        build_cot_component_order(mcp=mcp, features=features, strength=strength)
        if (features or strength)
        else None
    )
    system_prompt = make_cluster_analysis_system_prompt(
        screen_name=screen_name,
        screen_context_path=screen_context_path,
        screen_context=screen_context,
        mode=mode,
        mcp=mcp,
        component_order=component_order,
        component_overrides=component_overrides,
    )
    stepwise_turns = (
        compose_stepwise_user_turns(mcp, component_overrides) if mode == "stepwise" else None
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
            include_features=features,
            include_strength=strength,
        )
        try:
            parsed, raw_outputs = client.analyze(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                mode=mode,
                mcp=mcp,
                **({"stepwise_turns": stepwise_turns} if stepwise_turns is not None else {}),
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
            parsed = _add_coverage(parsed, cluster_to_bundle_map[cluster_id])
            # An empty classification with a pathway call is a parse failure,
            # not an abstention -- abstentions declare no coherent pathway.
            abstained = _NO_PATHWAY in str(parsed.get("dominant_process", "")).lower()
            if parsed["total_genes_in_cluster"] and not parsed[
                "classification_completeness"
            ] and not abstained:
                errors.setdefault(
                    cluster_id, "no genes parsed from the response (see the trace)"
                )
            results[cluster_id] = parsed

    tables = save_cluster_analysis(
        results,
        out_file_base=str(run_dir / screen_name),
        original_df=original_df,
    )
    latest = {
        "run_dir": run_dir.name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "screen_name": screen_name,
    }
    (run_dir.parent / "latest.json").write_text(json.dumps(latest, indent=2))
    return {
        "results": results,
        "gene_df": tables["gene_df"],
        "cluster_df": tables["cluster_df"],
        "run_dir": run_dir,
        "total_cost_usd": round(total_cost, 4),
        "errors": errors,
    }
