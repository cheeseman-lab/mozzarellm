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
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

from mozzarellm.clients.affinage_api_client import AffinageClient
from mozzarellm.clients.uniprot_api_client import UniProtClient
from mozzarellm.pipeline.bundle_builder import (
    build_evidence_bundles,
    get_or_append_stable_accession,
    validate_source,
)
from mozzarellm.prompts import (
    compose_stepwise_user_turns,
    default_order,
    make_cluster_analysis_system_prompt,
    make_single_cluster_analysis_user_prompt,
)
from mozzarellm.utils.cluster_utils import (
    attach_strength_ranks,
    build_cluster_id_to_bundle_path,
)
from mozzarellm.utils.io import load_table
from mozzarellm.utils.llm_analysis_utils import process_cluster_response, save_cluster_analysis
from mozzarellm.utils.pricing import compute_cost
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
    source: str = "uniprot",
    uniprot_client: UniProtClient | None = None,
    affinage_client: AffinageClient | None = None,
    rebuild: bool = False,
) -> dict:
    # organism_id is the NCBI taxonomy id the UniProt lookups are restricted to
    # (9606 human, 10090 mouse).
    """Cluster table -> stable accessions -> evidence bundles -> {cluster_id: path}.

    Bundles are cached under ``<output_dir>/<screen_name>_analysis/``; an
    existing bundle directory is reused unless ``rebuild=True``.

    Args:
        cluster_table: DataFrame or path to a CSV/TSV/XLSX with one row per
            gene, carrying ``gene_column`` and ``cluster_id_column`` (plus any
            ``feature_columns`` to embed in the bundles).
        strength_column: Optional per-gene perturbation-strength column — any
            metric (AUC, e-distance, ...). The raw values never enter the
            bundles: the column keeps its name and each gene's value becomes
            a scale-free ``"N/M"`` rank (rank among the table's M scored
            genes, 1 = strongest). A column already holding ``"N/M"`` strings
            passes through unchanged. Genes with missing strength carry no
            rank.
        strength_higher_is_stronger: Direction of the raw metric — True when
            larger values mean a stronger phenotype (e.g. AUC, e-distance).
        organism_id: NCBI taxonomy id for the UniProt lookups (9606 human,
            10090 mouse).
        control_prefix: Gene symbols with this prefix are treated as
            non-targeting controls (mapped to ``NON_TARGETING_CONTROL``
            instead of a UniProt lookup).
        source: Which functional annotation the bundles carry —
            ``"uniprot"`` (the default; UniProt FUNCTION comments),
            ``"affinage"`` (Affinage mechanistic narratives, alias-resolved
            and audit-noted), ``"both"`` (each fetched side by side as its
            own column), or ``"affinage_then_uniprot"``. Among the first
            three no source backfills another: a gene one source has nothing
            for stays empty for that source, and the model sees the gap.
            ``"affinage_then_uniprot"`` is the one mixed source: Affinage is
            fetched for every gene and UniProt is queried only for the genes
            whose Affinage annotation is absent, empty, or a refusal
            narrative, so request volume and prompt length stay close to pure
            Affinage. Each gene then carries ``annotation_source``
            ("affinage", "uniprot", or "" when neither source had anything),
            so a mixed bundle stays auditable per gene; a gene neither source
            has still arrives blank. The accession step always queries
            UniProt, since accessions are UniProt identifiers.
        uniprot_client: Injected ``UniProtClient`` (its cache path, timeouts
            and retries are the knobs); one is built per run when omitted.
        affinage_client: Injected ``AffinageClient``, likewise; only built
            when ``source`` asks for Affinage.
    """
    validate_source(source)
    output_dir = Path(output_dir)
    cluster_df = cluster_table if hasattr(cluster_table, "columns") else load_table(cluster_table)
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
            uniprot_client=uniprot_client,
            output_dir=output_dir,
        )
        build_evidence_bundles(
            screen_name=screen_name,
            acc_cluster_df=acc_cluster_df,
            gene_column=gene_column,
            cluster_id_column=cluster_id_column,
            stable_accession_col="accession",
            feature_columns=feature_columns or None,
            strength_column=strength_column,
            source=source,
            uniprot_client=uniprot_client,
            affinage_client=affinage_client,
            output_dir=output_dir,
        )
    else:
        logging.info(f"Using cached bundles at {bundle_dir}")

    return build_cluster_id_to_bundle_path(bundle_dir, screen_name=screen_name)


_NO_PATHWAY = "no coherent biological pathway"

# The client layer already retries 5xx/timeouts, but a 429 or an overloaded
# response escapes its MCP and stepwise paths -- and concurrency is what makes
# those likely. Bounded backoff here keeps one throttled cluster from failing.
_RATE_LIMIT_ATTEMPTS = 4
_RATE_LIMIT_BACKOFF_S = 5.0
_RATE_LIMIT_MARKERS = ("ratelimit", "rate_limit", "rate limit", "429", "overloaded", "529")


def _is_rate_limited(exc: Exception) -> bool:
    """True when the provider refused for load rather than for the request itself."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


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


def _resolve_phenotype_flag(
    requested, name: str, cluster_to_bundle_map: dict, block: str, gene_fields: tuple[str, ...]
) -> bool:
    """Resolve an "auto"/True/False phenotype flag against the actual bundles.

    The matching prompt steps enter the chain iff the data exists — a bundle
    carries the signal when it has the cluster-level ``block`` or any gene has
    one of ``gene_fields``. "auto" detects it, True demands it (error when
    absent), False strips it.
    """
    if requested not in ("auto", True, False):
        raise ValueError(f"{name} must be True, False, or 'auto'; got {requested!r}")
    present = False
    for path in cluster_to_bundle_map.values():
        bundle = json.loads(Path(path).read_text(encoding="utf-8"))
        genes = [g for g in bundle.get("cluster_genes", []) if isinstance(g, dict)]
        if block in bundle or any(g.get(f) for g in genes for f in gene_fields):
            present = True
            break
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
    component_order: list[str] | None = None,
    original_df=None,
    resume: bool = False,
    dry_run: bool = False,
    max_workers: int = 1,
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
        include_features: Feed the cluster's feature_coherence table (built
            from the per-gene up/down lists, bounded to features covering
            >= 25% of the cluster) to the model and add the
            feature-interpretation reasoning steps (cFC, cPC).
            "auto" (default) includes them iff the bundles carry the table;
            True requires it (error when absent); False strips it. The steps
            enter the prompt only when the data does. Supported for
            mode="cot" (with or without MCP).
        include_strength: Same contract for the phenotype-strength step (cPS)
            and the per-gene rank field (bundles built with ``strength_column``).
        component_overrides: {component_key: text} replacements for individual
            prompt components (see mozzarellm.prompts.components
            COMPONENTS) -- run your own wording for any reasoning step
            without editing the package.
        component_order: Your own chain of component keys in place of the
            default for the mode (see mozzarellm.prompts DEFAULT_ORDERS). Used
            verbatim; the phenotype steps enter the prompt iff you include them
            (cFC/cPC, cPS), and their data must then exist in the bundles.
        original_df: Optional per-cluster metadata table (must carry
            ``cluster_id``); its columns merge into the output tables.
        resume: Skip clusters whose ``traces/cluster_<id>.json`` in
            ``run_dir`` already holds a response; their results are re-read
            from the trace at no cost (``resumed`` lists them).
        dry_run: Assemble every prompt, write them under
            ``run_dir/prompts_used/`` and return per-cluster token and cost
            estimates (``estimates``; input side, ~4 chars per token) without
            calling the model.
        max_workers: How many clusters to analyze concurrently; the per-cluster
            API call runs in a thread pool. 1 (the default) keeps the strictly
            sequential path. ``results``, ``errors`` and ``resumed`` are always
            ordered by ``cluster_to_bundle_map``, and ``total_cost_usd`` summed
            in that same order, so the outputs do not depend on completion
            order. Resumed clusters are read from their traces before the pool
            starts and never occupy a worker; ``dry_run`` makes no API calls
            and stays sequential.

    Returns:
        dict with ``results`` (per-cluster parsed JSON), ``gene_df`` /
        ``cluster_df`` (the tabular view, also written as CSVs),
        ``total_cost_usd``, ``errors`` ({cluster_id: message}), ``resumed``,
        and, for a dry run, ``estimates``.
    """
    if component_order is not None:
        features = _resolve_phenotype_flag(
            "cFC" in component_order,
            "include_features",
            cluster_to_bundle_map,
            "feature_coherence",
            (),
        )
        strength = _resolve_phenotype_flag(
            "cPS" in component_order,
            "include_strength",
            cluster_to_bundle_map,
            "phenotype_strength",
            (),
        )
    elif mode != "cot":
        if include_features is True or include_strength is True:
            raise ValueError(
                "include_features/include_strength are supported for mode='cot' "
                "(the phenotype reasoning steps are CoT components)"
            )
        features = strength = False  # "auto" resolves off where the steps don't exist
    else:
        features = _resolve_phenotype_flag(
            include_features, "include_features", cluster_to_bundle_map, "feature_coherence", ()
        )
        strength = _resolve_phenotype_flag(
            include_strength, "include_strength", cluster_to_bundle_map, "phenotype_strength", ()
        )

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    mode_label = f"{mode}_mcp" if mcp else mode
    if features:
        mode_label += "_feat"
    if strength:
        mode_label += "_strength"

    if component_order is None and (features or strength):
        component_order = default_order(mode, mcp, features=features, strength=strength)
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
        compose_stepwise_user_turns(mcp, component_overrides, component_order)
        if mode == "stepwise"
        else None
    )

    if dry_run:
        return _dry_run(
            run_dir,
            screen_name,
            client,
            system_prompt,
            stepwise_turns,
            cluster_to_bundle_map,
            features,
            strength,
        )

    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1; got {max_workers!r}")

    def _call_cluster(cluster_id: str) -> dict:
        """One cluster's API call and trace write; returns its accumulation record."""
        user_prompt = make_single_cluster_analysis_user_prompt(
            cluster_id,
            screen_name,
            cluster_to_bundle_map,
            include_features=features,
            include_strength=strength,
        )
        try:
            parsed, raw_outputs = _analyze_with_backoff(
                client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                mode=mode,
                mcp=mcp,
                stepwise_turns=stepwise_turns,
            )
        except Exception as e:  # noqa: BLE001 -- one bad cluster must not kill the run
            save_trace(
                run_dir,
                cluster_id,
                model=client.model,
                mode=mode_label,
                raw_response="",
                error=str(e),
            )
            logging.warning(f"Cluster {cluster_id} failed: {e}")
            return {"parsed": None, "error": str(e), "cost": 0.0, "resumed": False}

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
        return {
            "parsed": parsed,
            "error": raw_outputs.get("error"),
            "cost": raw_outputs.get("cost_usd") or 0.0,
            "resumed": False,
        }

    # Resumed clusters are pure trace reads, so they are settled here rather
    # than handed to a worker; only the clusters that still need the model go
    # to the pool.
    cluster_ids = [str(cluster_id) for cluster_id in cluster_to_bundle_map]
    records: dict[str, dict] = {}
    pending: list[str] = []
    for cluster_id in cluster_ids:
        prior = _prior_response(run_dir, cluster_id) if resume else None
        if prior is None:
            pending.append(cluster_id)
        else:
            records[cluster_id] = {
                "parsed": process_cluster_response(prior),
                "error": None,
                "cost": 0.0,
                "resumed": True,
            }

    if max_workers > 1 and len(pending) > 1:
        lock = threading.Lock()
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_call_cluster, cid): cid for cid in pending}
            for future in as_completed(futures):
                cluster_id = futures[future]
                record = future.result()
                # One self-contained line per completion, emitted under the
                # lock, so interleaved workers stay readable in the log.
                with lock:
                    records[cluster_id] = record
                    done += 1
                    logging.info(f"[{done}/{len(pending)}] cluster {cluster_id} done")
    else:
        for cluster_id in pending:
            records[cluster_id] = _call_cluster(cluster_id)

    # Accumulate in cluster_to_bundle_map order, never completion order, so the
    # results, the errors, the resumed list and the cost sum are identical at
    # any max_workers.
    results: dict = {}
    errors: dict = {}
    resumed: list[str] = []
    total_cost = 0.0
    for cluster_id in cluster_ids:
        record = records[cluster_id]
        if record["resumed"]:
            resumed.append(cluster_id)
        total_cost += record["cost"]
        if record["error"]:
            errors[cluster_id] = record["error"]
        parsed = record["parsed"]
        if parsed is not None:
            parsed = _add_coverage(parsed, cluster_to_bundle_map[cluster_id])
            # An empty classification with a pathway call is a parse failure,
            # not an abstention -- abstentions declare no coherent pathway.
            abstained = _NO_PATHWAY in str(parsed.get("dominant_process", "")).lower()
            if (
                parsed["total_genes_in_cluster"]
                and not parsed["classification_completeness"]
                and not abstained
            ):
                errors.setdefault(cluster_id, "no genes parsed from the response (see the trace)")
            results[cluster_id] = parsed

    tables = save_cluster_analysis(
        results,
        out_file_base=str(run_dir / screen_name),
        original_df=original_df,
        gene_extra=_strength_ranks(cluster_to_bundle_map) if strength else None,
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
        "resumed": resumed,
    }


def _analyze_with_backoff(client, *, system_prompt, user_prompt, mode, mcp, stepwise_turns):
    """``client.analyze`` with bounded backoff on a rate-limit/overload refusal."""
    extra = {"stepwise_turns": stepwise_turns} if stepwise_turns is not None else {}
    for attempt in range(_RATE_LIMIT_ATTEMPTS):
        try:
            return client.analyze(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                mode=mode,
                mcp=mcp,
                **extra,
            )
        except Exception as e:
            if attempt == _RATE_LIMIT_ATTEMPTS - 1 or not _is_rate_limited(e):
                raise
            wait = _RATE_LIMIT_BACKOFF_S * 2**attempt
            logging.warning(f"Rate limited ({e}); retrying in {wait:.0f}s")
            time.sleep(wait)


def _prior_response(run_dir: Path, cluster_id: str) -> str | None:
    """The raw response recorded for this cluster in ``run_dir``, if it completed."""
    path = run_dir / "traces" / f"cluster_{cluster_id}.json"
    if not path.exists():
        return None
    trace = json.loads(path.read_text(encoding="utf-8"))
    if trace.get("error") or not trace.get("raw_response"):
        return None
    return trace["raw_response"]


def _strength_ranks(cluster_to_bundle_map: dict) -> dict:
    """{cluster_id: {gene: {"phenotype_strength_rank": "N/M"}}} from the bundles."""
    out: dict = {}
    for cluster_id, path in cluster_to_bundle_map.items():
        bundle = json.loads(Path(path).read_text(encoding="utf-8"))
        column = (bundle.get("phenotype_strength") or {}).get("column")
        if not column:
            continue
        out[str(cluster_id)] = {
            g["gene_symbol"]: {"phenotype_strength_rank": g[column]}
            for g in bundle.get("cluster_genes", [])
            if isinstance(g, dict) and g.get("gene_symbol") and isinstance(g.get(column), str)
        }
    return out


def _dry_run(
    run_dir,
    screen_name,
    client,
    system_prompt,
    stepwise_turns,
    cluster_to_bundle_map,
    features,
    strength,
) -> dict:
    """Write every prompt and estimate the input side; no model call."""
    prompts_dir = run_dir / "prompts_used"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
    turns_chars = sum(len(t["content"]) for t in stepwise_turns or [])
    rows = []
    for cluster_id, path in cluster_to_bundle_map.items():
        cluster_id = str(cluster_id)
        user_prompt = make_single_cluster_analysis_user_prompt(
            cluster_id,
            screen_name,
            cluster_to_bundle_map,
            include_features=features,
            include_strength=strength,
        )
        (prompts_dir / f"user_prompt_cluster_{cluster_id}.txt").write_text(
            user_prompt, encoding="utf-8"
        )
        n_genes = len(json.loads(Path(path).read_text(encoding="utf-8")).get("cluster_genes", []))
        tokens = math.ceil((len(system_prompt) + len(user_prompt) + turns_chars) / 4)
        cost, _ = compute_cost(client.model, tokens, 0)
        rows.append(
            {
                "cluster_id": cluster_id,
                "n_genes": n_genes,
                "est_input_tokens": tokens,
                "est_input_cost_usd": cost,
            }
        )
    estimates = pd.DataFrame(rows)
    return {
        "results": {},
        "gene_df": pd.DataFrame(),
        "cluster_df": pd.DataFrame(),
        "run_dir": run_dir,
        "total_cost_usd": 0.0,
        "errors": {},
        "resumed": [],
        "estimates": estimates,
    }
