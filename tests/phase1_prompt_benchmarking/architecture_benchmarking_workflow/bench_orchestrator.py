"""Benchmark orchestrator -- the shared run loop.

Runs the LLM Analysis benchmark across configured routes, clusters, and
replicates. Produces JSONL outputs, traces, and a Markdown report. This is a
library: the pipeline entry points (``run_source.py`` and the other ``run_*``
scripts) build their RunSpecs and drive ``_run_benchmark_loop`` directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path for imports
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=_REPO_ROOT / ".env")
except ImportError:
    pass  # dotenv not required if env vars are set externally

import pandas as pd

from mozzarellm.clients.llm_api_clients import create_client
from mozzarellm.pipeline.literature_mcp import get_available_mcp_servers
from mozzarellm.utils.prompt_factory import (
    compose_stepwise_user_turns,
    make_cluster_analysis_system_prompt,
    make_single_cluster_analysis_user_prompt,
)

from .arch_bench_routes import Route
from .bench_configparse import BenchmarkConfig, TimingConfig
from .bench_dry_run import (
    _load_bundle_genes,
    generate_mock_parsed_output,
    generate_mock_raw_outputs,
)
from .bench_metricfns import compute_all_metrics
from .bench_reportgen import generate_report
from .order_bench_orderings import compose_stepwise_turns_from_route

# =============================================================================
# HELPERS
# =============================================================================

# Serializes shared-file writes (JSONL appends, cached system-prompt .txt) when
# runs execute concurrently in the thread pool.
_IO_LOCK = threading.Lock()


def _client_from_config(config: BenchmarkConfig, api_key: str | None):
    """Build an LLM client from the full model config (temperature, thinking, etc.)."""
    m = config.model
    return create_client(
        model=m.model_name,
        temperature=m.temperature,
        max_tokens=m.max_tokens,
        top_p=m.top_p,
        top_k=m.top_k,
        api_key=api_key,
        thinking=m.thinking,
    )


@dataclass(frozen=True)
class RunSpec:
    """Phase-agnostic work item for the unified benchmark loop.

    Both Phase 1+2 (architecture/order) and Phase 3 (wording) build a list of
    RunSpecs, then hand them to ``_run_benchmark_loop``.
    """

    route: Route
    condition_name: str
    component_overrides: dict[str, str] = field(default_factory=dict)
    extra_record_fields: dict[str, Any] = field(default_factory=dict)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _build_run_id(
    experiment_id: str, route_name: str, screen_name: str, cluster_id: str, replicate: int
) -> str:
    return f"{experiment_id}__{route_name}__{screen_name}__cluster_{cluster_id}__rep_{replicate}"


def _resolve_bundle_path(bundles_dir: Path, screen_name: str, cluster_id: str) -> Path | None:
    """Locate the evidence bundle for a given (screen, cluster) pair."""
    expected = bundles_dir / f"{screen_name}__cluster_{cluster_id}__bundle.json"
    if expected.exists():
        return expected
    # Fallback: glob search
    pattern = f"{screen_name}__cluster_{cluster_id}__bundle.json"
    matches = list(bundles_dir.glob(pattern))
    return matches[0] if matches else None


def _resolve_screen_context_path(inputs_dir: Path, screen_name: str) -> Path | None:
    """Locate screen context JSON for a given screen."""
    expected = inputs_dir / f"{screen_name}_screen_context.json"
    return expected if expected.exists() else None


def _load_benchmark_clusters(csv_path: Path) -> pd.DataFrame:
    """Load benchmark clusters CSV (columns: screen_name, cluster_id, gene_symbol)."""
    df = pd.read_csv(csv_path)
    required = {"screen_name", "cluster_id", "gene_symbol"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"benchmark_clusters.csv must have columns: {required}. Got: {list(df.columns)}"
        )
    df["cluster_id"] = df["cluster_id"].astype(str)
    return df


def _filter_clusters(df: pd.DataFrame, config: BenchmarkConfig) -> list[tuple[str, str]]:
    """Return deduplicated (screen_name, cluster_id) pairs after applying config filters."""
    pairs = df[["screen_name", "cluster_id"]].drop_duplicates()

    # Screen filter
    if config.screens_include != "all":
        pairs = pairs[pairs["screen_name"].isin(config.screens_include)]

    # Cluster filter
    if config.clusters_include != "all":
        filter_set = {(c.screen_name, c.cluster_id) for c in config.clusters_include}
        pairs = pairs[
            pairs.apply(lambda r: (r["screen_name"], r["cluster_id"]) in filter_set, axis=1)
        ]

    return list(pairs.itertuples(index=False, name=None))


def _append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON record to a .jsonl file (thread-safe)."""
    line = json.dumps(record, default=str) + "\n"
    with _IO_LOCK, open(path, "a", encoding="utf-8") as f:
        f.write(line)


# =============================================================================
# PROMPT CONSTRUCTION
# =============================================================================


def construct_prompts(
    route: Route,
    screen_name: str,
    screen_context_path: Path,
    cluster_id: str,
    bundle_path: Path,
    output_dir: Path,
    overwrite_outputs: bool = False,
    component_overrides: dict[str, str] | None = None,
    condition_name: str | None = None,
    source: str = "both",
) -> dict:
    """Construct system and user prompts for a given route+cluster.

    Returns a dict with keys: system_prompt, user_prompt, stepwise_turns (or None).

    Args:
        component_overrides: Phase 3 wording overrides — {component_key: text}
            passed through to ``make_cluster_analysis_system_prompt``.
        condition_name: Stable label for cached prompt filenames. For wording runs
            this is the per-condition run-route name so different wording conditions
            on the same base route do not collide in the prompt cache.
    """
    # Build a pseudo cluster_to_bundle_path_map for the existing utility
    cluster_to_bundle_map = {str(cluster_id): bundle_path}

    # Phase 2 order-variant routes use the flexible component_order assembly
    # path. Phase 1 routes (order_variant == "") use the original mode-based
    # default assembly to preserve backward compatibility exactly.
    is_order_variant = bool(route.order_variant)
    # Label used for the on-disk prompt filename; defaults to the route name.
    prompt_label = condition_name or route.name

    # For 3a/3b routes: save system prompt once with stable filename, reuse on subsequent calls.
    # For 3c (stepwise) routes: build system prompt in memory only
    # to save space, the multi-turn conversation is recorded in traces/prompts.jsonl only -- can extract later if needed.
    if route.delivery == "multi_turn":
        if is_order_variant:
            # Stepwise order-variant: build system prompt from system_components
            # using mode="standard" to avoid step numbering in the system prompt.
            system_prompt = make_cluster_analysis_system_prompt(
                screen_name=screen_name,
                screen_context_path=screen_context_path,
                mode="standard",
                mcp=route.mcp,
                component_order=list(route.system_components),
                component_overrides=component_overrides,
                output_dir=None,
            )
        else:
            system_prompt = make_cluster_analysis_system_prompt(
                screen_name=screen_name,
                screen_context_path=screen_context_path,
                mode=route.mode,
                mcp=route.mcp,
                component_overrides=component_overrides,
                output_dir=None,
            )
    else:
        # screen_name is part of the cache key: the system prompt embeds
        # SCREEN_CONTEXT, so a route's prompt differs per screen. prompt_label
        # distinguishes wording conditions on the same base route.
        prompt_filename = f"system_prompt_{prompt_label}_{route.mode}_{screen_name}"
        prompt_file = output_dir / "prompts_used" / f"{prompt_filename}.txt"
        with _IO_LOCK:
            if prompt_file.exists() and not is_order_variant:
                system_prompt = prompt_file.read_text(encoding="utf-8")
            else:
                system_prompt = make_cluster_analysis_system_prompt(
                    screen_name=screen_name,
                    screen_context_path=screen_context_path,
                    mode=route.mode,
                    mcp=route.mcp,
                    component_order=(list(route.component_order) if is_order_variant else None),
                    component_overrides=component_overrides,
                    output_dir=output_dir / "prompts_used",
                    prompt_filename=prompt_filename,
                )

    # Features reach the model only when a feature-interpretation component
    # (cFC / cPC) is in the route; otherwise they are stripped from the bundle so
    # the prompt carries no uninterpreted feature data.
    route_components = set(route.component_order or ()) | set(
        getattr(route, "system_components", None) or ()
    )
    include_features = bool(route_components & {"cFC", "cPC"})
    user_prompt = make_single_cluster_analysis_user_prompt(
        cluster_id,
        screen_name,
        cluster_to_bundle_map,
        include_features=include_features,
        source=source,
    )

    stepwise_turns = None
    if route.delivery == "multi_turn":
        if is_order_variant and route.user_turns:
            stepwise_turns = compose_stepwise_turns_from_route(route, screen_context_path)
        else:
            stepwise_turns = compose_stepwise_user_turns(route.mcp)

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "stepwise_turns": stepwise_turns,
    }


# =============================================================================
# SINGLE RUN EXECUTION
# =============================================================================


def _build_timing_dict(
    route: Route,
    t_prompt: float | None,
    t_model: float | None,
    t_metrics: float | None,
    t_io: float | None,
    t_full_run: float | None,
    raw_outputs: dict,
    timing_cfg: TimingConfig | None = None,
) -> dict:
    """Assemble the per-run timing dict, respecting TimingConfig flags."""
    if timing_cfg is None:
        timing_cfg = TimingConfig()

    n_api_calls = 1
    step_latencies = None
    slowest_step = None
    mcp_tool_latency = None

    steps = raw_outputs.get("steps", [])
    if timing_cfg.track_step_latencies and route.delivery == "multi_turn" and steps:
        n_api_calls = len(steps)
        step_latencies = []
        max_lat, max_comp = 0.0, None
        for i, s in enumerate(steps):
            lat = s.get("elapsed_s", s.get("latency_seconds", 0.0)) or 0.0
            comp = s.get("component", f"step_{i + 1}")
            step_latencies.append(
                {
                    "step_index": i + 1,
                    "component": comp,
                    "mcp": bool(s.get("mcp", False)),
                    "latency_seconds": lat,
                }
            )
            if lat > max_lat:
                max_lat, max_comp = lat, comp
        slowest_step = max_comp
    elif route.delivery == "multi_turn" and steps:
        n_api_calls = len(steps)

    # MCP tool-level latency (sum of tool call durations if available)
    if timing_cfg.track_mcp_tool_latency:
        tool_calls = raw_outputs.get("tool_calls", [])
        if route.mcp and tool_calls:
            mcp_tool_latency = (
                sum(tc.get("latency_seconds", 0.0) for tc in tool_calls if isinstance(tc, dict))
                or None
            )

    return {
        "full_run_time_seconds": t_full_run if timing_cfg.track_full_run else None,
        "prompt_construction_seconds": t_prompt if timing_cfg.track_prompt_construction else None,
        "model_latency_seconds": t_model if timing_cfg.track_model_latency else None,
        "parse_time_seconds": None,  # parse is inside client.analyze; not separately measurable
        "metrics_time_seconds": t_metrics if timing_cfg.track_metrics else None,
        "io_write_time_seconds": t_io if timing_cfg.track_io else None,
        "mcp_preflight_seconds": None,  # measured once at benchmark level, not per-run
        "mcp_tool_latency_seconds": mcp_tool_latency,
        "n_api_calls": n_api_calls,
        "step_latencies_seconds": step_latencies,
        "slowest_step_component": slowest_step,
    }


def execute_single_run(
    route: Route,
    screen_name: str,
    cluster_id: str,
    bundle_path: Path,
    screen_context_path: Path,
    replicate: int,
    config: BenchmarkConfig,
    client,
    output_dir: Path,
    component_overrides: dict[str, str] | None = None,
    condition_name: str | None = None,
    extra_record_fields: dict | None = None,
) -> dict:
    """Execute a single benchmark run and return the full record.

    Args:
        component_overrides: Phase 3 wording overrides passed to prompt construction.
        condition_name: Per-condition label (e.g. wording run-route name). Used for
            run_id, the record's "route" field, and prompt-cache filenames so that
            distinct conditions on the same base route do not collide.
        extra_record_fields: Extra key/values merged into the output record (e.g.
            wording metadata).
    """
    t_run_start = time.perf_counter()
    record_route_name = condition_name or route.name
    run_id = _build_run_id(
        config.experiment_id, record_route_name, screen_name, cluster_id, replicate
    )

    # --- Prompt construction ---
    t0 = time.perf_counter()
    prompts = construct_prompts(
        route=route,
        screen_name=screen_name,
        screen_context_path=screen_context_path,
        cluster_id=cluster_id,
        bundle_path=bundle_path,
        output_dir=output_dir,
        overwrite_outputs=config.run.overwrite_outputs,
        component_overrides=component_overrides,
        condition_name=condition_name,
        source=config.paths.bundle_source,
    )
    system_prompt = prompts["system_prompt"]
    user_prompt = prompts["user_prompt"]
    t_prompt = time.perf_counter() - t0

    system_prompt_hash = _sha256(system_prompt)
    user_prompt_hash = _sha256(user_prompt)

    # Load bundle genes for metrics
    bundle_genes = _load_bundle_genes(bundle_path)

    # --- Model execution ---
    parsed = None
    raw_outputs = {}
    error = None

    t0 = time.perf_counter()
    if config.run.dry_run:
        parsed = generate_mock_parsed_output(cluster_id, bundle_genes, route)
        raw_outputs = generate_mock_raw_outputs(route)
    else:
        try:
            parsed, raw_outputs = client.analyze(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                mode=route.mode,
                mcp=route.mcp,
            )
            error = raw_outputs.get("error")
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            raw_outputs = {
                "response_text": "",
                "tool_calls": [],
                "input_tokens": 0,
                "output_tokens": 0,
                "elapsed_s": 0.0,
                "cost_usd": 0.0,
                "pricing_warning": None,
                "schema_warnings": [],
                "error": error,
                "steps": [],
            }
    t_model = time.perf_counter() - t0

    # --- Metrics computation ---
    t0 = time.perf_counter()
    metrics = compute_all_metrics(
        parsed=parsed,
        raw_outputs=raw_outputs,
        cluster_id=cluster_id,
        bundle_genes=bundle_genes,
        mcp_enabled=route.mcp,
    )
    t_metrics = time.perf_counter() - t0

    # --- Build timing ---
    # (IO time filled in after _save_run_artifacts)
    timing = _build_timing_dict(
        route=route,
        t_prompt=t_prompt,
        t_model=t_model,
        t_metrics=t_metrics,
        t_io=0.0,  # placeholder, updated below
        t_full_run=0.0,  # placeholder, updated below
        raw_outputs=raw_outputs,
        timing_cfg=config.timing,
    )

    # Build record
    record = {
        "run_id": run_id,
        "experiment_id": config.experiment_id,
        "route": record_route_name,
        "mode": route.mode,
        "mcp": route.mcp,
        "delivery": route.delivery,
        "screen_name": screen_name,
        "cluster_id": cluster_id,
        "replicate": replicate,
        "model_name": config.model.model_name,
        "temperature": config.model.temperature,
        "system_prompt_hash": system_prompt_hash,
        "user_prompt_hash": user_prompt_hash,
        "evidence_bundle_path": str(bundle_path),
        "screen_context_path": str(screen_context_path),
        "parsed_output": parsed,
        "metrics": metrics,
        "timing": timing,
        "trace_path": str(output_dir / "traces" / f"{run_id}.json"),
        "latency_seconds": raw_outputs.get("elapsed_s", 0.0),
        "input_tokens": raw_outputs.get("input_tokens"),
        "output_tokens": raw_outputs.get("output_tokens"),
        "total_tokens": (raw_outputs.get("input_tokens", 0) or 0)
        + (raw_outputs.get("output_tokens", 0) or 0),
        "estimated_cost_usd": raw_outputs.get("cost_usd"),
        "schema_warnings": raw_outputs.get("schema_warnings", []),
        "mcp_tool_calls": raw_outputs.get("tool_calls", []),
        "mcp_gapfill": raw_outputs.get("mcp_gapfill", []),
        "error": error,
        # Order benchmarking metadata (Phase 2)
        "base_route": route.base_route or route.name,
        "order_variant": route.order_variant or "none",
        "order_hypothesis": route.order_hypothesis or None,
        "component_order": list(route.component_order),
        "system_components": list(route.system_components) if route.system_components else [],
        "user_turns_order": (
            [{"component": t.component, "mcp": t.mcp} for t in route.user_turns]
            if route.user_turns
            else []
        ),
    }

    # Merge any caller-provided extra fields (e.g. Phase 3 wording metadata).
    if extra_record_fields:
        record.update(extra_record_fields)

    # --- IO writes ---
    t0 = time.perf_counter()
    _save_run_artifacts(
        record=record,
        prompts=prompts,
        raw_outputs=raw_outputs,
        parsed=parsed,
        route=route,
        config=config,
        output_dir=output_dir,
    )
    t_io = time.perf_counter() - t0

    # Finalize timing
    t_full_run = time.perf_counter() - t_run_start
    timing["io_write_time_seconds"] = t_io
    timing["full_run_time_seconds"] = t_full_run

    return record


def _save_run_artifacts(
    record: dict,
    prompts: dict,
    raw_outputs: dict,
    parsed: dict | None,
    route: Route,
    config: BenchmarkConfig,
    output_dir: Path,
) -> None:
    """Persist all per-run artifacts to JSONL files and trace."""
    run_id = record["run_id"]
    # Condition label (base route for Phase 1/2; wording run-route name for Phase 3).
    route_label = record.get("route", route.name)

    # Prompts
    if config.run.save_prompts:
        prompt_record = {
            "run_id": run_id,
            "route": route_label,
            "screen_name": record["screen_name"],
            "cluster_id": record["cluster_id"],
            "replicate": record["replicate"],
            "mode": route.mode,
            "mcp": route.mcp,
            "delivery": route.delivery,
            "system_prompt": prompts["system_prompt"],
            "user_prompt": prompts["user_prompt"],
            "stepwise_turns": (
                [
                    {
                        "step_index": i + 1,
                        "component": t["content"][:50],
                        "mcp": t["mcp"],
                        "content": t["content"],
                    }
                    for i, t in enumerate(prompts["stepwise_turns"])
                ]
                if prompts["stepwise_turns"]
                else None
            ),
            "system_prompt_hash": record["system_prompt_hash"],
            "user_prompt_hash": record["user_prompt_hash"],
        }
        _append_jsonl(output_dir / "prompts.jsonl", prompt_record)

    # Raw outputs
    if config.run.save_raw_outputs:
        raw_record = {
            "run_id": run_id,
            "route": route_label,
            "raw_output": raw_outputs.get("response_text", ""),
            "tool_calls": raw_outputs.get("tool_calls", []),
            "mcp_gapfill": raw_outputs.get("mcp_gapfill", []),
            "steps": raw_outputs.get("steps", []),
            "error": raw_outputs.get("error"),
        }
        _append_jsonl(output_dir / "raw_outputs.jsonl", raw_record)

    # Parsed outputs
    if config.run.save_parsed_outputs:
        parsed_record = {
            "run_id": run_id,
            "route": route_label,
            "parsed_output": parsed,
        }
        _append_jsonl(output_dir / "parsed_outputs.jsonl", parsed_record)

    # Metrics
    metrics_record = {"run_id": run_id, "route": route_label, **record["metrics"]}
    _append_jsonl(output_dir / "metrics.jsonl", metrics_record)

    # Trace
    if config.run.save_traces:
        trace_dir = output_dir / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_data = {
            "run_id": run_id,
            "route": route_label,
            "mode": route.mode,
            "mcp": route.mcp,
            "raw_response": raw_outputs.get("response_text", ""),
            "parsed_output": parsed,
            "tokens": {
                "input": raw_outputs.get("input_tokens"),
                "output": raw_outputs.get("output_tokens"),
            },
            "cost": raw_outputs.get("cost_usd", 0.0),
            "timing": record.get("timing", {}),
            "error": raw_outputs.get("error"),
        }
        if route.delivery == "multi_turn":
            trace_data["steps"] = raw_outputs.get("steps", [])
            trace_data["final_parsed_output"] = parsed
        (trace_dir / f"{run_id}.json").write_text(
            json.dumps(trace_data, indent=2, default=str), encoding="utf-8"
        )


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================


def _maybe_truncate_outputs(output_dir: Path, config: BenchmarkConfig) -> None:
    """Truncate JSONL files and clear trace/prompt caches if overwrite_outputs is enabled.

    When the resolved output path is NOT under ``_workflow_testing``, prompts
    for interactive confirmation before truncating non-empty files to prevent
    accidental data loss on production experiment directories.
    """
    if not config.run.overwrite_outputs:
        return

    is_testing_path = "_workflow_testing" in str(output_dir)

    if not is_testing_path:
        existing = list(output_dir.glob("*.jsonl"))
        non_empty = [f for f in existing if f.stat().st_size > 0]
        if non_empty:
            print(
                f"\n  [overwrite_outputs] WARNING: about to truncate "
                f"{len(non_empty)} non-empty JSONL file(s) in:\n"
                f"    {output_dir}\n"
                f"  This path is NOT under _workflow_testing."
            )
            try:
                answer = input("  Proceed? [y/N] ").strip().lower()
            except EOFError:
                answer = ""
            if answer != "y":
                print("  Aborted.")
                raise SystemExit(1)

    for jsonl_file in output_dir.glob("*.jsonl"):
        jsonl_file.write_text("", encoding="utf-8")

    traces_dir = output_dir / "traces"
    if traces_dir.is_dir():
        for trace_file in traces_dir.glob("*.json"):
            trace_file.unlink()

    prompts_dir = output_dir / "prompts_used"
    if prompts_dir.is_dir():
        for prompt_file in prompts_dir.glob("*.txt"):
            prompt_file.unlink()

    print(f"  [overwrite_outputs] Truncated JSONL files in {output_dir}")


def _build_config_snapshot(config: BenchmarkConfig, **phase_fields: Any) -> dict:
    """Build serialisable config snapshot, merging in phase-specific keys."""
    snapshot: dict[str, Any] = {
        "experiment_id": config.experiment_id,
        "model": asdict(config.model),
        "run": asdict(config.run),
        "screens_include": config.screens_include,
        "clusters_include": (
            config.clusters_include
            if config.clusters_include == "all"
            else [asdict(c) for c in config.clusters_include]
        ),
        "mcp": asdict(config.mcp),
        "evaluation": asdict(config.evaluation),
        "timing": asdict(config.timing),
        "paths": {k: str(v) for k, v in asdict(config.paths).items()},
    }
    snapshot.update(phase_fields)
    return snapshot


def _run_benchmark_loop(
    config: BenchmarkConfig,
    run_specs: list[RunSpec],
    config_snapshot: dict,
    phase_label: str = "benchmark",
    manifest_extra: dict[str, Any] | None = None,
) -> list[dict]:
    """Shared execution loop for all benchmark phases.

    Handles output-dir setup, MCP preflight, API-key resolution,
    path resolution with skip logging, threaded execution,
    manifest writing, and report generation.
    """
    output_dir = config.experiment_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _maybe_truncate_outputs(output_dir, config)

    (output_dir / "config_snapshot.yaml").write_text(
        json.dumps(config_snapshot, indent=2, default=str), encoding="utf-8"
    )

    # Load benchmark data
    print(f"Loading benchmark clusters from: {config.paths.benchmark_clusters_csv}")
    clusters_df = _load_benchmark_clusters(config.paths.benchmark_clusters_csv)
    cluster_pairs = _filter_clusters(clusters_df, config)
    print(f"  {len(cluster_pairs)} (screen, cluster) pairs after filtering.")

    print(f"  {len(run_specs)} conditions: {[s.condition_name for s in run_specs]}")

    # MCP preflight
    mcp_specs = [s for s in run_specs if s.route.mcp]
    mcp_available = True
    if mcp_specs and config.mcp.preflight and not config.run.dry_run:
        print("  MCP preflight check...", end=" ")
        available_servers = get_available_mcp_servers()
        if "pubmed" not in available_servers:
            print("FAILED (PubMed unavailable)")
            if config.mcp.fail_if_unavailable:
                raise RuntimeError("PubMed MCP server unavailable and fail_if_unavailable=True")
            print("  WARNING: MCP routes will be skipped.")
            mcp_available = False
        else:
            print("OK")

    # Resolve API key once; clients are created per work item (the client
    # stores last_usage on self, so a shared instance would cross-contaminate
    # token/cost accounting across threads).
    api_key = None
    probe = None
    if not config.run.dry_run:
        api_key = (
            os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        probe = _client_from_config(config, api_key)
        print(f"  Client: {type(probe).__name__} ({config.model.model_name})")

    # Build the flat work list, resolving paths and applying skips up front.
    work_items: list[tuple[RunSpec, str, str, Path, Path, int]] = []
    for spec in run_specs:
        if spec.route.mcp and not mcp_available and not config.run.dry_run:
            print(f"  [SKIP] {spec.condition_name} -- MCP unavailable")
            continue
        for screen_name, cluster_id in cluster_pairs:
            bundle_path = _resolve_bundle_path(
                config.paths.evidence_bundles_dir, screen_name, cluster_id
            )
            if bundle_path is None:
                print(
                    f"  [SKIP] {spec.condition_name}/{screen_name}/cluster_{cluster_id}"
                    " -- bundle not found"
                )
                continue
            screen_context_path = _resolve_screen_context_path(
                config.paths.benchmark_inputs_dir, screen_name
            )
            if screen_context_path is None:
                print(
                    f"  [SKIP] {spec.condition_name}/{screen_name}/cluster_{cluster_id}"
                    " -- screen context not found"
                )
                continue
            for rep in range(1, config.run.num_replicates + 1):
                work_items.append(
                    (spec, screen_name, cluster_id, bundle_path, screen_context_path, rep)
                )

    total_runs = len(work_items)
    max_workers = 1 if config.run.dry_run else max(1, config.run.max_workers)
    print(f"\nStarting {phase_label}: {total_runs} total runs (max_workers={max_workers})")
    print(f"  Output: {output_dir}\n")

    def _run_one(
        item: tuple[RunSpec, str, str, Path, Path, int],
    ) -> dict:
        spec, screen_name, cluster_id, bundle_path, screen_context_path, rep = item
        run_client = None if config.run.dry_run else _client_from_config(config, api_key)
        return execute_single_run(
            route=spec.route,
            screen_name=screen_name,
            cluster_id=cluster_id,
            bundle_path=bundle_path,
            screen_context_path=screen_context_path,
            replicate=rep,
            config=config,
            client=run_client,
            output_dir=output_dir,
            component_overrides=spec.component_overrides or None,
            condition_name=spec.condition_name,
            extra_record_fields=spec.extra_record_fields or None,
        )

    all_records: list[dict] = []
    run_counter = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_one, item): item for item in work_items}
        for future in as_completed(futures):
            spec, screen_name, cluster_id, _, _, rep = futures[future]
            run_counter += 1
            run_id = _build_run_id(
                config.experiment_id, spec.condition_name, screen_name, cluster_id, rep
            )
            try:
                record = future.result()
            except Exception as e:
                print(f"  [{run_counter}/{total_runs}] {run_id} EXCEPTION: {e}")
                if not config.run.continue_on_error:
                    for f in futures:
                        f.cancel()
                    raise
                continue

            all_records.append(record)
            if record.get("error"):
                print(f"  [{run_counter}/{total_runs}] {run_id} ERROR: {record['error'][:80]}")
            elif config.run.dry_run:
                print(f"  [{run_counter}/{total_runs}] {run_id} OK (dry run)")
            else:
                full_t = record["timing"]["full_run_time_seconds"]
                model_lat = record["timing"]["model_latency_seconds"]
                cost = record["estimated_cost_usd"]
                print(
                    f"  [{run_counter}/{total_runs}] {run_id} "
                    f"OK (full={full_t:.1f}s, model={model_lat:.1f}s, ${cost:.4f})"
                )

    # Save run manifest
    manifest: dict[str, Any] = {
        "experiment_id": config.experiment_id,
        "timestamp": datetime.now().isoformat(),
        "total_runs": len(all_records),
        "routes_executed": list({r["route"] for r in all_records}),
        "screens_executed": list({r["screen_name"] for r in all_records}),
        "clusters_executed": list({(r["screen_name"], r["cluster_id"]) for r in all_records}),
        "errors": sum(1 for r in all_records if r.get("error")),
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    # Record the model params the client actually sent (resolution is
    # deterministic per config+model, so the probe speaks for every run client).
    if probe is not None and hasattr(probe, "_resolve_params"):
        probe._resolve_params()
        manifest["model_resolved_params"] = probe.resolved_params
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    # Generate report
    print("\nGenerating report...")
    report_path = generate_report(all_records, config_snapshot, output_dir)
    print(f"  Report: {report_path}")
    print(f"  Done. {len(all_records)} analysis runs completed, {manifest['errors']} errors.")

    return all_records
