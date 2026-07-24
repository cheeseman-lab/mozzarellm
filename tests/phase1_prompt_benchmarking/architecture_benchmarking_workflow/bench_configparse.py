"""Configuration loader and validator for architecture benchmarking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    provider: str = "anthropic"
    model_name: str = "claude-sonnet-4-5"
    temperature: float = 0.2
    max_tokens: int = 4000
    top_p: float | None = None
    top_k: int | None = None
    thinking: bool | None = None  # None = model default, False = off, True = on


@dataclass
class PathsConfig:
    benchmark_inputs_dir: Path = Path("benchmark_inputs")
    benchmark_clusters_csv: Path = Path("benchmark_inputs/benchmark_clusters.csv")
    evidence_bundles_dir: Path = Path("benchmark_evidence_bundles_uniprot")
    output_dir: Path = Path("1.architecture_testing_outputs")
    bundle_source: str = "uniprot"  # "uniprot" or "affinage"


@dataclass
class RunConfig:
    num_replicates: int = 3
    max_workers: int = 4
    dry_run: bool = False
    workflow_testing: bool = False
    overwrite_outputs: bool = False
    continue_on_error: bool = True
    save_prompts: bool = True
    save_raw_outputs: bool = True
    save_parsed_outputs: bool = True
    save_traces: bool = True


@dataclass
class McpConfig:
    preflight: bool = True
    fail_if_unavailable: bool = False


@dataclass
class EvaluationConfig:
    structural: bool = True
    logical_consistency: bool = True
    efficiency: bool = True
    robustness: bool = True


@dataclass
class TimingConfig:
    track_full_run: bool = True
    track_prompt_construction: bool = True
    track_model_latency: bool = True
    track_metrics: bool = True
    track_io: bool = True
    track_step_latencies: bool = True
    track_mcp_tool_latency: bool = True


@dataclass
class ArchitectureBenchmarkConfig:
    enabled: bool = True
    base_routes: list[str] = field(default_factory=lambda: ["single_call", "cot", "stepwise"])


@dataclass
class OrderBenchmarkConfig:
    enabled: bool = False
    base_routes: list[str] = field(default_factory=list)
    # "all", a range string like "O1-O4", or an explicit list like ["O", "O1"].
    variants: str | list[str] = field(default_factory=list)


@dataclass
class WordingBenchmarkConfig:
    """Phase 3 wording benchmark selector (compact, hypothesis-driven).

    Selects which W-numbered targets to run and how alternate-text sources are
    resolved. The alternate text lives outside this config
    (wording_bench_alternates.py).
    """

    enabled: bool = False
    base_routes: list[str] = field(default_factory=lambda: ["single_call"])
    # "all", a range string like "W1-W5", or an explicit list like ["W1", "W3"].
    targets: str | list[str] = "all"
    default_source: str | None = None
    force_source: str | None = None


@dataclass
class ClusterFilter:
    screen_name: str
    cluster_id: str


@dataclass
class BenchmarkConfig:
    """Top-level benchmark configuration."""

    experiment_id: str = "arch_bench_v1"
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    run: RunConfig = field(default_factory=RunConfig)
    screens_include: list[str] | str = "all"
    clusters_include: list[ClusterFilter] | str = "all"
    mcp: McpConfig = field(default_factory=McpConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    architecture_benchmark: ArchitectureBenchmarkConfig = field(
        default_factory=ArchitectureBenchmarkConfig
    )
    order_benchmark: OrderBenchmarkConfig = field(default_factory=OrderBenchmarkConfig)
    wording_benchmark: WordingBenchmarkConfig = field(default_factory=WordingBenchmarkConfig)

    @property
    def experiment_output_dir(self) -> Path:
        base = self.paths.output_dir
        if self.run.workflow_testing:
            base = base / "_workflow_testing"
            return base / self.experiment_id
        elif self.run.overwrite_outputs:
            # When intentionally overwriting, use consistent path (no timestamp)
            return base / self.experiment_id
        else:
            # Production runs with unique timestamps
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            bundle_suffix = self.paths.bundle_source
            return base / f"{self.experiment_id}_{bundle_suffix}_{timestamp}"


def _resolve_paths(paths_dict: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Resolve relative paths against the base directory."""
    resolved = {}
    for key, val in paths_dict.items():
        p = Path(val)
        resolved[key] = p if p.is_absolute() else base_dir / p
    return resolved


def load_config(config_path: Path) -> BenchmarkConfig:
    """Load a YAML config file and return a validated BenchmarkConfig."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Determine base directory for resolving relative paths
    base_dir = config_path.parent.parent  # configs/ -> phase1_prompt_benchmarking/

    cfg = BenchmarkConfig()
    cfg.experiment_id = raw.get("experiment_id", cfg.experiment_id)

    # Paths
    if "paths" in raw:
        paths_raw = _resolve_paths(raw["paths"], base_dir)
        # Parse bundle_source (defaults to "uniprot")
        bundle_source = raw["paths"].get("bundle_source", "uniprot")
        # Auto-resolve evidence_bundles_dir based on bundle_source if not explicitly set
        if "evidence_bundles_dir" in paths_raw:
            evidence_bundles_dir = paths_raw["evidence_bundles_dir"]
        else:
            evidence_bundles_dir = base_dir / f"benchmark_evidence_bundles_{bundle_source}"

        cfg.paths = PathsConfig(
            benchmark_inputs_dir=paths_raw.get(
                "benchmark_inputs_dir", base_dir / "benchmark_inputs"
            ),
            benchmark_clusters_csv=paths_raw.get(
                "benchmark_clusters_csv", base_dir / "benchmark_inputs" / "benchmark_clusters.csv"
            ),
            evidence_bundles_dir=evidence_bundles_dir,
            output_dir=paths_raw.get("output_dir", base_dir / "1.architecture_testing_outputs"),
            bundle_source=bundle_source,
        )

    # Model
    if "model" in raw:
        m = raw["model"]
        cfg.model = ModelConfig(
            provider=m.get("provider", cfg.model.provider),
            model_name=m.get("model_name", cfg.model.model_name),
            temperature=m.get("temperature", cfg.model.temperature),
            max_tokens=m.get("max_tokens", cfg.model.max_tokens),
            top_p=m.get("top_p"),
            top_k=m.get("top_k"),
            thinking=m.get("thinking"),
        )

    # Run
    if "run" in raw:
        r = raw["run"]
        cfg.run = RunConfig(
            num_replicates=r.get("num_replicates", cfg.run.num_replicates),
            max_workers=r.get("max_workers", cfg.run.max_workers),
            dry_run=r.get("dry_run", cfg.run.dry_run),
            workflow_testing=r.get("workflow_testing", cfg.run.workflow_testing),
            overwrite_outputs=r.get("overwrite_outputs", cfg.run.overwrite_outputs),
            continue_on_error=r.get("continue_on_error", cfg.run.continue_on_error),
            save_prompts=r.get("save_prompts", cfg.run.save_prompts),
            save_raw_outputs=r.get("save_raw_outputs", cfg.run.save_raw_outputs),
            save_parsed_outputs=r.get("save_parsed_outputs", cfg.run.save_parsed_outputs),
            save_traces=r.get("save_traces", cfg.run.save_traces),
        )

    # Architecture benchmark (Phase 1)
    if "architecture_benchmark" in raw:
        ab = raw["architecture_benchmark"]
        cfg.architecture_benchmark = ArchitectureBenchmarkConfig(
            enabled=ab.get("enabled", True),
            base_routes=ab.get("base_routes", cfg.architecture_benchmark.base_routes),
        )
    elif "routes" in raw:
        routes_section = raw["routes"]
        cfg.architecture_benchmark = ArchitectureBenchmarkConfig(
            enabled=True,
            base_routes=routes_section.get("include", cfg.architecture_benchmark.base_routes),
        )

    # Screens
    if "screens" in raw:
        cfg.screens_include = raw["screens"].get("include", "all")

    # Clusters
    if "clusters" in raw:
        clusters_raw = raw["clusters"].get("include", "all")
        if clusters_raw == "all":
            cfg.clusters_include = "all"
        else:
            cfg.clusters_include = [
                ClusterFilter(screen_name=c["screen_name"], cluster_id=str(c["cluster_id"]))
                for c in clusters_raw
            ]

    # MCP
    if "mcp" in raw:
        mc = raw["mcp"]
        cfg.mcp = McpConfig(
            preflight=mc.get("preflight", cfg.mcp.preflight),
            fail_if_unavailable=mc.get("fail_if_unavailable", cfg.mcp.fail_if_unavailable),
        )

    # Evaluation
    if "evaluation" in raw:
        ev = raw["evaluation"]
        cfg.evaluation = EvaluationConfig(
            structural=ev.get("structural", cfg.evaluation.structural),
            logical_consistency=ev.get("logical_consistency", cfg.evaluation.logical_consistency),
            efficiency=ev.get("efficiency", cfg.evaluation.efficiency),
            robustness=ev.get("robustness", cfg.evaluation.robustness),
        )

    # Timing
    if "timing" in raw:
        tm = raw["timing"]
        cfg.timing = TimingConfig(
            track_full_run=tm.get("track_full_run", cfg.timing.track_full_run),
            track_prompt_construction=tm.get(
                "track_prompt_construction", cfg.timing.track_prompt_construction
            ),
            track_model_latency=tm.get("track_model_latency", cfg.timing.track_model_latency),
            track_metrics=tm.get("track_metrics", cfg.timing.track_metrics),
            track_io=tm.get("track_io", cfg.timing.track_io),
            track_step_latencies=tm.get("track_step_latencies", cfg.timing.track_step_latencies),
            track_mcp_tool_latency=tm.get(
                "track_mcp_tool_latency", cfg.timing.track_mcp_tool_latency
            ),
        )

    # Order benchmark (Phase 2)
    if "order_benchmark" in raw:
        ob = raw["order_benchmark"]
        cfg.order_benchmark = OrderBenchmarkConfig(
            enabled=ob.get("enabled", False),
            base_routes=ob.get("base_routes", []),
            variants=ob.get("variants", []),
        )

    # Wording benchmark (Phase 3)
    if "wording_benchmark" in raw:
        wb = raw["wording_benchmark"]
        cfg.wording_benchmark = WordingBenchmarkConfig(
            enabled=wb.get("enabled", False),
            base_routes=wb.get("base_routes", ["single_call"]),
            targets=wb.get("targets", "all"),
            default_source=wb.get("default_source"),
            force_source=wb.get("force_source"),
        )

    return cfg
