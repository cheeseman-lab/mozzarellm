"""Benchmark configuration dataclasses (built programmatically by bench_experiment)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


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
    inputs_dir: Path = Path("inputs")
    benchmark_clusters_csv: Path = Path("inputs/benchmark_clusters.csv")
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
