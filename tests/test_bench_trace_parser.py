"""Unit tests for bench_trace_parser.py — filename parsing, gene-row
extraction, and end-to-end CSV writing."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.phase1_prompt_benchmarking.architecture_benchmarking_workflow.bench_trace_parser import (
    CSV_COLUMNS,
    TRACE_FILENAME_RE,
    _extract_gene_rows,
    extract_predictions_from_traces,
    parse_trace_filename,
    write_prediction_csvs,
)


# ---- helpers ---------------------------------------------------------------


def _make_meta(**overrides) -> dict:
    base = {
        "cluster_id": "21",
        "screen_name": "denali",
        "route": "3a",
        "replicate": 1,
        "run_id": "exp__3a__denali__cluster_21__rep_1",
    }
    base.update(overrides)
    return base


def _make_trace_json(
    *,
    route: str = "3a",
    run_id: str = "exp__3a__denali__cluster_21__rep_1",
    established_genes: list[str] | None = None,
    include_parsed_output: bool = True,
) -> dict:
    trace: dict = {"route": route, "run_id": run_id}
    if include_parsed_output:
        trace["parsed_output"] = {
            "established_genes": established_genes or ["GeneX"],
            "dominant_process": "cell cycle",
            "pathway_confidence": "Medium",
        }
    return trace


# ===========================================================================
# 2.1  Filename parsing
# ===========================================================================


class TestFilenameParser:
    def test_phase1_standard_route(self):
        result = parse_trace_filename(Path("exp__3a__denali__cluster_21__rep_1.json"))
        assert result["experiment_id"] == "exp"
        assert result["route"] == "3a"
        assert result["screen_name"] == "denali"
        assert result["cluster_id"] == "21"
        assert result["replicate"] == 1
        assert result["filename_parse_status"] == "success"

    def test_phase1_mcp_route(self):
        result = parse_trace_filename(Path("exp__3a_mcp__denali__cluster_21__rep_1.json"))
        assert result["route"] == "3a_mcp"
        assert result["filename_parse_status"] == "success"

    def test_phase1_stepwise(self):
        result = parse_trace_filename(Path("exp__3c__screen__cluster_42__rep_3.json"))
        assert result["route"] == "3c"
        assert result["replicate"] == 3
        assert result["filename_parse_status"] == "success"

    def test_phase2_order_variant(self):
        result = parse_trace_filename(
            Path("exp__3b_O1_late_screen_context__denali__cluster_21__rep_1.json")
        )
        assert result["route"] == "3b_O1_late_screen_context"
        assert result["filename_parse_status"] == "success"

    def test_phase2_mcp_order_variant(self):
        result = parse_trace_filename(
            Path("exp__3a_mcp_O3_early_output_format__denali__cluster_77__rep_2.json")
        )
        assert result["route"] == "3a_mcp_O3_early_output_format"
        assert result["replicate"] == 2
        assert result["cluster_id"] == "77"
        assert result["filename_parse_status"] == "success"

    def test_phase3_wording_route(self):
        result = parse_trace_filename(
            Path(
                "exp__3a_wording_v1_W3_imperative_gene_classification"
                "__denali__cluster_21__rep_1.json"
            )
        )
        assert result["filename_parse_status"] == "success"
        assert result["route"] == "3a_wording_v1_W3_imperative_gene_classification"

    def test_underscore_heavy_screen_name(self):
        result = parse_trace_filename(
            Path("exp__3a__aconcagua_interphase_shuffled__cluster_244__rep_1.json")
        )
        assert result["screen_name"] == "aconcagua_interphase_shuffled"
        assert result["filename_parse_status"] == "success"

    def test_unparseable_filename(self):
        result = parse_trace_filename(Path("garbage.json"))
        assert result["filename_parse_status"] == "failed"
        assert result["experiment_id"] is None
        assert result["route"] is None
        assert result["screen_name"] is None
        assert result["cluster_id"] is None
        assert result["replicate"] is None

    def test_multi_digit_cluster_id(self):
        result = parse_trace_filename(Path("exp__3b__denali__cluster_109__rep_1.json"))
        assert result["cluster_id"] == "109"
        assert result["filename_parse_status"] == "success"


# ===========================================================================
# 2.2  Gene-row extraction
# ===========================================================================


class TestGeneRowExtraction:
    def test_standard_parsed_output(self):
        parsed_output = {
            "established_genes": ["GeneA", "GeneB"],
            "novel_role_genes": [
                {"gene": "GeneC", "class": "NO_EVIDENCE", "rationale": "r", "evidence": "e"},
            ],
            "uncharacterized_genes": [
                {"gene": "GeneD", "class": "DARK_GENE", "rationale": "r", "evidence": "e"},
            ],
            "dominant_process": "DNA repair",
            "pathway_confidence": "High",
        }
        meta = _make_meta()
        rows = _extract_gene_rows(parsed_output, meta, "x.json")

        assert len(rows) == 4

        genes_by_symbol = {r["gene_symbol"]: r for r in rows}

        assert genes_by_symbol["GeneA"]["predicted_class"] == "ESTABLISHED"
        assert genes_by_symbol["GeneB"]["predicted_class"] == "ESTABLISHED"

        assert genes_by_symbol["GeneC"]["predicted_class"] == "NOVEL_ROLE"
        assert genes_by_symbol["GeneC"]["predicted_subclass"] == "NO_EVIDENCE"

        assert genes_by_symbol["GeneD"]["predicted_class"] == "UNCHARACTERIZED"
        assert genes_by_symbol["GeneD"]["predicted_subclass"] == "DARK_GENE"

        for row in rows:
            assert row["pathway"] == "DNA repair"
            assert row["pathway_confidence"] == "High"

    def test_empty_parsed_output(self):
        meta = _make_meta()
        assert _extract_gene_rows(None, meta, "x.json") == []
        assert _extract_gene_rows({}, meta, "x.json") == []

    def test_missing_gene_lists(self):
        parsed_output = {"dominant_process": "signaling"}
        meta = _make_meta()
        rows = _extract_gene_rows(parsed_output, meta, "x.json")
        assert rows == []

    def test_order_metadata_propagated(self):
        parsed_output = {
            "established_genes": ["GeneZ"],
            "dominant_process": "transport",
            "pathway_confidence": "Low",
        }
        meta = _make_meta(
            base_route="3b",
            order_variant="late_screen_context",
            order_hypothesis="screen_context_anchoring",
            component_order="system,screen_context,gene_list,output_format",
        )
        rows = _extract_gene_rows(parsed_output, meta, "trace.json")

        assert len(rows) == 1
        row = rows[0]
        assert row["base_route"] == "3b"
        assert row["order_variant"] == "late_screen_context"
        assert row["order_hypothesis"] == "screen_context_anchoring"
        assert row["component_order"] == "system,screen_context,gene_list,output_format"


# ===========================================================================
# 2.3  End-to-end
# ===========================================================================


class TestEndToEnd:
    def _write_trace(self, directory: Path, filename: str, data: dict) -> Path:
        path = directory / filename
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_extract_and_write_csvs(self, tmp_path: Path):
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()

        self._write_trace(
            traces_dir,
            "exp__3a__denali__cluster_21__rep_1.json",
            _make_trace_json(
                route="3a",
                run_id="exp__3a__denali__cluster_21__rep_1",
                established_genes=["G1", "G2"],
            ),
        )
        self._write_trace(
            traces_dir,
            "exp__3a__denali__cluster_22__rep_1.json",
            _make_trace_json(
                route="3a",
                run_id="exp__3a__denali__cluster_22__rep_1",
                established_genes=["G3"],
            ),
        )
        self._write_trace(
            traces_dir,
            "exp__3b__denali__cluster_21__rep_1.json",
            _make_trace_json(
                route="3b",
                run_id="exp__3b__denali__cluster_21__rep_1",
                established_genes=["G4"],
            ),
        )

        route_rows = extract_predictions_from_traces(traces_dir)
        assert "3a" in route_rows
        assert "3b" in route_rows
        assert len(route_rows["3a"]) == 3
        assert len(route_rows["3b"]) == 1

        output_dir = tmp_path / "output"
        written = write_prediction_csvs(route_rows, output_dir, "exp", overwrite=True)

        assert len(written) == 2
        assert all(p.exists() for p in written)

        for csv_path in written:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                assert list(reader.fieldnames) == CSV_COLUMNS
                rows = list(reader)
                if "3a" in csv_path.name:
                    assert len(rows) == 3
                else:
                    assert len(rows) == 1

    def test_trace_missing_parsed_output(self, tmp_path: Path):
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()

        self._write_trace(
            traces_dir,
            "exp__3a__denali__cluster_21__rep_1.json",
            _make_trace_json(
                route="3a",
                run_id="exp__3a__denali__cluster_21__rep_1",
                include_parsed_output=False,
            ),
        )
        self._write_trace(
            traces_dir,
            "exp__3a__denali__cluster_22__rep_1.json",
            _make_trace_json(
                route="3a",
                run_id="exp__3a__denali__cluster_22__rep_1",
                established_genes=["G1"],
            ),
        )

        route_rows = extract_predictions_from_traces(traces_dir)
        assert "3a" in route_rows
        assert len(route_rows["3a"]) == 1

    def test_trace_invalid_json(self, tmp_path: Path):
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()

        bad = traces_dir / "exp__3a__denali__cluster_21__rep_1.json"
        bad.write_text("{not valid json!!!}", encoding="utf-8")

        self._write_trace(
            traces_dir,
            "exp__3a__denali__cluster_22__rep_1.json",
            _make_trace_json(
                route="3a",
                run_id="exp__3a__denali__cluster_22__rep_1",
                established_genes=["G1"],
            ),
        )

        route_rows = extract_predictions_from_traces(traces_dir)
        assert "3a" in route_rows
        assert len(route_rows["3a"]) == 1
