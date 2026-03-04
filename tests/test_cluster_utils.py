"""
Unit tests for mozzarellm.utils.cluster_utils
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mozzarellm.utils.cluster_utils import (
    build_cluster_id_to_bundle_path,
    cluster_chunker,
)


####################### TEST CONSTANTS #######################

CLUSTER_COL = "cluster"
SCREEN_NAME = "test"

####################### FIXTURES #######################


@pytest.fixture
def evidence_dir(tmp_path):
    """Empty evidence_bundles directory under tmp_path."""
    d = tmp_path / "evidence_bundles"
    d.mkdir()
    return d


####################### HELPERS #######################


def _bundle_file(directory, screen: str, cluster_id) -> Path:
    """Write an empty bundle JSON and return its Path."""
    p = directory / f"{screen}__cluster_{cluster_id}__bundle.json"
    p.write_text("{}")
    return p


# =============================================================================
# Test: cluster_chunker
# =============================================================================


def test_cluster_chunker_basic():
    """Test basic chunking of DataFrame by cluster ID"""
    df = pd.DataFrame(
        {
            "gene": ["TP53", "BRCA1", "CDK1", "MYC"],
            "cluster": [1, 1, 2, 2],
            "value": [10, 20, 30, 40],
        }
    )

    chunks = cluster_chunker(df, CLUSTER_COL)

    assert len(chunks) == 2
    assert len(chunks[0]) == 2  # Cluster 1 has 2 genes
    assert len(chunks[1]) == 2  # Cluster 2 has 2 genes
    assert all(isinstance(chunk, pd.DataFrame) for chunk in chunks)


def test_cluster_chunker_preserves_row_order():
    """Test that row order within each cluster is preserved"""
    df = pd.DataFrame(
        {
            "gene": ["A", "B", "C", "D"],
            "cluster": [1, 1, 1, 1],
            "order": [1, 2, 3, 4],
        }
    )

    chunks = cluster_chunker(df, CLUSTER_COL)

    assert len(chunks) == 1
    assert list(chunks[0]["gene"]) == ["A", "B", "C", "D"]
    assert list(chunks[0]["order"]) == [1, 2, 3, 4]


def test_cluster_chunker_handles_interleaved_clusters():
    """Test chunking with interleaved cluster IDs"""
    df = pd.DataFrame(
        {
            "gene": ["A", "B", "C", "D", "E", "F"],
            "cluster": [1, 2, 1, 2, 1, 2],
            "value": [10, 20, 30, 40, 50, 60],
        }
    )

    chunks = cluster_chunker(df, CLUSTER_COL)

    assert len(chunks) == 2
    # Cluster 1 should have genes A, C, E in that order
    cluster_1 = chunks[0]
    assert list(cluster_1["gene"]) == ["A", "C", "E"]
    # Cluster 2 should have genes B, D, F in that order
    cluster_2 = chunks[1]
    assert list(cluster_2["gene"]) == ["B", "D", "F"]


def test_cluster_chunker_single_cluster():
    """Test DataFrame with only one cluster"""
    df = pd.DataFrame(
        {
            "gene": ["TP53", "BRCA1", "CDK1"],
            "cluster": [1, 1, 1],
        }
    )

    chunks = cluster_chunker(df, CLUSTER_COL)

    assert len(chunks) == 1
    assert len(chunks[0]) == 3


def test_cluster_chunker_single_gene_per_cluster():
    """Test DataFrame where each cluster has only one gene"""
    df = pd.DataFrame(
        {
            "gene": ["TP53", "BRCA1", "CDK1"],
            "cluster": [1, 2, 3],
        }
    )

    chunks = cluster_chunker(df, CLUSTER_COL)

    assert len(chunks) == 3
    assert all(len(chunk) == 1 for chunk in chunks)


def test_cluster_chunker_empty_dataframe():
    """Test handling of empty DataFrame"""
    df = pd.DataFrame(columns=["gene", "cluster"])

    chunks = cluster_chunker(df, CLUSTER_COL)

    assert isinstance(chunks, list)
    assert len(chunks) == 0


def test_cluster_chunker_string_cluster_ids():
    """Test chunking with string cluster IDs"""
    df = pd.DataFrame(
        {
            "gene": ["TP53", "BRCA1", "CDK1"],
            "cluster": ["A", "A", "B"],
        }
    )

    chunks = cluster_chunker(df, CLUSTER_COL)

    assert len(chunks) == 2
    assert len(chunks[0]) == 2  # Cluster A
    assert len(chunks[1]) == 1  # Cluster B


def test_cluster_chunker_preserves_first_seen_order():
    """Test that sort=False preserves first-seen cluster order"""
    df = pd.DataFrame(
        {
            "gene": ["A", "B", "C", "D"],
            "cluster": [3, 1, 3, 1],
        }
    )

    chunks = cluster_chunker(df, CLUSTER_COL)

    # First cluster should be 3 (seen first), then 1
    assert chunks[0][CLUSTER_COL].iloc[0] == 3
    assert chunks[1][CLUSTER_COL].iloc[0] == 1


def test_cluster_chunker_missing_column_raises():
    """Test that missing cluster column raises ValueError"""
    df = pd.DataFrame(
        {
            "gene": ["TP53", "BRCA1"],
            "wrong_column": [1, 2],
        }
    )

    with pytest.raises(ValueError, match=f"Cluster ID column '{CLUSTER_COL}' not found"):
        cluster_chunker(df, CLUSTER_COL)


def test_cluster_chunker_preserves_all_columns():
    """Test that all DataFrame columns are preserved in chunks"""
    df = pd.DataFrame(
        {
            "gene": ["TP53", "BRCA1"],
            "cluster": [1, 1],
            "value1": [10, 20],
            "value2": ["a", "b"],
            "value3": [True, False],
        }
    )

    chunks = cluster_chunker(df, CLUSTER_COL)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert list(chunk.columns) == ["gene", CLUSTER_COL, "value1", "value2", "value3"]


def test_cluster_chunker_with_nan_cluster_ids():
    """Test handling of NaN cluster IDs"""
    df = pd.DataFrame(
        {
            "gene": ["TP53", "BRCA1", "CDK1"],
            "cluster": [1, pd.NA, 2],
        }
    )

    chunks = cluster_chunker(df, CLUSTER_COL)

    # Should create separate chunks for each unique value including NaN
    assert len(chunks) >= 2


def test_cluster_chunker_numeric_vs_string_cluster_ids():
    """Test that numeric and string cluster IDs are treated as different"""
    df = pd.DataFrame(
        {
            "gene": ["A", "B", "C"],
            "cluster": [1, "1", 1],
        }
    )

    chunks = cluster_chunker(df, CLUSTER_COL)

    # Should have 2 chunks: one for int 1, one for str "1"
    assert len(chunks) == 2


# =============================================================================
# Test: build_cluster_id_to_bundle_path
# =============================================================================


def test_build_cluster_id_to_bundle_path_basic(evidence_dir):
    """Test basic cluster ID to bundle path mapping"""
    _bundle_file(evidence_dir, SCREEN_NAME, 1)
    _bundle_file(evidence_dir, SCREEN_NAME, 2)
    _bundle_file(evidence_dir, SCREEN_NAME, 42)

    result = build_cluster_id_to_bundle_path(
        evidence_bundle_dir=evidence_dir, screen_name=SCREEN_NAME
    )

    assert isinstance(result, dict)
    assert len(result) == 3
    assert "1" in result
    assert "2" in result
    assert "42" in result
    assert isinstance(result["1"], Path)


def test_build_cluster_id_to_bundle_path_filters_by_screen_name(evidence_dir):
    """Test that only bundles matching screen name are included"""
    _bundle_file(evidence_dir, "screen1", 1)
    _bundle_file(evidence_dir, "screen2", 2)
    _bundle_file(evidence_dir, "screen1", 3)

    result = build_cluster_id_to_bundle_path(
        evidence_bundle_dir=evidence_dir, screen_name="screen1"
    )

    assert len(result) == 2
    assert "1" in result
    assert "3" in result
    assert "2" not in result


def test_build_cluster_id_to_bundle_path_empty_directory(evidence_dir):
    """Test handling of empty evidence bundle directory"""
    result = build_cluster_id_to_bundle_path(
        evidence_bundle_dir=evidence_dir, screen_name=SCREEN_NAME
    )

    assert isinstance(result, dict)
    assert len(result) == 0


def test_build_cluster_id_to_bundle_path_ignores_non_bundle_files(evidence_dir):
    """Test that non-bundle files are ignored"""
    _bundle_file(evidence_dir, SCREEN_NAME, 1)

    # Create files that should be ignored
    (evidence_dir / f"{SCREEN_NAME}__cluster_2.json").write_text("{}")  # Missing __bundle
    (evidence_dir / f"{SCREEN_NAME}_cluster_3__bundle.json").write_text("{}")  # Wrong separator
    (evidence_dir / "readme.txt").write_text("info")
    (evidence_dir / f"{SCREEN_NAME}__cluster_4__data.json").write_text("{}")  # Wrong suffix

    result = build_cluster_id_to_bundle_path(
        evidence_bundle_dir=evidence_dir, screen_name=SCREEN_NAME
    )

    assert len(result) == 1
    assert "1" in result


def test_build_cluster_id_to_bundle_path_numeric_cluster_ids(evidence_dir):
    """Test extraction of numeric cluster IDs"""
    _bundle_file(evidence_dir, SCREEN_NAME, 0)
    _bundle_file(evidence_dir, SCREEN_NAME, 999)
    _bundle_file(evidence_dir, SCREEN_NAME, 12345)

    result = build_cluster_id_to_bundle_path(
        evidence_bundle_dir=evidence_dir, screen_name=SCREEN_NAME
    )

    assert "0" in result
    assert "999" in result
    assert "12345" in result


def test_build_cluster_id_to_bundle_path_string_cluster_ids(evidence_dir):
    """Test extraction of string cluster IDs"""
    _bundle_file(evidence_dir, SCREEN_NAME, "abc")
    _bundle_file(evidence_dir, SCREEN_NAME, "cluster_A")

    result = build_cluster_id_to_bundle_path(
        evidence_bundle_dir=evidence_dir, screen_name=SCREEN_NAME
    )

    assert "abc" in result
    assert "cluster_A" in result


def test_build_cluster_id_to_bundle_path_returns_path_objects(evidence_dir):
    """Test that returned values are Path objects"""
    bundle_file = _bundle_file(evidence_dir, SCREEN_NAME, 1)

    result = build_cluster_id_to_bundle_path(
        evidence_bundle_dir=evidence_dir, screen_name=SCREEN_NAME
    )

    assert isinstance(result["1"], Path)
    assert result["1"] == bundle_file


def test_build_cluster_id_to_bundle_path_handles_underscores_in_screen_name(evidence_dir):
    """Test screen names with underscores"""
    _bundle_file(evidence_dir, "my_screen_name", 1)
    _bundle_file(evidence_dir, "my_screen_name", 2)

    result = build_cluster_id_to_bundle_path(
        evidence_bundle_dir=evidence_dir, screen_name="my_screen_name"
    )

    assert len(result) == 2
    assert "1" in result
    assert "2" in result


def test_build_cluster_id_to_bundle_path_skips_malformed_filenames(evidence_dir):
    """Test that malformed filenames are skipped gracefully"""
    _bundle_file(evidence_dir, SCREEN_NAME, 1)

    # Malformed bundles
    (evidence_dir / f"{SCREEN_NAME}__cluster___bundle.json").write_text("{}")  # Empty cluster ID
    (evidence_dir / "__cluster_2__bundle.json").write_text("{}")  # Missing screen name

    result = build_cluster_id_to_bundle_path(
        evidence_bundle_dir=evidence_dir, screen_name=SCREEN_NAME
    )

    # Should only include valid bundle
    assert len(result) == 1
    assert "1" in result


def test_build_cluster_id_to_bundle_path_nonexistent_directory():
    """Test handling of non-existent directory"""
    nonexistent_dir = Path("/nonexistent/path/to/bundles")

    # Should either raise FileNotFoundError or return empty dict
    try:
        result = build_cluster_id_to_bundle_path(
            evidence_bundle_dir=nonexistent_dir, screen_name="test"
        )
        # If it doesn't raise, should return empty dict
        assert isinstance(result, dict)
        assert len(result) == 0
    except FileNotFoundError:
        # This is also acceptable behavior
        pass


def test_build_cluster_id_to_bundle_path_preserves_cluster_id_as_string(evidence_dir):
    """Test that cluster IDs are always strings in the result"""
    _bundle_file(evidence_dir, SCREEN_NAME, 123)

    result = build_cluster_id_to_bundle_path(
        evidence_bundle_dir=evidence_dir, screen_name=SCREEN_NAME
    )

    # Keys should be strings, not integers
    assert "123" in result
    assert 123 not in result
    assert isinstance(list(result.keys())[0], str)


def test_build_cluster_id_to_bundle_path_duplicate_cluster_ids(evidence_dir):
    """Test handling when multiple files have same cluster ID"""
    _bundle_file(evidence_dir, SCREEN_NAME, 1)
    (evidence_dir / f"{SCREEN_NAME}__cluster_1__bundle.json.bak").write_text(
        "{}"
    )  # Only .json should match

    result = build_cluster_id_to_bundle_path(
        evidence_bundle_dir=evidence_dir, screen_name=SCREEN_NAME
    )

    # Should have one entry for cluster 1
    assert "1" in result
    assert len(result) == 1
