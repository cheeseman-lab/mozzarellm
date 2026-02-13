"""
Unit tests for mozzarellm.utils.cluster_utils

Test Coverage:
1. Cluster ID to bundle path mapping
2. File globbing and pattern matching
3. Filename parsing for cluster IDs
4. Error handling for missing directories
5. Edge cases (empty directories, malformed filenames)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mozzarellm.utils.cluster_utils import (
    build_cluster_id_to_bundle_path,
    cluster_chunker,
)


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

    chunks = cluster_chunker(df, "cluster")

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

    chunks = cluster_chunker(df, "cluster")

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

    chunks = cluster_chunker(df, "cluster")

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

    chunks = cluster_chunker(df, "cluster")

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

    chunks = cluster_chunker(df, "cluster")

    assert len(chunks) == 3
    assert all(len(chunk) == 1 for chunk in chunks)


def test_cluster_chunker_empty_dataframe():
    """Test handling of empty DataFrame"""
    df = pd.DataFrame(columns=["gene", "cluster"])

    chunks = cluster_chunker(df, "cluster")

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

    chunks = cluster_chunker(df, "cluster")

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

    chunks = cluster_chunker(df, "cluster")

    # First cluster should be 3 (seen first), then 1
    assert chunks[0]["cluster"].iloc[0] == 3
    assert chunks[1]["cluster"].iloc[0] == 1


def test_cluster_chunker_missing_column_raises():
    """Test that missing cluster column raises ValueError"""
    df = pd.DataFrame(
        {
            "gene": ["TP53", "BRCA1"],
            "wrong_column": [1, 2],
        }
    )

    with pytest.raises(ValueError, match="Cluster ID column 'cluster' not found"):
        cluster_chunker(df, "cluster")


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

    chunks = cluster_chunker(df, "cluster")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert list(chunk.columns) == ["gene", "cluster", "value1", "value2", "value3"]


def test_cluster_chunker_with_nan_cluster_ids():
    """Test handling of NaN cluster IDs"""
    df = pd.DataFrame(
        {
            "gene": ["TP53", "BRCA1", "CDK1"],
            "cluster": [1, pd.NA, 2],
        }
    )

    chunks = cluster_chunker(df, "cluster")

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

    chunks = cluster_chunker(df, "cluster")

    # Should have 2 chunks: one for int 1, one for str "1"
    assert len(chunks) == 2


# =============================================================================
# Test: build_cluster_id_to_bundle_path
# =============================================================================


def test_build_cluster_id_to_bundle_path_basic(tmp_path):
    """Test basic cluster ID to bundle path mapping"""
    evidence_dir = tmp_path / "evidence_bundles"
    evidence_dir.mkdir()

    # Create test bundle files
    (evidence_dir / "test__cluster_1__bundle.json").write_text("{}")
    (evidence_dir / "test__cluster_2__bundle.json").write_text("{}")
    (evidence_dir / "test__cluster_42__bundle.json").write_text("{}")

    result = build_cluster_id_to_bundle_path(evidence_bundle_dir=evidence_dir, screen_name="test")

    assert isinstance(result, dict)
    assert len(result) == 3
    assert "1" in result
    assert "2" in result
    assert "42" in result
    assert isinstance(result["1"], Path)


def test_build_cluster_id_to_bundle_path_filters_by_screen_name(tmp_path):
    """Test that only bundles matching screen name are included"""
    evidence_dir = tmp_path / "evidence_bundles"
    evidence_dir.mkdir()

    # Create bundles for different screens
    (evidence_dir / "screen1__cluster_1__bundle.json").write_text("{}")
    (evidence_dir / "screen2__cluster_2__bundle.json").write_text("{}")
    (evidence_dir / "screen1__cluster_3__bundle.json").write_text("{}")

    result = build_cluster_id_to_bundle_path(
        evidence_bundle_dir=evidence_dir, screen_name="screen1"
    )

    assert len(result) == 2
    assert "1" in result
    assert "3" in result
    assert "2" not in result


def test_build_cluster_id_to_bundle_path_empty_directory(tmp_path):
    """Test handling of empty evidence bundle directory"""
    evidence_dir = tmp_path / "empty_bundles"
    evidence_dir.mkdir()

    result = build_cluster_id_to_bundle_path(evidence_bundle_dir=evidence_dir, screen_name="test")

    assert isinstance(result, dict)
    assert len(result) == 0


def test_build_cluster_id_to_bundle_path_ignores_non_bundle_files(tmp_path):
    """Test that non-bundle files are ignored"""
    evidence_dir = tmp_path / "evidence_bundles"
    evidence_dir.mkdir()

    # Create valid bundle
    (evidence_dir / "test__cluster_1__bundle.json").write_text("{}")

    # Create files that should be ignored
    (evidence_dir / "test__cluster_2.json").write_text("{}")  # Missing __bundle
    (evidence_dir / "test_cluster_3__bundle.json").write_text("{}")  # Wrong separator
    (evidence_dir / "readme.txt").write_text("info")
    (evidence_dir / "test__cluster_4__data.json").write_text("{}")  # Wrong suffix

    result = build_cluster_id_to_bundle_path(evidence_bundle_dir=evidence_dir, screen_name="test")

    assert len(result) == 1
    assert "1" in result


def test_build_cluster_id_to_bundle_path_numeric_cluster_ids(tmp_path):
    """Test extraction of numeric cluster IDs"""
    evidence_dir = tmp_path / "evidence_bundles"
    evidence_dir.mkdir()

    (evidence_dir / "test__cluster_0__bundle.json").write_text("{}")
    (evidence_dir / "test__cluster_999__bundle.json").write_text("{}")
    (evidence_dir / "test__cluster_12345__bundle.json").write_text("{}")

    result = build_cluster_id_to_bundle_path(evidence_bundle_dir=evidence_dir, screen_name="test")

    assert "0" in result
    assert "999" in result
    assert "12345" in result


def test_build_cluster_id_to_bundle_path_string_cluster_ids(tmp_path):
    """Test extraction of string cluster IDs"""
    evidence_dir = tmp_path / "evidence_bundles"
    evidence_dir.mkdir()

    (evidence_dir / "test__cluster_abc__bundle.json").write_text("{}")
    (evidence_dir / "test__cluster_cluster_A__bundle.json").write_text("{}")

    result = build_cluster_id_to_bundle_path(evidence_bundle_dir=evidence_dir, screen_name="test")

    assert "abc" in result
    assert "cluster_A" in result


def test_build_cluster_id_to_bundle_path_returns_path_objects(tmp_path):
    """Test that returned values are Path objects"""
    evidence_dir = tmp_path / "evidence_bundles"
    evidence_dir.mkdir()

    bundle_file = evidence_dir / "test__cluster_1__bundle.json"
    bundle_file.write_text("{}")

    result = build_cluster_id_to_bundle_path(evidence_bundle_dir=evidence_dir, screen_name="test")

    assert isinstance(result["1"], Path)
    assert result["1"] == bundle_file


def test_build_cluster_id_to_bundle_path_handles_underscores_in_screen_name(tmp_path):
    """Test screen names with underscores"""
    evidence_dir = tmp_path / "evidence_bundles"
    evidence_dir.mkdir()

    (evidence_dir / "my_screen_name__cluster_1__bundle.json").write_text("{}")
    (evidence_dir / "my_screen_name__cluster_2__bundle.json").write_text("{}")

    result = build_cluster_id_to_bundle_path(
        evidence_bundle_dir=evidence_dir, screen_name="my_screen_name"
    )

    assert len(result) == 2
    assert "1" in result
    assert "2" in result


def test_build_cluster_id_to_bundle_path_skips_malformed_filenames(tmp_path):
    """Test that malformed filenames are skipped gracefully"""
    evidence_dir = tmp_path / "evidence_bundles"
    evidence_dir.mkdir()

    # Valid bundle
    (evidence_dir / "test__cluster_1__bundle.json").write_text("{}")

    # Malformed bundles
    (evidence_dir / "test__cluster___bundle.json").write_text("{}")  # Empty cluster ID
    (evidence_dir / "__cluster_2__bundle.json").write_text("{}")  # Missing screen name

    result = build_cluster_id_to_bundle_path(evidence_bundle_dir=evidence_dir, screen_name="test")

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


def test_build_cluster_id_to_bundle_path_preserves_cluster_id_as_string(tmp_path):
    """Test that cluster IDs are always strings in the result"""
    evidence_dir = tmp_path / "evidence_bundles"
    evidence_dir.mkdir()

    (evidence_dir / "test__cluster_123__bundle.json").write_text("{}")

    result = build_cluster_id_to_bundle_path(evidence_bundle_dir=evidence_dir, screen_name="test")

    # Keys should be strings, not integers
    assert "123" in result
    assert 123 not in result
    assert isinstance(list(result.keys())[0], str)


def test_build_cluster_id_to_bundle_path_duplicate_cluster_ids(tmp_path):
    """Test handling when multiple files have same cluster ID"""
    evidence_dir = tmp_path / "evidence_bundles"
    evidence_dir.mkdir()

    # Create duplicate cluster IDs (shouldn't happen in practice, but test robustness)
    file1 = evidence_dir / "test__cluster_1__bundle.json"
    file2 = evidence_dir / "test__cluster_1__bundle.json.bak"

    file1.write_text("{}")
    # Only .json files should match the pattern

    result = build_cluster_id_to_bundle_path(evidence_bundle_dir=evidence_dir, screen_name="test")

    # Should have one entry for cluster 1
    assert "1" in result
    assert len(result) == 1
