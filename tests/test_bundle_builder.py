"""
Unit tests for mozzarellm.pipeline.bundle_builder
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import pytest

from mozzarellm.pipeline.bundle_builder import (
    get_or_append_stable_accession,
    build_evidence_bundles,
    add_functional_annotations_to_chunk,
    _lookup_accession,
    DEFAULT_ACCESSION_COL,
)

####################### TEST PARAMETERS #######################


TEST_SCREEN_NAME = "test"
ORGANISM_ID = 9606  # human

# required cluster df
TEST_CLUSTER_DF = pd.DataFrame({"gene_symbol": ["TP53", "BRCA1"], "cluster": [1, 1]})
LARGER_TEST_CLUSTER_DF = pd.DataFrame(
    {"gene_symbol": ["TP53", "BRCA1", "AATF", "BYSL"], "cluster": [1, 1, 2, 2]}
)
CLUSTER_ID_COLUMN = "cluster"
GENE_COLUMN = "gene_symbol"
FEATURE_COLUMNS = [
    "up_features",
    "down_features",
]

# optional user-provided accession table
ACCESSION_TABLE_GENE_COLUMN = "gene"
STABLE_ACCESSION_COLUMN = "stable_id"


####################### FIXTURES #######################


@pytest.fixture
def mock_uniprot_client():
    """Fixture that mocks expected behavior of UniProtClient methods"""
    mock_client = Mock()

    # Mock get_accession_from_gene_symbol
    def get_accession_side_effect(gene_symbol, *args, **kwargs):
        accession_map = {"TP53": "P04637", "BRCA1": "P38398", "AATF": "Q9NY61", "BYSL": "Q13895"}
        return accession_map.get(gene_symbol, "")

    mock_client.get_accession_from_gene_symbol.side_effect = get_accession_side_effect

    # Mock fetch_functional_annotations
    def fetch_annotations_side_effect(chunk, accession_col):
        """Return mock annotations for the given chunk - only accession and annotation columns"""
        annotation_map = {
            "P04637": "Tumor suppressor",
            "P38398": "DNA repair",
            "Q9NY61": "Apoptosis-antagonizing transcription factor",
            "Q13895": "Bystin-like protein",
        }

        # Return only accession and annotation columns (matching real UniProtClient behavior)
        annotations = pd.DataFrame(
            {
                accession_col: chunk[accession_col].unique(),
            }
        )
        annotations["UniProt_functional_annotation"] = annotations[accession_col].map(
            lambda acc: annotation_map.get(acc, "Unknown function")
        )
        return annotations

    mock_client.fetch_functional_annotations.side_effect = fetch_annotations_side_effect

    return mock_client


####################### TEST FUNCTIONS #######################


# =============================================================================
# Test: get_or_append_stable_accession - External Table
# =============================================================================


## helper function tests
@pytest.mark.parametrize("gene_symbol", [np.nan, ""])
def test__lookup_accession_helper_graceful_failure(gene_symbol, mock_uniprot_client):
    """Test lookup accession helper function with NaN"""
    result = _lookup_accession(
        gene_symbol=gene_symbol,
        organism_id=ORGANISM_ID,
        warn_on_fallback=False,
        uniprot_client=mock_uniprot_client,  # no call expected
    )
    assert result == ""
    assert mock_uniprot_client.get_accession_from_gene_symbol.call_count == 0


def test__lookup_accession_helper_nontargeting(mock_uniprot_client):
    """Test lookup accession helper function with non-target gene"""
    result = _lookup_accession(
        gene_symbol="nontargeting_0_2",
        organism_id=ORGANISM_ID,
        warn_on_fallback=False,
        uniprot_client=mock_uniprot_client,  # no call expected
    )
    assert result == "NON_TARGETING_CONTROL"
    assert mock_uniprot_client.get_accession_from_gene_symbol.call_count == 0


def test__lookup_accession_helper_target(mock_uniprot_client):
    """Test lookup accession helper function with target gene"""
    result = _lookup_accession(
        gene_symbol="TP53",
        organism_id=ORGANISM_ID,
        warn_on_fallback=False,
        uniprot_client=mock_uniprot_client,  # Mocked client should be called
    )
    assert result == "P04637"
    assert mock_uniprot_client.get_accession_from_gene_symbol.call_count == 1


## appended accessions: well-formatted CSV file
def test_get_or_append_stable_accession_from_external_table_csv(tmp_path):
    """Test appending accessions from external table"""

    # create test csv
    test_accession_table = tmp_path / "test_accessions.csv"
    accession_df = pd.DataFrame(
        {"gene": ["TP53", "BRCA1"], f"{STABLE_ACCESSION_COLUMN}": ["P04637", "P38398"]}
    )
    accession_df.to_csv(test_accession_table, index=False)

    result = get_or_append_stable_accession(
        screen_name=TEST_SCREEN_NAME,
        cluster_df=TEST_CLUSTER_DF,
        gene_column=GENE_COLUMN,
        organism_id=ORGANISM_ID,
        warn_on_fallback=False,
        accession_table=test_accession_table,
        accession_col=STABLE_ACCESSION_COLUMN,
        accession_table_gene_col=ACCESSION_TABLE_GENE_COLUMN,
        accession_table_sheetname=None,
        accession_table_sep=None,
    )

    assert STABLE_ACCESSION_COLUMN in result.columns
    assert result.loc[result[GENE_COLUMN] == "TP53", STABLE_ACCESSION_COLUMN].values[0] == "P04637"
    assert result.loc[result[GENE_COLUMN] == "BRCA1", STABLE_ACCESSION_COLUMN].values[0] == "P38398"


## appended accessions: well-formatted TSV file
def test_get_or_append_stable_accession_from_external_table_tsv(tmp_path):
    """Test appending accessions from external table"""

    # create test tsv
    test_accession_table = tmp_path / "test_accessions.tsv"
    accession_df = pd.DataFrame(
        {"gene": ["TP53", "BRCA1"], f"{STABLE_ACCESSION_COLUMN}": ["P04637", "P38398"]}
    )
    accession_df.to_csv(test_accession_table, index=False, sep="\t")

    result = get_or_append_stable_accession(
        screen_name=TEST_SCREEN_NAME,
        cluster_df=TEST_CLUSTER_DF,
        gene_column=GENE_COLUMN,
        organism_id=ORGANISM_ID,
        warn_on_fallback=False,
        accession_table=test_accession_table,
        accession_col=STABLE_ACCESSION_COLUMN,  # also testing that this can be different from the default column name
        accession_table_gene_col=ACCESSION_TABLE_GENE_COLUMN,
        accession_table_sheetname=None,
        accession_table_sep="\t",
    )

    assert STABLE_ACCESSION_COLUMN in result.columns
    assert result.loc[result[GENE_COLUMN] == "TP53", STABLE_ACCESSION_COLUMN].values[0] == "P04637"
    assert result.loc[result[GENE_COLUMN] == "BRCA1", STABLE_ACCESSION_COLUMN].values[0] == "P38398"


## appended accessions: well-formatted Excel file
def test_get_or_append_stable_accession_from_external_table_excel(tmp_path):
    """Test appending accessions from Excel file"""

    # create test excel
    test_accession_table = tmp_path / "test_accessions.xlsx"
    accession_df = pd.DataFrame({"gene": ["TP53"], f"{STABLE_ACCESSION_COLUMN}": ["P04637"]})
    accession_df.to_excel(test_accession_table, index=False, sheet_name="Sheet1")

    result = get_or_append_stable_accession(
        screen_name="test",
        cluster_df=TEST_CLUSTER_DF,
        gene_column=GENE_COLUMN,
        accession_table=test_accession_table,
        accession_table_gene_col=ACCESSION_TABLE_GENE_COLUMN,
        accession_col=STABLE_ACCESSION_COLUMN,
        accession_table_sheetname="Sheet1",
        organism_id=ORGANISM_ID,
        warn_on_fallback=False,
    )

    assert STABLE_ACCESSION_COLUMN in result.columns
    assert result.loc[0, STABLE_ACCESSION_COLUMN] == "P04637"


## appended accessions: user-provided accession table with duplicate entries
def test_get_or_append_stable_accession_merge_with_duplicate_entries(tmp_path):
    """Test that duplicate gene entries in accession table are handled correctly"""
    # Create test accession table with duplicates
    test_accession_table = tmp_path / "test_accessions_with_duplicates.csv"
    accession_df = pd.DataFrame(
        {
            "gene": ["TP53", "BRCA1", "TP53"],
            f"{STABLE_ACCESSION_COLUMN}": ["P04637", "P38398", "P04637"],
        }
    )
    accession_df.to_csv(test_accession_table, index=False)

    result = get_or_append_stable_accession(
        screen_name=TEST_SCREEN_NAME,
        cluster_df=TEST_CLUSTER_DF,
        gene_column=GENE_COLUMN,
        accession_table=test_accession_table,
        accession_table_gene_col=ACCESSION_TABLE_GENE_COLUMN,
        accession_col=STABLE_ACCESSION_COLUMN,
        organism_id=ORGANISM_ID,
        warn_on_fallback=False,
    )

    assert STABLE_ACCESSION_COLUMN in result.columns
    assert len(result) == 2  # Should have 2 rows (TP53, BRCA1)
    assert result.loc[result[GENE_COLUMN] == "TP53", STABLE_ACCESSION_COLUMN].values[0] == "P04637"
    assert result.loc[result[GENE_COLUMN] == "BRCA1", STABLE_ACCESSION_COLUMN].values[0] == "P38398"


## appended accessions: missing accessions after merge (partial fallback case)
def test_get_or_append_stable_accession_missing_accessions_fallback(tmp_path, mock_uniprot_client):
    """Test fallback to UniProt API when some genes are missing from accession table"""
    # Create test csv with only 2 genes, but cluster_df has 4 genes
    test_accession_table = tmp_path / "test_accessions_partial.csv"
    accession_df = pd.DataFrame(
        {"gene": ["TP53", "BRCA1"], f"{STABLE_ACCESSION_COLUMN}": ["P04637", "P38398"]}
    )
    accession_df.to_csv(test_accession_table, index=False)

    # Use larger cluster df with 4 genes (AATF and BYSL will be missing)
    with patch(
        "mozzarellm.pipeline.bundle_builder.UniProtClient", return_value=mock_uniprot_client
    ):
        result = get_or_append_stable_accession(
            screen_name=TEST_SCREEN_NAME,
            cluster_df=LARGER_TEST_CLUSTER_DF,
            gene_column=GENE_COLUMN,
            accession_table=test_accession_table,
            accession_table_gene_col=ACCESSION_TABLE_GENE_COLUMN,
            accession_col=STABLE_ACCESSION_COLUMN,
            organism_id=ORGANISM_ID,
            warn_on_fallback=False,
        )

    # Verify all 4 genes have accessions
    assert STABLE_ACCESSION_COLUMN in result.columns
    assert len(result) == 4
    # From table
    assert result.loc[result[GENE_COLUMN] == "TP53", STABLE_ACCESSION_COLUMN].values[0] == "P04637"
    assert result.loc[result[GENE_COLUMN] == "BRCA1", STABLE_ACCESSION_COLUMN].values[0] == "P38398"
    # From UniProt fallback
    assert result.loc[result[GENE_COLUMN] == "AATF", STABLE_ACCESSION_COLUMN].values[0] == "Q9NY61"
    assert result.loc[result[GENE_COLUMN] == "BYSL", STABLE_ACCESSION_COLUMN].values[0] == "Q13895"
    # Verify UniProt was called for missing genes only (2 calls)
    assert mock_uniprot_client.get_accession_from_gene_symbol.call_count == 2


## assigned accessions: no accession table provided (complete fallback case)
def test_get_or_append_stable_accession_no_accessions_fallback(mock_uniprot_client):
    """Test complete fallback to UniProt API when no accession table is provided"""
    with patch(
        "mozzarellm.pipeline.bundle_builder.UniProtClient", return_value=mock_uniprot_client
    ):
        result = get_or_append_stable_accession(
            screen_name=TEST_SCREEN_NAME,
            cluster_df=LARGER_TEST_CLUSTER_DF,
            gene_column=GENE_COLUMN,
            accession_table=None,
            accession_col=None,
            organism_id=ORGANISM_ID,
            warn_on_fallback=False,
        )

    assert DEFAULT_ACCESSION_COL in result.columns
    assert len(result) == 4
    assert result.loc[result[GENE_COLUMN] == "TP53", DEFAULT_ACCESSION_COL].values[0] == "P04637"
    assert result.loc[result[GENE_COLUMN] == "BRCA1", DEFAULT_ACCESSION_COL].values[0] == "P38398"
    assert result.loc[result[GENE_COLUMN] == "AATF", DEFAULT_ACCESSION_COL].values[0] == "Q9NY61"
    assert result.loc[result[GENE_COLUMN] == "BYSL", DEFAULT_ACCESSION_COL].values[0] == "Q13895"
    # Verify UniProt was called for all genes (2 calls)
    assert mock_uniprot_client.get_accession_from_gene_symbol.call_count == 4


# exception handling
def test_get_or_append_stable_accession_missing_gene_column_raises():
    """Test that missing gene column raises appropriate error"""
    cluster_df = pd.DataFrame({"wrong_column": ["TP53"], "cluster": [1]})

    # Should raise KeyError or similar when trying to access gene_symbol column
    with pytest.raises(Exception):
        get_or_append_stable_accession(
            screen_name="test",
            cluster_df=cluster_df,
            gene_column="gene_symbol",
            organism_id=9606,
            warn_on_fallback=False,
        )


# =============================================================================
# Test: add_functional_annotations_to_chunk
# =============================================================================


def test_add_functional_annotations_to_chunk_success(mock_uniprot_client):
    """Test adding functional annotations to gene chunk"""
    chunk = pd.DataFrame(
        {"gene_symbol": ["TP53", "BRCA1"], "cluster": [1, 1], "accession": ["P04637", "P38398"]}
    )

    with patch(
        "mozzarellm.pipeline.bundle_builder.UniProtClient", return_value=mock_uniprot_client
    ):
        result = add_functional_annotations_to_chunk(
            chunk=chunk,
            cluster_id_column="cluster",
            screen_name="test",
            stable_accession_col="accession",
        )

    assert "UniProt_functional_annotation" in result.columns
    assert result.loc[0, "UniProt_functional_annotation"] == "Tumor suppressor"
    assert result.loc[1, "UniProt_functional_annotation"] == "DNA repair"


def test_add_functional_annotations_creates_intermediates_dir(mock_uniprot_client):
    """Test that intermediates directory is created"""
    chunk = pd.DataFrame({"gene_symbol": ["TP53"], "cluster": [1], "accession": ["P04637"]})

    with patch(
        "mozzarellm.pipeline.bundle_builder.UniProtClient", return_value=mock_uniprot_client
    ):
        with patch("mozzarellm.pipeline.bundle_builder.Path") as mock_path:
            mock_intermediates = MagicMock()
            mock_path.return_value.__truediv__.return_value = mock_intermediates

            add_functional_annotations_to_chunk(
                chunk=chunk,
                cluster_id_column="cluster",
                screen_name="test",
                stable_accession_col="accession",
            )

            # Should create intermediates directory
            mock_intermediates.mkdir.assert_called()


def test_add_functional_annotations_handles_api_failure(mock_uniprot_client):
    """Test graceful handling when UniProt API fails - should return unmodified chunk with warning"""
    chunk = pd.DataFrame({"gene_symbol": ["TP53"], "cluster": [1], "accession": ["P04637"]})

    # Override the fixture's fetch_functional_annotations to raise an exception
    mock_uniprot_client.fetch_functional_annotations.side_effect = Exception("API error")

    with patch(
        "mozzarellm.pipeline.bundle_builder.UniProtClient", return_value=mock_uniprot_client
    ):
        # Should catch exception and return unmodified chunk
        with pytest.warns(UserWarning, match="UniProt lookup failed for cluster '1'"):
            result = add_functional_annotations_to_chunk(
                chunk=chunk,
                cluster_id_column="cluster",
                screen_name="test",
                stable_accession_col="accession",
            )

        # Should return the original chunk unmodified (no annotation column added)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert "UniProt_functional_annotation" not in result.columns
        # Original columns should still be present
        assert "gene_symbol" in result.columns
        assert "cluster" in result.columns
        assert "accession" in result.columns


# =============================================================================
# Test: build_evidence_bundles
# =============================================================================


def test_build_evidence_bundles_creates_json_files(tmp_path, mock_uniprot_client):
    """Test that evidence bundles are created as JSON files"""
    acc_cluster_df = pd.DataFrame(
        {
            "gene_symbol": ["TP53", "BRCA1"],
            "cluster": [1, 1],
            STABLE_ACCESSION_COLUMN: ["P04637", "P38398"],
            "up_features": ["feature1,feature2", "feature3"],
            "down_features": ["feature4", "feature5,feature6"],
        }
    )

    # Use real filesystem with tmp_path
    output_base = tmp_path / "output"

    with patch(
        "mozzarellm.pipeline.bundle_builder.UniProtClient", return_value=mock_uniprot_client
    ):
        with patch("mozzarellm.pipeline.bundle_builder.Path") as mock_path:
            # Make Path("output") return our tmp_path
            def path_constructor(path_str):
                if path_str == "output":
                    return output_base
                return Path(path_str)

            mock_path.side_effect = path_constructor

            result = build_evidence_bundles(
                screen_name="test",
                acc_cluster_df=acc_cluster_df,
                gene_column="gene_symbol",
                cluster_id_column="cluster",
                stable_accession_col=STABLE_ACCESSION_COLUMN,
                feature_columns=["up_features", "down_features"],
            )

    # Verify directories were created
    analysis_dir = output_base / "test_analysis"
    bundles_dir = analysis_dir / "test_evidence_bundles"
    assert bundles_dir.exists(), "Evidence bundles directory should be created"

    # Verify JSON bundle file was created
    bundle_files = list(bundles_dir.glob("test__cluster_1__bundle.json"))
    assert len(bundle_files) == 1, "Should create one bundle file for cluster 1"

    # Verify bundle content
    import json

    bundle_path = bundle_files[0]
    with open(bundle_path) as f:
        bundle = json.load(f)

    assert bundle["screen_name"] == "test"
    assert bundle["cluster_id"] == "1"
    assert len(bundle["cluster_genes"]) == 2
    assert "feature_overlaps" in bundle

    # Verify gene data in bundle
    gene_symbols = [gene["gene_symbol"] for gene in bundle["cluster_genes"]]
    assert "TP53" in gene_symbols
    assert "BRCA1" in gene_symbols


def test_build_evidence_bundles_groups_by_cluster(tmp_path, mock_uniprot_client):
    """Test that genes are grouped by cluster - should create separate bundle files"""
    acc_cluster_df = pd.DataFrame(
        {
            "gene_symbol": ["TP53", "BRCA1", "CDK1"],
            "cluster": [1, 1, 2],
            "accession": ["P04637", "P38398", "P06493"],
            "up_features": ["f1", "f2", "f3"],
            "down_features": ["f4", "f5", "f6"],
        }
    )

    output_base = tmp_path / "output"

    with patch(
        "mozzarellm.pipeline.bundle_builder.UniProtClient", return_value=mock_uniprot_client
    ):
        with patch("mozzarellm.pipeline.bundle_builder.Path") as mock_path:

            def path_constructor(path_str):
                if path_str == "output":
                    return output_base
                return Path(path_str)

            mock_path.side_effect = path_constructor

            build_evidence_bundles(
                screen_name="test",
                acc_cluster_df=acc_cluster_df,
                gene_column="gene_symbol",
                cluster_id_column="cluster",
                stable_accession_col="accession",
                feature_columns=["up_features", "down_features"],
            )

    # Verify two separate bundle files were created (one per cluster)
    bundles_dir = output_base / "test_analysis" / "test_evidence_bundles"
    assert bundles_dir.exists()

    bundle_files = list(bundles_dir.glob("test__cluster_*__bundle.json"))
    assert len(bundle_files) == 2, "Should create two bundle files for two clusters"

    # Verify cluster 1 has 2 genes (TP53, BRCA1)
    import json

    cluster_1_file = bundles_dir / "test__cluster_1__bundle.json"
    with open(cluster_1_file) as f:
        cluster_1_bundle = json.load(f)
    assert len(cluster_1_bundle["cluster_genes"]) == 2

    # Verify cluster 2 has 1 gene (CDK1)
    cluster_2_file = bundles_dir / "test__cluster_2__bundle.json"
    with open(cluster_2_file) as f:
        cluster_2_bundle = json.load(f)
    assert len(cluster_2_bundle["cluster_genes"]) == 1
    assert cluster_2_bundle["cluster_genes"][0]["gene_symbol"] == "CDK1"


def test_build_evidence_bundles_empty_dataframe(tmp_path, mock_uniprot_client):
    """Test handling of empty cluster dataframe - should create directory but no bundles"""
    empty_df = pd.DataFrame(columns=["gene_symbol", "cluster", "accession"])

    output_base = tmp_path / "output"

    with patch(
        "mozzarellm.pipeline.bundle_builder.UniProtClient", return_value=mock_uniprot_client
    ):
        with patch("mozzarellm.pipeline.bundle_builder.Path") as mock_path:

            def path_constructor(path_str):
                if path_str == "output":
                    return output_base
                return Path(path_str)

            mock_path.side_effect = path_constructor

            # Should complete without error for empty dataframe
            build_evidence_bundles(
                screen_name="test",
                acc_cluster_df=empty_df,
                gene_column="gene_symbol",
                cluster_id_column="cluster",
                stable_accession_col="accession",
                feature_columns=[],
            )

    # Verify directory was created but no bundle files
    bundles_dir = output_base / "test_analysis" / "test_evidence_bundles"
    assert bundles_dir.exists(), "Directory should be created even for empty dataframe"

    bundle_files = list(bundles_dir.glob("*.json"))
    assert len(bundle_files) == 0, "No bundle files should be created for empty dataframe"
