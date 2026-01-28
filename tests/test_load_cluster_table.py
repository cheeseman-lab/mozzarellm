from __future__ import annotations

import pandas as pd
import pytest

from mozzarellm.utils.io import load_table


@pytest.mark.parametrize("delimiter,extension", [(",", ".csv"), ("\t", ".tsv"), ("\t", ".txt")])
def test_load_cluster_table_format_default_sep(tmp_path, delimiter, extension):
    test_file = tmp_path / f"clusters{extension}"
    test_file.write_text(
        f"gene_symbol{delimiter}cluster{delimiter}up_features{delimiter}down_features{delimiter}phenotypic_strength\n"
        f'AATF{delimiter}21{delimiter}"interphase_cell_correlation_dapi_tubulin,interphase_cell_correlation_tubulin_phalloidin,interphase_nucleus_dapi_mean"{delimiter}"interphase_nucleus_area,interphase_cell_area,interphase_nucleus_solidity"{delimiter}669/5299\n'
        f'ABT1{delimiter}21{delimiter}"interphase_nucleus_gh2ax_mean,interphase_nucleus_dapi_mean,interphase_cell_correlation_dapi_tubulin,interphase_cell_correlation_tubulin_phalloidin,interphase_cell_phalloidin_mean"{delimiter}"interphase_nucleus_area,interphase_cell_area,interphase_nucleus_solidity"{delimiter}560/5299\n'
        f'BMS1{delimiter}21{delimiter}"interphase_nucleus_dapi_mean,interphase_cell_correlation_dapi_tubulin,interphase_cell_correlation_tubulin_phalloidin,interphase_nucleus_gh2ax_mean,interphase_cell_phalloidin_mean"{delimiter}"interphase_nucleus_area,interphase_cell_area,interphase_nucleus_solidity"{delimiter}446/5299\n',
        encoding="utf-8",
    )

    expected = pd.DataFrame(
        {
            "gene_symbol": ["AATF", "ABT1", "BMS1"],
            "cluster": [21, 21, 21],
            "up_features": [
                "interphase_cell_correlation_dapi_tubulin,interphase_cell_correlation_tubulin_phalloidin,interphase_nucleus_dapi_mean",
                "interphase_nucleus_gh2ax_mean,interphase_nucleus_dapi_mean,interphase_cell_correlation_dapi_tubulin,interphase_cell_correlation_tubulin_phalloidin,interphase_cell_phalloidin_mean",
                "interphase_nucleus_dapi_mean,interphase_cell_correlation_dapi_tubulin,interphase_cell_correlation_tubulin_phalloidin,interphase_nucleus_gh2ax_mean,interphase_cell_phalloidin_mean",
            ],
            "down_features": [
                "interphase_nucleus_area,interphase_cell_area,interphase_nucleus_solidity",
                "interphase_nucleus_area,interphase_cell_area,interphase_nucleus_solidity",
                "interphase_nucleus_area,interphase_cell_area,interphase_nucleus_solidity",
            ],
            "phenotypic_strength": ["669/5299", "560/5299", "446/5299"],
        }
    )

    df = load_table(test_file)
    pd.testing.assert_frame_equal(df, expected, check_dtype=True)
    assert list(df.columns) == [
        "gene_symbol",
        "cluster",
        "up_features",
        "down_features",
        "phenotypic_strength",
    ]
    assert df.iloc[0]["gene_symbol"] == "AATF"


def test_load_cluster_table_override_sep(tmp_path):
    test_tsv = tmp_path / "clusters.tsv"
    test_tsv.write_text("cluster_id,genes\n1,TP53;MDM2\n", encoding="utf-8")
    df = load_table(test_tsv, sep=",")
    assert list(df.columns) == ["cluster_id", "genes"]


def test_load_cluster_table_xlsx_uses_read_excel(tmp_path):
    test_xlsx = tmp_path / "clusters.xlsx"

    expected = pd.DataFrame({"cluster_id": [1], "genes": ["TP53"]})
    expected.to_excel(test_xlsx, index=False, sheet_name="Sheet1")

    df = load_table(test_xlsx, sheet_name="Sheet1")
    pd.testing.assert_frame_equal(df, expected, check_dtype=True)


def test_load_cluster_table_unsupported_suffix_raises(tmp_path):
    test_json = tmp_path / "clusters.json"
    test_json.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError) as e:
        load_table(test_json)
    assert "Unsupported input format" in str(e.value)


def test_load_cluster_table_csv_wrong_delimiter_without_override(tmp_path):
    test_csv = tmp_path / "clusters.csv"
    # Tab-delimited content stored in a .csv file; load_cluster_table will assume comma.
    test_csv.write_text(
        "gene_symbol\tcluster\nAATF\t21\n",
        encoding="utf-8",
    )
    df = load_table(test_csv)
    assert list(df.columns) != ["gene_symbol", "cluster"]
