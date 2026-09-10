"""Unit tests for the gene/cluster tables save_cluster_analysis produces."""

import json

from mozzarellm.utils.llm_analysis_utils import save_cluster_analysis

_PARSED = {
    "dominant_process": "ribosome biogenesis",
    "pathway_confidence": "High",
    "established_genes": ["RPL3", "RPS6"],
    "novel_role_genes": [
        {"gene": "C1orf131", "class": "PARTIAL_EVIDENCE", "rationale": "r", "evidence": "e"}
    ],
    "uncharacterized_genes": [{"gene": "CXorf58", "class": "DARK_GENE", "rationale": "r2"}],
    "total_genes_in_cluster": 5,
    "missed_genes": ["XYZ1"],
    "classification_completeness": 0.8,
}


def test_gene_table_has_one_row_per_gene_with_subclass():
    out = save_cluster_analysis({"21": _PARSED}, save_outputs=False)
    df = out["gene_df"]
    assert len(df) == 4  # ESTABLISHED genes are rows too, not just flagged ones
    assert set(df["category"]) == {"ESTABLISHED", "NOVEL_ROLE", "UNCHARACTERIZED"}
    novel = df[df["gene"] == "C1orf131"].iloc[0]
    assert novel["subclass"] == "PARTIAL_EVIDENCE"
    assert novel["rationale"] == "r" and novel["evidence"] == "e"
    assert (df["dominant_process"] == "ribosome biogenesis").all()


def test_cluster_table_counts_and_coverage():
    out = save_cluster_analysis({"21": _PARSED}, save_outputs=False)
    row = out["cluster_df"].iloc[0]
    assert row["n_genes"] == 5 and row["n_classified"] == 4
    assert row["n_established"] == 2
    assert row["established_genes"] == "RPL3;RPS6"
    assert row["missed_genes"] == "XYZ1"
    assert row["classification_completeness"] == 0.8


def test_csvs_and_json_written(tmp_path):
    base = tmp_path / "run"
    save_cluster_analysis({"21": _PARSED}, out_file_base=str(base))
    assert (tmp_path / "run_genes.csv").exists()
    assert (tmp_path / "run_clusters.csv").exists()
    data = json.loads((tmp_path / "run_clusters.json").read_text())
    assert "21" in data["clusters"]


def test_original_df_columns_merge_in(tmp_path):
    import pandas as pd

    original = pd.DataFrame({"cluster_id": ["21"], "screen": ["funk_2022"]})
    out = save_cluster_analysis({"21": _PARSED}, save_outputs=False, original_df=original)
    assert (out["gene_df"]["screen"] == "funk_2022").all()
    assert (out["cluster_df"]["screen"] == "funk_2022").all()


def test_alphanumeric_cluster_ids_keep_every_row():
    # pandas argsort marks NaNs as -1; the old sort duplicated the last cluster
    # and dropped the rest whenever cluster ids were non-numeric.
    clusters = {cid: dict(_PARSED) for cid in ("C5255", "C5415", "C0001")}
    out = save_cluster_analysis(clusters, save_outputs=False)
    assert sorted(out["cluster_df"]["cluster_id"]) == ["C0001", "C5255", "C5415"]
    assert out["cluster_df"]["cluster_id"].is_unique
