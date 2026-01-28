from pathlib import Path
from typing import Any
from pydantic import ValidationError
import json
import warnings
import pandas as pd
from mozzarellm.schemas.bundle_schemas import (
    BundleGene,
    BundleGeneAnnotations,
    EvidenceBundle,
    ScreenContext,
)
from mozzarellm.utils.io import load_table, write_bundle
from mozzarellm.utils.screen_context_utils import load_screen_context_json
from mozzarellm.utils.local_retrieval import local_knowledge_context_retriever
from mozzarellm.utils.cluster_utils import cluster_chunker, find_feature_overlaps
from mozzarellm.clients.uniprot_api_client import UniProtClient


# Default values
OUTPUT_DIR = Path("output")


def get_or_append_stable_accession(
    *,
    cluster_df: pd.DataFrame,
    accession_table: Path | None = None,
    accession_col: Path | None = None,
    accession_table_sheetname: str | None = None,
    accession_table_sep: str | None = None,
    output_dir: Path = OUTPUT_DIR,
    organism_id: int,
    warn_on_fallback: bool,
):
    """
    Assign stable accession numbers to gene symbols. Or append them from a provided table.
    """
    df = cluster_df.copy()  # avoid modifying original
    if accession_table is not None and accession_col is not None:
        # Append accession numbers from the provided table
        accession_df = load_table(accession_table, accession_table_sep, accession_table_sheetname)
        # slice just the accession column
        accession_col_df = accession_df[[accession_col]]
        # Merge acession col with cluster_df on gene symbol
        accession_merged_cluster_df = df.merge(
            accession_col_df, left_on="gene_symbol", right_on=accession_col, how="left"
        )
        # Rename the accession column to accession
        accession_merged_cluster_df = accession_merged_cluster_df.rename(
            columns={accession_col: "accession"}
        )

        # save as csv; output dir interface/output/
        output_dir.mkdir(
            parents=True, exist_ok=True
        )  # defense: make or assert that output dir exists
        accession_merged_cluster_df.to_csv(
            output_dir / "accession_merged_cluster_df.csv", index=False
        )

        return accession_merged_cluster_df
    else:
        if "gene_symbol" not in df.columns:
            raise ValueError("Expected column 'gene_symbol' to assign stable accessions")

        # init client
        uniprot_client = UniProtClient()

        def _lookup_accession(gene_symbol: str) -> str:
            gene_symbol = str(gene_symbol)
            if not gene_symbol or gene_symbol == "nan":
                return ""
            if gene_symbol.startswith("nontargeting_"):
                return "NON_TARGETING_CONTROL"
            try:
                accession = uniprot_client.get_accession_from_gene_symbol(
                    gene_symbol=gene_symbol,
                    organism_id=organism_id,
                    warn_on_fallback=warn_on_fallback,
                )
            except Exception as e:
                warnings.warn(f"UniProt lookup failed for gene_symbol '{gene_symbol}': {e}")
                return ""
            return accession or ""

        df["accession"] = df["gene_symbol"].map(_lookup_accession)
        output_dir.mkdir(
            parents=True, exist_ok=True
        )  # defense: make or assert that output dir exists
        df.to_csv(output_dir / "fetched_accession_cluster_df.csv", index=False)

        return df


def add_functional_annotations_to_chunk(
    chunk: pd.DataFrame,
    *,
    cluster_id_column: str,
    stable_accession_col: str | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """Add annotation columns to a chunk of gene-level data. Calls UniprotAPIClient to fetch annotations."""
    if stable_accession_col is None:
        stable_accession_col = "accession"
    chunk_annotated = chunk.copy()
    cluster_id = chunk_annotated[cluster_id_column].iloc[0]

    # initialize client
    uniprot_client = UniProtClient()
    # fetch annotations as 2 column dataframe
    annotations = uniprot_client.fetch_functional_annotations(chunk_annotated, stable_accession_col)
    # add annotations to chunk
    chunk_annotated = chunk_annotated.merge(annotations, on=stable_accession_col, how="left")
    # save as csv; output dir interface/output/
    output_dir.mkdir(parents=True, exist_ok=True)  # defense: make or assert that output dir exists
    # TODO: add override option for output to json
    chunk_annotated.to_csv(output_dir / f"cluster_{cluster_id}_chunk_annotated.csv", index=False)
    return chunk_annotated


def add_local_evidence_to_chunk(
    chunk: pd.DataFrame,
    knowledge_dir: str | Path | None = None,
    cluster_id_column: str | None = None,
    stable_accession_col: str | None = None,
) -> pd.DataFrame:
    """Add local evidence to a chunk of gene-level data. Calls local_knowledge_context_retriever to fetch evidence."""
    # TODO: implement improved local knowledge context retriever and add logic here
    pass


def build_evidence_bundles(
    *,
    screen_name: str | None = None,
    screen_context_path: str | Path | None = None,
    acc_cluster_df: pd.DataFrame | None = None,  # cluster table must have accession numbers
    gene_column: str | None = None,
    cluster_id_column: str | None = None,
    stable_accession_col: str | None = None,
    feature_columns: list[str] | None = None,
    override_screen_context: bool = False,
    use_retrieval: bool = False,  # false for now; true for future use
    knowledge_dir: str
    | Path
    | None = None,  # optionally change the directory where the knowledge files are stored
    top_k: int = 10,
    output_dir: Path = OUTPUT_DIR,
) -> list[Path]:
    # validate required columns in cluster table
    if cluster_id_column not in acc_cluster_df.columns:
        raise ValueError(f"Missing column '{cluster_id_column}' in cluster table")
    if gene_column not in acc_cluster_df.columns:
        raise ValueError(f"Missing column '{gene_column}' in cluster table")
    print(f"Using gene column: {gene_column}")
    # defense: make or assert that output dir exists
    output_dir = Path(output_dir / f"{screen_name}_evidence_bundles")
    output_dir.mkdir(parents=True, exist_ok=True)

    screen_context = load_screen_context_json(screen_context_path, override=override_screen_context) # includes validation
    if not screen_context:
        raise ValueError("Screen context not found")

    # chunk cluster df
    cluster_chunks = cluster_chunker(acc_cluster_df, cluster_id_column)
    print(f"Processing {len(cluster_chunks)} clusters")

    # annotate each cluster
    for chunk in cluster_chunks:
        cluster_id = chunk[cluster_id_column].iloc[0]
        annotated_chunk = add_functional_annotations_to_chunk(
            chunk,
            cluster_id_column=cluster_id_column,
            stable_accession_col=stable_accession_col,
            output_dir=output_dir,
        )
        print(f"Processing cluster {cluster_id}")

        # prune redundant cluster column
        annotated_chunk.drop(columns=[cluster_id_column], inplace=True)
        # set NaNs to empty strings
        annotated_chunk.fillna("", inplace=True)
        # convert to json
        cluster_as_json = annotated_chunk.to_dict(orient="records")

        # screen context + cluster info = evidence bundle
        evidence_bundle = {
            "screen_name": screen_name,
            "screen_context": screen_context,
            "cluster_id": str(cluster_id),
            "cluster_genes": cluster_as_json,
        }

        if feature_columns:
            evidence_bundle["feature_overlaps"] = find_feature_overlaps(chunk, feature_columns)

        # save bundle as json
        output_path = Path(output_dir / f"{screen_name}_{cluster_id}_bundle.json")
        write_bundle(evidence_bundle, output_path)  # includes validation
        print(f"Saved bundle to {output_path}")
