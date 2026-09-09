import warnings
from pathlib import Path

import pandas as pd

from mozzarellm.clients.affinage_api_client import ANNOTATION_COL as AFFINAGE_COL
from mozzarellm.clients.affinage_api_client import AUDIT_NOTE_COL as AFFINAGE_AUDIT_COL
from mozzarellm.clients.affinage_api_client import AffinageClient
from mozzarellm.clients.uniprot_api_client import UniProtClient
from mozzarellm.utils.cluster_utils import cluster_chunker, compute_feature_coherence
from mozzarellm.utils.io import load_table, write_bundle

DEFAULT_ACCESSION_COL = "accession"


def _lookup_accession(
    gene_symbol: str, organism_id: int, warn_on_fallback: bool, uniprot_client: UniProtClient
) -> str:
    gene_symbol = str(gene_symbol)
    if not gene_symbol or gene_symbol == "nan":
        return ""
    if gene_symbol.startswith("nontargeting_"):  # TODO: make this a config option
        return "NON_TARGETING_CONTROL"
    try:
        accession = uniprot_client.get_accession_from_gene_symbol(
            gene_symbol=gene_symbol,
            organism_id=organism_id,
            warn_on_fallback=warn_on_fallback,
        )
    except Exception as e:
        warnings.warn(f"UniProt lookup failed for gene_symbol '{gene_symbol}': {e}", stacklevel=2)
        return ""
    return accession or ""


def get_or_append_stable_accession(
    *,
    screen_name: str,
    cluster_df: pd.DataFrame,
    gene_column: str,
    organism_id: int,
    warn_on_fallback: bool,
    accession_table: Path | None = None,
    accession_col: Path | None = None,
    accession_table_gene_col: str | None = None,
    accession_table_sheetname: str | None = None,
    accession_table_sep: str | None = None,
    output_dir: Path
    | str
    | None = None,  # override default output dir (currently just used for unit tests)
):
    """
    Assign stable accession numbers to gene symbols. Or append them from a provided table.
    """
    # init client
    uniprot_client = UniProtClient()
    OUTPUT_DIR = (
        Path(output_dir if output_dir is not None else "output") / f"{screen_name}_analysis"
    )
    df = cluster_df.copy()  # avoid modifying original
    if (
        accession_table is not None and accession_col is not None
    ):  # Append accession numbers from the provided table
        accession_df = load_table(accession_table, accession_table_sep, accession_table_sheetname)
        # slice just the gene and accession columns
        accession_col_df = accession_df[[accession_table_gene_col, accession_col]]

        # Check for and handle duplicates in accession table
        duplicates = accession_col_df[accession_table_gene_col].duplicated(keep="first")
        if duplicates.any():
            dup_genes = accession_col_df[duplicates][accession_table_gene_col].unique()
            print(
                f"Warning: Found {len(dup_genes)} duplicate genes in accession table. Keeping first occurrence: {list(dup_genes)[:5]}..."
            )
            accession_col_df = accession_col_df.drop_duplicates(
                subset=[accession_table_gene_col], keep="first"
            )

        # Merge accession col with cluster_df on gene symbol
        accession_merged_cluster_df = df.merge(
            accession_col_df, left_on=gene_column, right_on=accession_table_gene_col, how="left"
        )
        # Drop the duplicate gene column from accession table
        accession_merged_cluster_df = accession_merged_cluster_df.drop(
            columns=[accession_table_gene_col]
        )

        # Check for missing accessions after merge (before renaming)
        missing_accessions = accession_merged_cluster_df[accession_col].isna()
        if missing_accessions.any():
            missing_genes = accession_merged_cluster_df[missing_accessions][gene_column].unique()
            print(
                f"Warning: {len(missing_genes)} genes not found in accession table: {list(missing_genes)[:5]}..."
            )
            print("Falling back to UniProt API for missing genes...")

            # Fill missing accessions using UniProt API
            mask = accession_merged_cluster_df[accession_col].isna()
            accession_merged_cluster_df.loc[mask, accession_col] = accession_merged_cluster_df.loc[
                mask, gene_column
            ].apply(lambda x: _lookup_accession(x, organism_id, warn_on_fallback, uniprot_client))

        # save as csv; output dir interface/output/
        OUTPUT_DIR.mkdir(
            parents=True, exist_ok=True
        )  # defense: make or assert that output dir exists
        (OUTPUT_DIR / "intermediates").mkdir(parents=True, exist_ok=True)
        accession_merged_cluster_df.to_csv(
            OUTPUT_DIR / "intermediates" / "appended_accession_cluster_df.csv", index=False
        )
        print(
            "Appended accession numbers to cluster df. Saved to: ",
            OUTPUT_DIR / "intermediates" / "appended_accession_cluster_df.csv",
        )
        return accession_merged_cluster_df
    else:
        if gene_column not in df.columns:
            raise ValueError(f"Expected column '{gene_column}' to assign stable accessions")

        df[DEFAULT_ACCESSION_COL] = df[gene_column].map(
            lambda x: _lookup_accession(x, organism_id, warn_on_fallback, uniprot_client)
        )
        OUTPUT_DIR.mkdir(
            parents=True, exist_ok=True
        )  # defense: make or assert that output dir exists
        (OUTPUT_DIR / "intermediates").mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_DIR / "intermediates" / "assigned_accession_cluster_df.csv", index=False)
        print(
            "Assigned accession numbers to cluster df. Saved to: ",
            OUTPUT_DIR / "intermediates" / "assigned_accession_cluster_df.csv",
        )
        return df


def add_functional_annotations_to_chunk(
    *,
    chunk: pd.DataFrame,
    screen_name: str,
    cluster_id_column: str,
    gene_column: str | None = None,
    stable_accession_col: str | None = None,
    source: str = "uniprot",
    uniprot_client: UniProtClient | None = None,
    affinage_client: AffinageClient | None = None,
    output_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Add annotation columns to a chunk of gene-level data.

    Each source is kept pure — no silent cross-source backfill:
      - "uniprot": fetches UniProt only.
      - "affinage": fetches Affinage only (alias-resolved, audit-gated).
      - "both": fetches both side-by-side as separate columns.
    Genes a source fails on stay empty for that source; the LLM sees the gap.
    """
    if stable_accession_col is None:
        stable_accession_col = DEFAULT_ACCESSION_COL
    chunk_annotated = chunk.copy()
    cluster_id = chunk_annotated[cluster_id_column].iloc[0]

    if source in ("uniprot", "both"):
        if uniprot_client is None:
            uniprot_client = UniProtClient()
        try:
            annotations = uniprot_client.fetch_functional_annotations(
                chunk_annotated, stable_accession_col
            )
            chunk_annotated = chunk_annotated.merge(
                annotations, on=stable_accession_col, how="left"
            )
        except Exception as e:
            warnings.warn(f"UniProt lookup failed for cluster '{cluster_id}': {e}", stacklevel=2)

    if source in ("affinage", "both"):
        if affinage_client is None:
            affinage_client = AffinageClient()
        try:
            affinage = affinage_client.fetch_functional_annotations(chunk_annotated, gene_column)
            chunk_annotated = chunk_annotated.merge(affinage, on=gene_column, how="left")
        except Exception as e:
            warnings.warn(f"Affinage lookup failed for cluster '{cluster_id}': {e}", stacklevel=2)
    # save as csv; output dir interface/output/
    OUTPUT_DIR = (
        Path(output_dir if output_dir is not None else "output") / f"{screen_name}_analysis"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # defense: make or assert that output dir exists
    (OUTPUT_DIR / "intermediates").mkdir(parents=True, exist_ok=True)
    # TODO: add override option for output to json
    chunk_annotated.to_csv(
        OUTPUT_DIR / "intermediates" / f"cluster_{cluster_id}_chunk_annotated.csv", index=False
    )
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
    acc_cluster_df: pd.DataFrame | None = None,  # cluster table must have accession numbers
    gene_column: str | None = None,
    cluster_id_column: str | None = None,
    stable_accession_col: str | None = None,
    feature_columns: list[str] | None = None,
    use_retrieval: bool = False,  # false for now; true for future use
    knowledge_dir: str
    | Path
    | None = None,  # optionally change the directory where the knowledge files are stored
    top_k: int = 10,
    source: str = "uniprot",
    uniprot_client: UniProtClient | None = None,  # Inject dependency
    affinage_client: AffinageClient | None = None,
    output_dir: Path | str | None = None,
    flat_output: bool = False,
):
    if uniprot_client is None:
        uniprot_client = UniProtClient()  # Create once
    if affinage_client is None and source in ("affinage", "both"):
        affinage_client = AffinageClient()
    # validate required columns in cluster table
    if cluster_id_column not in acc_cluster_df.columns:
        raise ValueError(f"Missing column '{cluster_id_column}' in cluster table")
    if gene_column not in acc_cluster_df.columns:
        raise ValueError(f"Missing column '{gene_column}' in cluster table")
    print(f"Using gene column: {gene_column}")

    # resolve bundle output directory
    if flat_output:
        OUTPUT_DIR = Path(output_dir if output_dir is not None else "output")
    else:
        OUTPUT_DIR = (
            Path(output_dir if output_dir is not None else "output")
            / f"{screen_name}_analysis"
            / f"{screen_name}_evidence_bundles"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # chunk cluster df
    cluster_chunks = cluster_chunker(acc_cluster_df, cluster_id_column)
    print(f"Processing {len(cluster_chunks)} clusters")

    # annotate each cluster
    for chunk in cluster_chunks:
        cluster_id = chunk[cluster_id_column].iloc[0]
        print(f"Processing cluster {cluster_id}")
        annotated_chunk = add_functional_annotations_to_chunk(
            chunk=chunk,
            screen_name=screen_name,
            cluster_id_column=cluster_id_column,
            gene_column=gene_column,
            stable_accession_col=stable_accession_col,
            source=source,
            uniprot_client=uniprot_client,
            affinage_client=affinage_client,
            output_dir=output_dir,
        )

        # prune redundant cluster column
        annotated_chunk.drop(columns=[cluster_id_column], inplace=True)
        # set NaNs to empty strings
        annotated_chunk.fillna("", inplace=True)
        # convert to json
        cluster_as_json = annotated_chunk.to_dict(orient="records")

        # for affinage/both, drop the empty backup-source key so each gene shows only its source
        if source in ("affinage", "both"):
            for gene in cluster_as_json:
                for col in ("UniProt_functional_annotation", AFFINAGE_COL, AFFINAGE_AUDIT_COL):
                    if gene.get(col) == "":
                        gene.pop(col, None)

        # screen context + cluster info = evidence bundle
        evidence_bundle = {
            "screen_name": screen_name,
            "cluster_id": str(cluster_id),
            "cluster_genes": cluster_as_json,
        }

        if feature_columns:
            evidence_bundle["feature_coherence"] = compute_feature_coherence(
                chunk, feature_columns, gene_column=gene_column
            )

        # save bundle as json
        output_path = Path(OUTPUT_DIR / f"{screen_name}__cluster_{cluster_id}__bundle.json")
        write_bundle(evidence_bundle, output_path)  # includes validation
        print(f"Saved bundle to {output_path}")
