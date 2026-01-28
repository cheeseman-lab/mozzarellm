from pathlib import Path
from typing import Any, Dict, Mapping
from datetime import datetime
import json
import warnings
import pandas as pd
from mozzarellm.utils.bundle_schemas import (
    BundleGene,
    BundleGeneAnnotations,
    EvidenceBundle,
    ScreenContext,
)
from mozzarellm.utils.io import write_bundle
from mozzarellm.utils.retrieval import local_knowledge_context_retriever
from mozzarellm.clients.uniprot_api_client import UniProtClient


# Default values
OUTPUT_DIR = Path("output")
JSON_BYTE_CAP = 5_000  # 5 KB -- conservative cap, needs to be adjusted


### Screen Context Utilities ###
def context_json_validator(data) -> bool:
    """Validate that the context JSON is valid and doesn't contain TODO fields."""
    if "TODO" in data.keys():
        raise ValueError(
            "Screen context JSON contains TODO field. Please double check the file and remove it."
        )
    if (
        len(json.dumps(data).encode("utf-8")) > JSON_BYTE_CAP
    ):  # intended as a model agnostic cap; chars is an alt option
        raise ValueError("Screen context JSON is too large. Please reduce the size of the file.")
    return True


def load_screen_context_json(
    path: str | Path | None,
    *,
    override: bool = False,  # optional kwarg
) -> Dict[str, Any] | None:
    """Load structured screen context from JSON."""
    try:
        if path is None:
            if override:  # testing convenience, to see how much context changes response
                return {}
            raise ValueError("Screen context path is required")

        json_path = Path(path)
        if not json_path.exists():
            raise FileNotFoundError(f"Screen context file not found: {json_path}")

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        context_json_validator(data)  # size and completion check
        model = ScreenContext.model_validate(data)  # schema validation
        return model.model_dump()
    except Exception as e:
        raise Exception(f"Error loading screen context JSON: {e}") from e


def load_table(
    input_path: str | Path,
    sep: str | None = None,
    sheet_name: str | int | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Load a cluster/gene table from CSV/TSV/TXT/XLSX into a DataFrame. Extra pandas kwargs are forwarded to pandas read_csv/read_excel."""
    path = Path(input_path)
    suffix = path.suffix.lower()

    if suffix == ".xlsx":
        return pd.read_excel(path, sheet_name=sheet_name, **kwargs)

    if suffix in {".csv", ".tsv", ".txt"}:
        default_sep = "," if suffix == ".csv" else "\t"
        return pd.read_csv(path, sep=(sep or default_sep), **kwargs)

    raise ValueError(
        f"Unsupported input format for cluster table: {suffix}. "
        "Expected one of .csv, .tsv, .txt, .xlsx."
    )


### Cluster Utilities ###
def find_feature_overlaps(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, str]:
    """Find overlapping comma-separated features in specified columns.

    Returns a mapping from column name to a comma-separated string of features that
    appear in 2+ rows.
    """
    overlaps: dict[str, str] = {}
    for col in feature_columns:
        if col not in df.columns:
            overlaps[col] = ""
            continue

        feature_counts: dict[str, int] = {}
        for feature_str in df[col].dropna():
            if not isinstance(feature_str, str) or not feature_str:
                continue
            features = [f.strip() for f in feature_str.split(",") if f.strip()]
            for feature in features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1

        overlapping = [f for f, count in feature_counts.items() if count >= 2]
        overlapping.sort(key=lambda f: (-feature_counts[f], f))
        overlaps[col] = ",".join(overlapping) if overlapping else ""

    return overlaps


def find_average_phenotypic_strength(df: pd.DataFrame, phenotypic_strength_column: str) -> str:
    """Find average phenotypic strength in a DataFrame."""
    if phenotypic_strength_column not in df.columns:
        raise ValueError(f"Column {phenotypic_strength_column} not found in DataFrame")

    numerators: list[float] = []
    denominators: set[str] = set()
    for strength in df[phenotypic_strength_column].dropna().tolist():
        if not isinstance(strength, str) or "/" not in strength:
            continue
        num_str, denom_str = strength.split("/", 1)
        try:
            numerators.append(float(num_str))
        except ValueError:
            continue
        denom_str = denom_str.strip()
        if denom_str:
            denominators.add(denom_str)

    if not numerators or len(denominators) != 1:
        return ""

    denom = next(iter(denominators))
    avg_numerator = sum(numerators) / len(numerators)
    return f"{avg_numerator:.1f}/{denom}"


# **NOTE above method under review: is this average biologically relevant? or misleading?
# FOR: (potentially) indicator of overall cluster salience; may be useful for ranking/prioritizing clusters
# AGAINST: may mask important nuances


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


def cluster_chunker(df: pd.DataFrame, cluster_id_column: str) -> list[pd.DataFrame]:
    """Chunk a gene-level table into smaller per-cluster DataFrames slices.

    Returns:
        List of DataFrames, one for each cluster.

    Raises:
        ValueError: If cluster_id_column is not found in DataFrame.

    Note:
        Will handle both sorted and interleaved cluster IDs. Relative order of genes within each cluster is preserved from the original DataFrame.
    """
    if cluster_id_column not in df.columns:
        raise ValueError(f"Cluster ID column '{cluster_id_column}' not found in DataFrame.")
    # Single-pass split; sort=False preserves first-seen cluster order; row order within each chunk is preserved.
    return [group for _cluster_id, group in df.groupby(cluster_id_column, sort=False)]


def _generate_cluster_search_query(chunk: pd.DataFrame, stable_accession_col: str) -> str:
    """Generate a search query for a chunk of gene-level data."""
    if stable_accession_col in chunk.columns:
        chunk_genes = chunk[stable_accession_col].tolist()
    else:
        chunk_genes = chunk["accession"].tolist()
    return "(" + " OR ".join(chunk_genes) + ") AND reviewed:true"
    # TODO: handle edge case where chunk is >100 genes (search query limit)


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
    # generate search query
    search_query = _generate_cluster_search_query(chunk_annotated, stable_accession_col)
    # fetch annotations as 2 column dataframe
    annotations = uniprot_client.fetch_functional_annotations(search_query)
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
    # TODO: implement improvel local knowledge context retriever and add logic here
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
    # validate required columns
    if cluster_id_column not in acc_cluster_df.columns:
        raise ValueError(f"Missing column '{cluster_id_column}' in cluster table")
    if gene_column not in acc_cluster_df.columns:
        raise ValueError(f"Missing column '{gene_column}' in cluster table")
    print(f"Using gene column: {gene_column}")
    # defense: make or assert that output dir exists
    output_dir = Path(output_dir / f"{screen_name}_evidence_bundles")
    output_dir.mkdir(parents=True, exist_ok=True)

    screen_context = load_screen_context_json(screen_context_path, override=override_screen_context)
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
            "screen_context": screen_context,
            "cluster_id": str(cluster_id),
            "cluster_genes": cluster_as_json,
        }

        if feature_columns:
            evidence_bundle["feature_overlaps"] = find_feature_overlaps(chunk, feature_columns)

        # save bundle as json
        output_path = Path(output_dir / f"{screen_name}_{cluster_id}_bundle.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(evidence_bundle, f, indent=2)
        print(f"Saved bundle to {output_path}")
