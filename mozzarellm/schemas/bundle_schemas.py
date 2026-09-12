from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, model_validator


# Validation functions
def _validate_required_string(v: Any) -> str:
    if not isinstance(v, str):
        raise TypeError("Expected a string")
    vv = v.strip()
    if not vv:
        raise ValueError("Value cannot be empty")
    if vv.lower() in {"required", "template", "adjustable", "optional"}:
        raise ValueError("Value cannot be a template placeholder")
    return vv


def _validate_dict(v: Any) -> dict[str, Any]:
    if not isinstance(v, dict):
        raise TypeError("Expected an object (dict)")
    return v


# defining type aliases
RequiredStr = Annotated[str, BeforeValidator(_validate_required_string)]
RequiredDict = Annotated[dict[str, Any], BeforeValidator(_validate_dict)]


# Schema models
class SemiFlexModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    # sets defined fields as required, but user may add additional fields not defined in the schema as needed


class Perturbation(SemiFlexModel):
    type: RequiredStr
    library_or_reagent: RequiredStr


class Readout(SemiFlexModel):
    measurement: RequiredStr
    instrument_or_platform: RequiredStr
    primary_metric: RequiredStr


class PhenotypeReadout(SemiFlexModel):
    """Human-written descriptions of the optional per-gene phenotype evidence.

    Free prose the model reads verbatim: what the up/down feature lists are and
    what cutoff produced them, and what the perturbation-strength metric
    measures. Both optional -- screens without the corresponding data omit them.
    """

    features_description: str | None = None
    strength_description: str | None = None


class Clustering(SemiFlexModel):
    method: RequiredStr
    parameters: RequiredDict


class Controls(SemiFlexModel):
    negative_controls: RequiredStr
    positive_controls: RequiredStr


class Provenance(SemiFlexModel):
    dataset_name: RequiredStr
    citation: RequiredStr
    data_source: RequiredStr

    @model_validator(mode="before")
    @classmethod
    def _resolve_aliases(cls, values: dict) -> dict:
        """Accept 'lab' as alias for 'citation', 'screen_name' as alias for 'dataset_name'."""
        if not isinstance(values, dict):
            return values
        if not values.get("citation") and values.get("lab"):
            values["citation"] = values["lab"]
        if not values.get("dataset_name") and values.get("screen_name"):
            values["dataset_name"] = values["screen_name"]
        return values


class ScreenContext(SemiFlexModel):
    assay_type: RequiredStr
    target_phenotype: RequiredStr
    organism: RequiredStr
    cell_line_or_system: RequiredStr
    perturbation: Perturbation
    readout: Readout
    clustering: Clustering
    controls: Controls
    provenance: Provenance
    phenotype_readout: PhenotypeReadout | None = None


class BundleGene(SemiFlexModel):
    gene_symbol: str | None = None
    up_features: str | None = None
    down_features: str | None = None
    phenotypic_strength: str | None = None
    # canonical, per-gene annotations (e.g. Uniprot functional annotations)
    UniProt_functional_annotation: str | None = None
    affinage_functional_annotation: str | None = None
    affinage_audit_note: str | None = None


# Main schema model
class EvidenceBundle(SemiFlexModel):
    screen_name: str | None = None
    cluster_id: str
    # per-gene metadata keyed by gene symbol:
    cluster_genes: list[BundleGene]
