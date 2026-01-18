from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Annotated

from pydantic import BaseModel, Field, BeforeValidator, ConfigDict


SchemaVersion = Literal["0.1"]


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


class BundleGeneAnnotations(BaseModel):
    functional_text: str | None = None
    source: str | None = None
    retrieved_at: datetime | None = None


class BundleGene(BaseModel):
    up_features: list[dict[str, Any]] = Field(default_factory=list)
    down_features: list[dict[str, Any]] = Field(default_factory=list)
    annotations: BundleGeneAnnotations = Field(
        default_factory=BundleGeneAnnotations
    )  # canonical, per-gene annotations (e.g. Uniprot functional annotations)
    evidence: list[dict[str, Any]] = Field(default_factory=list)  # from data/knowledge retrieval


# Main schema model
class EvidenceBundle(BaseModel):
    schema_version: SchemaVersion = "0.1"
    screen_name: str | None = None
    cluster_id: str
    created_at: datetime
    screen_context: ScreenContext
    genes: dict[str, BundleGene] = Field(default_factory=dict)

    @property
    def gene_symbols(self) -> list[str]:
        return list(self.genes.keys())
