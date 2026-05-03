"""Pydantic schemas for MCP literature-validation output.

These mirror the structures requested by `LITERATURE_VALIDATION_OUTPUT_FORMAT`
and `STEP_LITERATURE_VALIDATION` in `mozzarellm.prompt_components`. They
validate the literature-specific portions of the cluster JSON returned by
`analyze_and_validate_unified()`; gene categorization fields are validated by
the existing parser in `llm_analysis_utils`.

Validation is applied softly — failures are logged as warnings on
`_validation_metadata.schema_warnings` rather than raising.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

GeneCategory = Literal["ESTABLISHED", "NOVEL_ROLE", "UNCHARACTERIZED"]
LiteratureSupport = Literal["none", "weak", "moderate", "strong"]


class RelevantPaper(BaseModel):
    model_config = ConfigDict(extra="ignore")
    pmid: str
    title: str
    year: str
    key_finding: str


class LiteratureValidation(BaseModel):
    """Per-gene `literature_validation` block (novel_role + uncharacterized genes)."""

    model_config = ConfigDict(extra="ignore")
    literature_support: LiteratureSupport
    relevant_papers: list[RelevantPaper] = Field(default_factory=list)
    pathway_connection: str | None = None
    suggested_reclassification: GeneCategory | None = None
    suggested_subclass: str | None = None
    rationale: str


class LiteratureReclassification(BaseModel):
    """Entry in top-level `literature_informed_reclassifications` array."""

    model_config = ConfigDict(extra="ignore")
    gene: str
    initial_category: GeneCategory
    final_category: GeneCategory
    driving_pmids: list[str] = Field(default_factory=list)
    rationale: str


class LiteraturePathwayRevision(BaseModel):
    """Top-level `literature_informed_pathway_revision` object."""

    model_config = ConfigDict(extra="ignore")
    pre_literature_pathway: str
    post_literature_pathway: str
    pathway_changed: bool
    rationale: str


class ValidationMetadata(BaseModel):
    """`_validation_metadata` block attached to the parsed cluster result."""

    model_config = ConfigDict(extra="allow")
    mode: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    time_seconds: float
    tool_calls: int
    error: str | None = None
    schema_warnings: list[str] | None = None
