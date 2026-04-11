"""
Pydantic schemas for literature validation output.

These extend the existing ClusterResult output format with
literature validation metadata per gene.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RelevantPaper(BaseModel):
    title: str
    year: str
    doi: str | None = None
    key_finding: str


class GeneLiteratureValidation(BaseModel):
    gene: str
    literature_support: Literal["none", "weak", "moderate", "strong"]
    relevant_papers: list[RelevantPaper] = Field(default_factory=list)
    suggested_reclassification: Literal["established", "novel_role"] | None = None
    rationale: str


class ClusterValidationResult(BaseModel):
    """Per-cluster validation result with literature evidence."""

    cluster_id: str
    pathway: str
    validations: list[GeneLiteratureValidation]
    mode: Literal["mcp", "direct_api"]
    cost_usd: float | None = None
    tokens_used: int | None = None
