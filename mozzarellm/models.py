"""
Pydantic models for domain objects in mozzarellm.

These models provide type safety, validation, and clear contracts
for data flowing through the analysis pipeline.
"""

from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class GeneClassification(BaseModel):
    """Classification of a single gene with priority and rationale."""

    gene: str = Field(..., description="Gene symbol")
    priority: int = Field(..., ge=1, le=10, description="Priority score (1-10)")
    rationale: str = Field(..., description="Explanation for classification and priority")

    @field_validator('gene')
    @classmethod
    def gene_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Gene symbol cannot be empty")
        return v.strip()


class ClusterResult(BaseModel):
    """Analysis result for a single gene cluster."""

    cluster_id: str = Field(..., description="Unique cluster identifier")
    dominant_process: str = Field(..., description="Identified biological pathway or process")
    pathway_confidence: Literal["High", "Medium", "Low"] = Field(
        ...,
        description="Confidence in pathway identification"
    )
    established_genes: List[str] = Field(
        default_factory=list,
        description="Well-known members of the identified pathway"
    )
    uncharacterized_genes: List[GeneClassification] = Field(
        default_factory=list,
        description="Genes with minimal functional annotation in any literature"
    )
    novel_role_genes: List[GeneClassification] = Field(
        default_factory=list,
        description="Genes with known functions in other pathways"
    )
    summary: str = Field(..., description="Concise summary of key findings")
    raw_response: Optional[str] = Field(None, description="Raw LLM response text")

    # Quality metrics
    missed_genes: List[str] = Field(
        default_factory=list,
        description="Genes in cluster that were not classified by LLM"
    )
    total_genes_in_cluster: int = Field(
        default=0,
        description="Total number of genes in the input cluster"
    )
    classification_completeness: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Ratio of classified genes to total genes (0.0-1.0)"
    )
    established_gene_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Ratio of established genes to total genes (0.0-1.0)"
    )

    @field_validator('cluster_id')
    @classmethod
    def cluster_id_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Cluster ID cannot be empty")
        return str(v).strip()

    def get_all_flagged_genes(self) -> List[GeneClassification]:
        """Return all flagged genes (uncharacterized + novel_role)."""
        return self.uncharacterized_genes + self.novel_role_genes

    def get_high_priority_genes(self, threshold: int = 8) -> List[GeneClassification]:
        """Return genes with priority >= threshold."""
        return [
            gene for gene in self.get_all_flagged_genes()
            if gene.priority >= threshold
        ]


class AnalysisResult(BaseModel):
    """Complete analysis results for all clusters."""

    clusters: Dict[str, ClusterResult] = Field(
        default_factory=dict,
        description="Dictionary mapping cluster IDs to results"
    )
    metadata: Dict[str, any] = Field(
        default_factory=dict,
        description="Analysis metadata (timestamp, model, etc.)"
    )

    def get_cluster(self, cluster_id: str) -> Optional[ClusterResult]:
        """Get result for a specific cluster."""
        return self.clusters.get(str(cluster_id))

    def get_all_high_confidence_clusters(self) -> List[ClusterResult]:
        """Return clusters with High pathway confidence."""
        return [
            cluster for cluster in self.clusters.values()
            if cluster.pathway_confidence == "High"
        ]

    def get_total_flagged_genes(self) -> int:
        """Count total flagged genes across all clusters."""
        return sum(
            len(cluster.get_all_flagged_genes())
            for cluster in self.clusters.values()
        )


class ClusterInput(BaseModel):
    """Input data for a single cluster to analyze."""

    cluster_id: str = Field(..., description="Unique cluster identifier")
    genes: List[str] = Field(..., min_length=1, description="List of gene symbols in cluster")
    gene_annotations: Optional[Dict[str, str]] = Field(
        None,
        description="Optional annotations for genes (gene -> annotation text)"
    )

    @field_validator('cluster_id')
    @classmethod
    def cluster_id_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Cluster ID cannot be empty")
        return str(v).strip()

    @field_validator('genes')
    @classmethod
    def validate_genes(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Genes list cannot be empty")
        # Remove empty strings and strip whitespace
        cleaned = [g.strip() for g in v if g and g.strip()]
        if not cleaned:
            raise ValueError("No valid genes after cleaning")
        return cleaned


class RetrievalContext(BaseModel):
    """Retrieved evidence context for a cluster analysis."""

    snippets: List[Dict] = Field(
        default_factory=list,
        description="Evidence snippets with text, source, and relevance score"
    )
    citations: List[Dict] = Field(
        default_factory=list,
        description="Citation information for sources"
    )
    retrieval_metadata: Dict = Field(
        default_factory=dict,
        description="Metadata about the retrieval process"
    )
