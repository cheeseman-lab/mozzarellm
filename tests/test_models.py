"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError
from mozzarellm.models import (
    GeneClassification,
    ClusterResult,
    AnalysisResult,
    ClusterInput,
    RetrievalContext,
)


class TestGeneClassification:
    """Tests for GeneClassification model."""

    def test_valid_gene_classification(self):
        """Test creating a valid gene classification."""
        gene = GeneClassification(
            gene="BRCA1",
            priority=8,
            rationale="Strong evidence for novel pathway role"
        )
        assert gene.gene == "BRCA1"
        assert gene.priority == 8
        assert "novel pathway" in gene.rationale

    def test_priority_validation(self):
        """Test that priority must be between 1 and 10."""
        # Valid priorities
        GeneClassification(gene="TP53", priority=1, rationale="test")
        GeneClassification(gene="TP53", priority=10, rationale="test")

        # Invalid priorities
        with pytest.raises(ValidationError):
            GeneClassification(gene="TP53", priority=0, rationale="test")
        with pytest.raises(ValidationError):
            GeneClassification(gene="TP53", priority=11, rationale="test")

    def test_empty_gene_validation(self):
        """Test that gene cannot be empty."""
        with pytest.raises(ValidationError):
            GeneClassification(gene="", priority=5, rationale="test")
        with pytest.raises(ValidationError):
            GeneClassification(gene="   ", priority=5, rationale="test")


class TestClusterResult:
    """Tests for ClusterResult model."""

    def test_valid_cluster_result(self):
        """Test creating a valid cluster result."""
        result = ClusterResult(
            cluster_id="1",
            dominant_process="DNA repair",
            pathway_confidence="High",
            established_genes=["BRCA1", "BRCA2"],
            uncharacterized_genes=[
                GeneClassification(gene="GENE1", priority=8, rationale="Novel gene")
            ],
            novel_role_genes=[
                GeneClassification(gene="GENE2", priority=7, rationale="Novel role")
            ],
            summary="Strong DNA repair signature"
        )
        assert result.cluster_id == "1"
        assert result.pathway_confidence == "High"
        assert len(result.established_genes) == 2
        assert len(result.get_all_flagged_genes()) == 2

    def test_quality_metrics(self):
        """Test quality metrics fields."""
        result = ClusterResult(
            cluster_id="1",
            dominant_process="DNA repair",
            pathway_confidence="High",
            summary="Test",
            missed_genes=["GENE3", "GENE4"],
            total_genes_in_cluster=10,
            classification_completeness=0.8,
            established_gene_ratio=0.2,
        )
        assert result.missed_genes == ["GENE3", "GENE4"]
        assert result.total_genes_in_cluster == 10
        assert result.classification_completeness == 0.8
        assert result.established_gene_ratio == 0.2

    def test_pathway_confidence_validation(self):
        """Test that pathway_confidence must be High/Medium/Low."""
        # Valid
        ClusterResult(
            cluster_id="1",
            dominant_process="test",
            pathway_confidence="High",
            summary="test"
        )

        # Invalid
        with pytest.raises(ValidationError):
            ClusterResult(
                cluster_id="1",
                dominant_process="test",
                pathway_confidence="Very High",
                summary="test"
            )

    def test_get_high_priority_genes(self):
        """Test filtering genes by priority threshold."""
        result = ClusterResult(
            cluster_id="1",
            dominant_process="test",
            pathway_confidence="Medium",
            uncharacterized_genes=[
                GeneClassification(gene="A", priority=9, rationale="test"),
                GeneClassification(gene="B", priority=7, rationale="test"),
                GeneClassification(gene="C", priority=8, rationale="test"),
            ],
            summary="test"
        )
        high_priority = result.get_high_priority_genes(threshold=8)
        assert len(high_priority) == 2
        assert all(g.priority >= 8 for g in high_priority)


class TestAnalysisResult:
    """Tests for AnalysisResult model."""

    def test_valid_analysis_result(self):
        """Test creating a valid analysis result."""
        cluster1 = ClusterResult(
            cluster_id="1",
            dominant_process="DNA repair",
            pathway_confidence="High",
            summary="test"
        )
        cluster2 = ClusterResult(
            cluster_id="2",
            dominant_process="Cell cycle",
            pathway_confidence="Medium",
            summary="test"
        )

        result = AnalysisResult(
            clusters={"1": cluster1, "2": cluster2},
            metadata={"model": "gpt-4o", "timestamp": "2024-01-01"}
        )

        assert len(result.clusters) == 2
        assert result.metadata["model"] == "gpt-4o"

    def test_get_cluster(self):
        """Test retrieving a specific cluster."""
        cluster = ClusterResult(
            cluster_id="1",
            dominant_process="test",
            pathway_confidence="High",
            summary="test"
        )
        result = AnalysisResult(clusters={"1": cluster})

        retrieved = result.get_cluster("1")
        assert retrieved is not None
        assert retrieved.cluster_id == "1"

        not_found = result.get_cluster("999")
        assert not_found is None

    def test_get_high_confidence_clusters(self):
        """Test filtering clusters by confidence."""
        cluster1 = ClusterResult(
            cluster_id="1",
            dominant_process="test",
            pathway_confidence="High",
            summary="test"
        )
        cluster2 = ClusterResult(
            cluster_id="2",
            dominant_process="test",
            pathway_confidence="Medium",
            summary="test"
        )
        cluster3 = ClusterResult(
            cluster_id="3",
            dominant_process="test",
            pathway_confidence="High",
            summary="test"
        )

        result = AnalysisResult(clusters={"1": cluster1, "2": cluster2, "3": cluster3})
        high_confidence = result.get_all_high_confidence_clusters()

        assert len(high_confidence) == 2
        assert all(c.pathway_confidence == "High" for c in high_confidence)


class TestClusterInput:
    """Tests for ClusterInput model."""

    def test_valid_cluster_input(self):
        """Test creating valid cluster input."""
        cluster = ClusterInput(
            cluster_id="1",
            genes=["BRCA1", "TP53", "PTEN"],
            gene_annotations={"BRCA1": "DNA repair gene"}
        )
        assert cluster.cluster_id == "1"
        assert len(cluster.genes) == 3

    def test_empty_genes_validation(self):
        """Test that genes list cannot be empty."""
        with pytest.raises(ValidationError):
            ClusterInput(cluster_id="1", genes=[])

    def test_genes_cleaning(self):
        """Test that empty genes are filtered out."""
        cluster = ClusterInput(
            cluster_id="1",
            genes=["BRCA1", "", "  ", "TP53"]
        )
        # Should have cleaned out empty strings
        assert len(cluster.genes) == 2
        assert "BRCA1" in cluster.genes
        assert "TP53" in cluster.genes


class TestRetrievalContext:
    """Tests for RetrievalContext model."""

    def test_valid_retrieval_context(self):
        """Test creating valid retrieval context."""
        context = RetrievalContext(
            snippets=[
                {"text": "Evidence 1", "source": "pubmed", "relevance_score": 0.9}
            ],
            citations=[{"source": "pubmed", "id": "12345"}],
            retrieval_metadata={"k": 10, "total": 100}
        )
        assert len(context.snippets) == 1
        assert context.retrieval_metadata["k"] == 10

    def test_empty_retrieval_context(self):
        """Test creating empty retrieval context."""
        context = RetrievalContext()
        assert len(context.snippets) == 0
        assert len(context.citations) == 0
        assert len(context.retrieval_metadata) == 0
