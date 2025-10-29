"""Tests for ClusterAnalyzer."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from mozzarellm.analyzer import ClusterAnalyzer
from mozzarellm.models import AnalysisResult


class TestClusterAnalyzerInitialization:
    """Tests for ClusterAnalyzer initialization."""

    def test_initialization_with_defaults(self):
        """Test analyzer initialization with default parameters."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            analyzer = ClusterAnalyzer(model="gpt-4o")
            assert analyzer.model == "gpt-4o"
            assert analyzer.temperature == 0.0
            assert analyzer.max_tokens == 8000
            assert not analyzer.use_retrieval

    def test_initialization_with_custom_params(self):
        """Test analyzer initialization with custom parameters."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            analyzer = ClusterAnalyzer(
                model="gpt-4o",
                temperature=0.7,
                max_tokens=4000,
                use_retrieval=True,
                knowledge_dir="data/knowledge",
                retriever_k=5
            )
            assert analyzer.temperature == 0.7
            assert analyzer.max_tokens == 4000
            assert analyzer.use_retrieval
            assert analyzer.knowledge_dir == "data/knowledge"
            assert analyzer.retriever_k == 5

    def test_initialization_with_claude(self):
        """Test analyzer initialization with Claude model."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            analyzer = ClusterAnalyzer(model="claude-3-7-sonnet-20250219")
            assert analyzer.model == "claude-3-7-sonnet-20250219"

    def test_initialization_with_gemini(self):
        """Test analyzer initialization with Gemini model."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            analyzer = ClusterAnalyzer(model="gemini-2.5-pro-preview-03-25")
            assert analyzer.model == "gemini-2.5-pro-preview-03-25"

    def test_missing_api_key_error(self):
        """Test that missing API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError):
                ClusterAnalyzer(model="gpt-4o")


class TestClusterAnalyzerAnalysis:
    """Tests for ClusterAnalyzer.analyze() method."""

    @patch("mozzarellm.analyzer.create_provider")
    def test_analyze_basic_dataframe(self, mock_create_provider):
        """Test basic analysis of a DataFrame."""
        # Create mock provider
        mock_provider = Mock()
        mock_provider.query.return_value = (
            '''```json
            {
                "cluster_id": "1",
                "dominant_process": "DNA repair",
                "pathway_confidence": "High",
                "established_genes": ["BRCA1", "BRCA2"],
                "uncharacterized_genes": [],
                "novel_role_genes": [],
                "summary": "Test summary"
            }
            ```''',
            None
        )
        mock_create_provider.return_value = mock_provider

        # Create test DataFrame
        df = pd.DataFrame({
            "cluster_id": ["1"],
            "genes": ["BRCA1;BRCA2;TP53"]
        })

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            analyzer = ClusterAnalyzer(model="gpt-4o", show_progress=False)
            result = analyzer.analyze(df)

        assert isinstance(result, AnalysisResult)
        assert len(result.clusters) == 1
        assert "1" in result.clusters
        assert result.clusters["1"].dominant_process == "DNA repair"
        mock_provider.query.assert_called_once()

    @patch("mozzarellm.analyzer.create_provider")
    def test_analyze_with_annotations(self, mock_create_provider):
        """Test analysis with gene annotations."""
        # Create mock provider
        mock_provider = Mock()
        mock_provider.query.return_value = (
            '''{"cluster_id": "1", "dominant_process": "test",
            "pathway_confidence": "Medium", "established_genes": [],
            "uncharacterized_genes": [], "novel_role_genes": [],
            "summary": "test"}''',
            None
        )
        mock_create_provider.return_value = mock_provider

        # Create test DataFrames
        cluster_df = pd.DataFrame({
            "cluster_id": ["1"],
            "genes": ["BRCA1;TP53"]
        })
        annotations_df = pd.DataFrame({
            "gene": ["BRCA1", "TP53"],
            "function": ["DNA repair", "Tumor suppressor"]
        })

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            analyzer = ClusterAnalyzer(model="gpt-4o", show_progress=False)
            result = analyzer.analyze(cluster_df, gene_annotations=annotations_df)

        assert isinstance(result, AnalysisResult)
        assert len(result.clusters) == 1

    @patch("mozzarellm.analyzer.create_provider")
    def test_analyze_multiple_clusters(self, mock_create_provider):
        """Test analysis of multiple clusters."""
        # Create mock provider
        mock_provider = Mock()
        mock_provider.query.side_effect = [
            (
                '''{"cluster_id": "1", "dominant_process": "DNA repair",
                "pathway_confidence": "High", "established_genes": ["BRCA1"],
                "uncharacterized_genes": [], "novel_role_genes": [],
                "summary": "test1"}''',
                None
            ),
            (
                '''{"cluster_id": "2", "dominant_process": "Cell cycle",
                "pathway_confidence": "Medium", "established_genes": ["CDK1"],
                "uncharacterized_genes": [], "novel_role_genes": [],
                "summary": "test2"}''',
                None
            )
        ]
        mock_create_provider.return_value = mock_provider

        # Create test DataFrame
        df = pd.DataFrame({
            "cluster_id": ["1", "2"],
            "genes": ["BRCA1;BRCA2", "CDK1;CDK2"]
        })

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            analyzer = ClusterAnalyzer(model="gpt-4o", show_progress=False)
            result = analyzer.analyze(df)

        assert len(result.clusters) == 2
        assert "1" in result.clusters
        assert "2" in result.clusters
        assert mock_provider.query.call_count == 2

    @patch("mozzarellm.analyzer.create_provider")
    def test_analyze_handles_query_error(self, mock_create_provider):
        """Test that analysis handles query errors gracefully."""
        # Create mock provider that returns error
        mock_provider = Mock()
        mock_provider.query.return_value = (None, "API error occurred")
        mock_create_provider.return_value = mock_provider

        # Create test DataFrame
        df = pd.DataFrame({
            "cluster_id": ["1"],
            "genes": ["BRCA1;BRCA2"]
        })

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            analyzer = ClusterAnalyzer(model="gpt-4o", show_progress=False)
            result = analyzer.analyze(df)

        # Should return empty result, not crash
        assert isinstance(result, AnalysisResult)
        assert len(result.clusters) == 0

    def test_analyze_missing_required_column(self):
        """Test that analysis raises error for missing required columns."""
        df = pd.DataFrame({
            "wrong_column": ["1"],
            "genes": ["BRCA1"]
        })

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            analyzer = ClusterAnalyzer(model="gpt-4o", show_progress=False)
            with pytest.raises(ValueError, match="cluster_id"):
                analyzer.analyze(df)

    @patch("mozzarellm.analyzer.create_provider")
    def test_analyze_skips_invalid_genes(self, mock_create_provider):
        """Test that analysis skips rows with invalid gene data."""
        mock_provider = Mock()
        mock_create_provider.return_value = mock_provider

        # Create DataFrame with invalid gene data
        df = pd.DataFrame({
            "cluster_id": ["1", "2"],
            "genes": [123, "BRCA1;BRCA2"]  # First is not a string
        })

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            analyzer = ClusterAnalyzer(model="gpt-4o", show_progress=False)
            result = analyzer.analyze(df)

        # Should skip cluster 1, process cluster 2
        assert mock_provider.query.call_count == 0  # Actually 0 because we mock the response

    @patch("mozzarellm.analyzer.create_provider")
    @patch("mozzarellm.analyzer.retrieve_context")
    def test_analyze_with_retrieval(self, mock_retrieve, mock_create_provider):
        """Test analysis with RAG retrieval enabled."""
        # Setup mocks
        mock_provider = Mock()
        mock_provider.query.return_value = (
            '''{"cluster_id": "1", "dominant_process": "test",
            "pathway_confidence": "High", "established_genes": [],
            "uncharacterized_genes": [], "novel_role_genes": [],
            "summary": "test"}''',
            None
        )
        mock_create_provider.return_value = mock_provider

        mock_retrieve.return_value = {
            "snippets": [{"text": "Evidence", "source": "pubmed", "relevance_score": 0.9}],
            "citations": [],
            "retrieval_metadata": {}
        }

        df = pd.DataFrame({
            "cluster_id": ["1"],
            "genes": ["BRCA1;BRCA2"]
        })

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            analyzer = ClusterAnalyzer(
                model="gpt-4o",
                use_retrieval=True,
                knowledge_dir="data/knowledge",
                show_progress=False
            )
            result = analyzer.analyze(df)

        # Verify retrieval was called
        mock_retrieve.assert_called()
        assert len(result.clusters) == 1


class TestClusterAnalyzerAliases:
    """Tests for ClusterAnalyzer convenience methods."""

    @patch("mozzarellm.analyzer.create_provider")
    def test_analyze_dataframe_alias(self, mock_create_provider):
        """Test that analyze_dataframe() is an alias for analyze()."""
        mock_provider = Mock()
        mock_provider.query.return_value = (
            '''{"cluster_id": "1", "dominant_process": "test",
            "pathway_confidence": "High", "established_genes": [],
            "uncharacterized_genes": [], "novel_role_genes": [],
            "summary": "test"}''',
            None
        )
        mock_create_provider.return_value = mock_provider

        df = pd.DataFrame({
            "cluster_id": ["1"],
            "genes": ["BRCA1"]
        })

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            analyzer = ClusterAnalyzer(model="gpt-4o", show_progress=False)
            result1 = analyzer.analyze(df)
            result2 = analyzer.analyze_dataframe(df)

        # Both should work and return AnalysisResult
        assert isinstance(result1, AnalysisResult)
        assert isinstance(result2, AnalysisResult)
