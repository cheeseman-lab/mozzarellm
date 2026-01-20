"""
Main cluster analysis orchestration for mozzarellm.

This module provides the ClusterAnalyzer class, which is the primary
interface for analyzing gene clusters using LLMs.
"""

import logging
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from .models import AnalysisResult, ClusterResult
from .providers import create_provider
from .utils.llm_analysis_utils import process_cluster_response
from .utils.prompt_factory import make_cluster_analysis_prompt
from .utils.retrieval import local_knowledge_context_retriever

logger = logging.getLogger(__name__)


class ClusterAnalyzer:
    """
    Main analyzer for gene cluster analysis using LLMs.

    This class provides a clean, unified interface for analyzing gene clusters
    to identify biological pathways and prioritize novel genes.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 8000,
        top_p: float | None = None,
        top_k: int | None = None,
        stop_sequences: list[str] | None = None,
        system_prompt: str | None = None,
        use_retrieval: bool = False,
        knowledge_dir: str | None = None,
        retriever_k: int = 10,
        cot_instructions: str | None = None,
        api_key: str | None = None,
        show_progress: bool = True,
    ):
        """
        Initialize the ClusterAnalyzer.

        Args:
            model: Model identifier (e.g., "gpt-4o", "claude-sonnet-4-5-20250929")
            temperature: Temperature for generation (0.0-1.0)
            max_tokens: Maximum tokens to generate per request
            top_p: Nucleus sampling parameter (0.0-1.0, optional)
            top_k: Top-K sampling parameter (optional, Claude/Gemini only)
            stop_sequences: List of stop sequences (optional)
            system_prompt: Optional custom system prompt (uses default if None)
            use_retrieval: Whether to use RAG with retrieved evidence
            knowledge_dir: Directory containing knowledge files for RAG
            retriever_k: Number of evidence snippets to retrieve
            cot_instructions: Optional chain-of-thought reasoning instructions
            api_key: Optional API key (reads from environment if None)
            show_progress: Whether to display progress bars
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.show_progress = show_progress

        # RAG settings
        self.use_retrieval = use_retrieval
        self.knowledge_dir = knowledge_dir
        self.retriever_k = retriever_k
        self.cot_instructions = cot_instructions

        # Set system prompt
        self.system_prompt = system_prompt or self._get_default_system_prompt()

        # Create LLM provider
        self.provider = create_provider(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
            api_key=api_key,
        )

        logger.info(f"ClusterAnalyzer initialized with model: {model}")

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for genomics analysis."""
        return (
            "You are an AI assistant specializing in genomics and systems biology "
            "with expertise in pathway analysis. Your task is to analyze gene clusters "
            "to identify biological pathways and potential novel pathway members based "
            "on published literature and gaps in knowledge of gene function."
        )

    def analyze(
        self,
        cluster_df: pd.DataFrame,
        gene_annotations: pd.DataFrame | None = None,
        screen_context: str | None = None,
        cluster_analysis_prompt: str | None = None,
        gene_column: str = "genes",
        gene_sep: str = ";",
        cluster_id_column: str = "cluster_id",
    ) -> AnalysisResult:
        """
        Analyze gene clusters from a DataFrame.

        Args:
            cluster_df: DataFrame with cluster data (must have cluster_id and genes columns)
            gene_annotations: Optional DataFrame mapping genes to annotations
                             (column 0: gene ID, column 1: annotation text)
            screen_context: Optional context about the experimental screen
            cluster_analysis_prompt: Optional custom analysis prompt template
            gene_column: Column name containing genes (default: "genes")
            gene_sep: Separator for genes in gene_column (default: ";")
            cluster_id_column: Column name for cluster IDs (default: "cluster_id")

        Returns:
            AnalysisResult object with all cluster analysis results
        """
        # Validate inputs
        if cluster_id_column not in cluster_df.columns:
            raise ValueError(f"Column '{cluster_id_column}' not found in cluster_df")
        if gene_column not in cluster_df.columns:
            raise ValueError(f"Column '{gene_column}' not found in cluster_df")

        # Convert gene annotations DataFrame to dict if provided
        annotations_dict = None
        if gene_annotations is not None:
            gene_col = gene_annotations.columns[0]
            annot_col = gene_annotations.columns[1]
            annotations_dict = dict(
                zip(gene_annotations[gene_col], gene_annotations[annot_col], strict=True)
            )
            logger.info(f"Loaded {len(annotations_dict)} gene annotations")

        # Process each cluster
        results = {}
        iterator = (
            tqdm(cluster_df.iterrows(), total=len(cluster_df), desc="Analyzing clusters")
            if self.show_progress
            else cluster_df.iterrows()
        )

        for _idx, row in iterator:
            cluster_id = str(row[cluster_id_column])
            genes_str = row[gene_column]

            if not isinstance(genes_str, str):
                logger.warning(f"Skipping cluster {cluster_id}: genes not a string")
                continue

            genes = [g.strip() for g in genes_str.split(gene_sep) if g.strip()]

            if not genes:
                logger.warning(f"Skipping cluster {cluster_id}: no valid genes")
                continue

            if len(genes) > 1000:  # Max genes limit
                logger.warning(f"Skipping cluster {cluster_id}: too many genes ({len(genes)})")
                continue

            # Analyze this cluster
            cluster_result = self._analyze_single_cluster(
                cluster_id=cluster_id,
                genes=genes,
                gene_annotations=annotations_dict,
                screen_context=screen_context,
                cluster_analysis_prompt=cluster_analysis_prompt,
            )

            if cluster_result:
                results[cluster_id] = cluster_result

        # Create AnalysisResult with metadata
        analysis_result = AnalysisResult(
            clusters=results,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "temperature": self.temperature,
                "total_clusters": len(results),
                "use_retrieval": self.use_retrieval,
            },
        )

        logger.info(f"Analysis complete: {len(results)} clusters processed")
        return analysis_result

    def _validate_cluster_result(self, result: ClusterResult) -> list[str]:
        """
        Validate cluster result for quality issues.

        Args:
            result: ClusterResult to validate

        Returns:
            List of warning messages
        """
        warnings = []

        # Check for missed genes
        if result.missed_genes:
            warnings.append(
                f"Classification incomplete: {len(result.missed_genes)}/{result.total_genes_in_cluster} "
                f"genes not classified: {', '.join(result.missed_genes[:5])}"
                + ("..." if len(result.missed_genes) > 5 else "")
            )

        # Check classification completeness
        if result.classification_completeness < 0.9:
            warnings.append(
                f"Only {result.classification_completeness:.1%} of genes were classified"
            )

        # Check established ratio vs confidence - High confidence should have good support
        if result.pathway_confidence == "High":
            if result.established_gene_ratio < 0.05:
                warnings.append(
                    f"CRITICAL: High confidence but only {result.established_gene_ratio:.1%} "
                    f"({len(result.established_genes)}/{result.total_genes_in_cluster}) "
                    f"established genes - pathway assignment likely incorrect"
                )
            elif result.established_gene_ratio < 0.10:
                warnings.append(
                    f"WARNING: High confidence but only {result.established_gene_ratio:.1%} "
                    f"({len(result.established_genes)}/{result.total_genes_in_cluster}) "
                    f"established genes - verify pathway assignment"
                )

        # Check Medium confidence should have at least some support
        if result.pathway_confidence == "Medium" and result.established_gene_ratio < 0.03:
            warnings.append(
                f"Medium confidence with very few established genes "
                f"({result.established_gene_ratio:.1%}) - may need review"
            )

        return warnings

    def _analyze_single_cluster(
        self,
        cluster_id: str,
        genes: list[str],
        gene_annotations: dict[str, str] | None,
        screen_context: str | None,
        cluster_analysis_prompt: str | None,
    ) -> ClusterResult | None:
        """
        Analyze a single cluster.

        Args:
            cluster_id: Cluster identifier
            genes: List of gene symbols
            gene_annotations: Optional dict of gene annotations
            screen_context: Optional screen context
            cluster_analysis_prompt: Optional custom prompt template

        Returns:
            ClusterResult if successful, None if failed
        """
        # Retrieve evidence if RAG is enabled
        retrieved_ctx = None
        if self.use_retrieval:
            retrieved_ctx = local_knowledge_context_retriever(
                cluster_genes=genes,
                knowledge_dir=self.knowledge_dir,
                top_k=self.retriever_k,
            )

        # Build prompt
        prompt = make_cluster_analysis_prompt(
            cluster_id=cluster_id,
            genes=genes,
            gene_annotations_dict=gene_annotations,
            screen_context=screen_context,
            template_string=cluster_analysis_prompt,
            retrieved_context=retrieved_ctx,
            cot_instructions=self.cot_instructions,
        )

        # Query LLM
        response, error = self.provider.query(
            system_prompt=self.system_prompt,
            user_prompt=prompt,
        )

        if error:
            logger.error(f"Cluster {cluster_id} analysis failed: {error}")
            return None

        # Parse response
        try:
            parsed = process_cluster_response(response)

            # Ensure cluster_id matches
            if parsed.get("cluster_id") != cluster_id:
                logger.warning(
                    f"Cluster ID mismatch: expected {cluster_id}, "
                    f"got {parsed.get('cluster_id')}. Correcting."
                )
                parsed["cluster_id"] = cluster_id

            # Calculate quality metrics
            input_genes_set = set(genes)
            classified_genes_set = set(parsed["established_genes"])
            classified_genes_set.update(g["gene"] for g in parsed["uncharacterized_genes"])
            classified_genes_set.update(g["gene"] for g in parsed["novel_role_genes"])

            missed_genes = list(input_genes_set - classified_genes_set)
            total_genes = len(genes)
            classification_completeness = (
                len(classified_genes_set) / total_genes if total_genes > 0 else 1.0
            )
            established_ratio = (
                len(parsed["established_genes"]) / total_genes if total_genes > 0 else 0.0
            )

            # Convert to ClusterResult
            cluster_result = ClusterResult(
                cluster_id=parsed["cluster_id"],
                dominant_process=parsed["dominant_process"],
                pathway_confidence=parsed["pathway_confidence"],
                established_genes=parsed["established_genes"],
                uncharacterized_genes=parsed["uncharacterized_genes"],
                novel_role_genes=parsed["novel_role_genes"],
                summary=parsed["summary"],
                raw_response=response,
                missed_genes=missed_genes,
                total_genes_in_cluster=total_genes,
                classification_completeness=classification_completeness,
                established_gene_ratio=established_ratio,
            )

            # Validate quality and log warnings
            warnings = self._validate_cluster_result(cluster_result)
            for warning in warnings:
                logger.warning(f"Cluster {cluster_id}: {warning}")

            logger.info(f"Successfully analyzed cluster {cluster_id}")
            return cluster_result

        except Exception as e:
            logger.error(f"Failed to parse response for cluster {cluster_id}: {e}")
            return None

    def analyze_dataframe(
        self, df: pd.DataFrame, annotations_df: pd.DataFrame | None = None, **kwargs
    ) -> AnalysisResult:
        """
        Convenience method - alias for analyze().

        Args:
            df: DataFrame with cluster data
            annotations_df: Optional DataFrame with gene annotations
            **kwargs: Additional arguments passed to analyze()

        Returns:
            AnalysisResult object
        """
        return self.analyze(cluster_df=df, gene_annotations=annotations_df, **kwargs)
