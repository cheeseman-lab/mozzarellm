from __future__ import annotations

import pytest

from mozzarellm.utils.retrieval import local_knowledge_context_retriever


def test_local_knowledge_context_retriever_no_knowledge_dir_returns_empty():
    out = local_knowledge_context_retriever(cluster_genes=["TP53"], knowledge_dir=None, top_k=10)
    assert out["snippets"] == []
    assert out["citations"] == []
    assert out["retrieval_metadata"]["knowledge_dir"] is None
    assert out["retrieval_metadata"]["k"] == 10
    assert out["retrieval_metadata"]["genes_queried"] == 1
    assert out["retrieval_metadata"]["total_retrieved"] == 0


def test_local_knowledge_context_retriever_ignores_non_txt_md_files(tmp_path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()

    (knowledge_dir / "notes.pdf").write_text("TP53 appears here", encoding="utf-8")

    out = local_knowledge_context_retriever(cluster_genes=["TP53"], knowledge_dir=str(knowledge_dir), top_k=10)
    assert out["snippets"] == []
    assert out["citations"] == []
    assert out["retrieval_metadata"]["knowledge_snippets_found"] == 0


def test_local_knowledge_context_retriever_collects_snippets_and_citations(tmp_path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()

    # file_a has multiple TP53 hits plus a second term on the same line -> should rank higher
    (knowledge_dir / "file_a.txt").write_text(
        "Intro\nTP53 TP53 is mentioned with BRCA1 on the same line\nOutro\n",
        encoding="utf-8",
    )

    # file_b has a single hit
    (knowledge_dir / "file_b.md").write_text(
        "Some header\nTP53 is only mentioned once here\nFooter\n",
        encoding="utf-8",
    )

    out = local_knowledge_context_retriever(
        cluster_genes=["TP53", "BRCA1"],
        knowledge_dir=str(knowledge_dir),
        top_k=10,
        min_relevance_score=0,
    )

    snippets = out["snippets"]
    citations = out["citations"]

    assert len(snippets) >= 2

    for s in snippets:
        assert isinstance(s["text"], str)
        assert s["text"].strip()
        assert s["source"] == "knowledge_file"
        assert isinstance(s["meta"], dict)
        assert "path" in s["meta"]
        assert "line" in s["meta"]
        assert "score" in s["meta"]
        assert isinstance(s["relevance_score"], int)
        assert 0 <= s["relevance_score"] <= 90

    # Sorted by relevance_score descending
    assert snippets[0]["relevance_score"] >= snippets[1]["relevance_score"]

    # Citations should be deduplicated by file
    cited_paths = {c["path_or_id"] for c in citations}
    assert cited_paths.issubset({"file_a.txt", "file_b.md"})


def test_local_knowledge_context_retriever_respects_top_k(tmp_path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()

    (knowledge_dir / "a.txt").write_text("TP53\nTP53\nTP53\n", encoding="utf-8")
    (knowledge_dir / "b.txt").write_text("TP53\n", encoding="utf-8")

    out = local_knowledge_context_retriever(
        cluster_genes=["TP53"],
        knowledge_dir=str(knowledge_dir),
        top_k=1,
        min_relevance_score=0,
    )

    assert len(out["snippets"]) == 1
    assert len(out["citations"]) == 1


def test_local_knowledge_context_retriever_min_relevance_score_filters_all(tmp_path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()

    (knowledge_dir / "a.txt").write_text("TP53\n", encoding="utf-8")

    out = local_knowledge_context_retriever(
        cluster_genes=["TP53"],
        knowledge_dir=str(knowledge_dir),
        top_k=10,
        min_relevance_score=10_000,
    )

    assert out["snippets"] == []
    assert out["citations"] == []
    assert out["retrieval_metadata"]["total_retrieved"] == 0
