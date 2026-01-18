"""
Lightweight retrieval utilities for preliminary CoT-driven RAG.

This module deliberately avoids heavy vector-store deps. It harvests evidence from:
- Optional local knowledge files under data/knowledge/ (txt, md) by simple keyword match

Returned snippets include simple provenance fields so they can be logged downstream.
"""
#TODO: add gene aliases in addition to gene symbols after uniprot api proxy
#TODO: add pdf parsing functionality

from __future__ import annotations

import os
import re


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _iter_knowledge_files(base_dir: str) -> list[str]:
    paths: list[str] = []
    if not base_dir or not os.path.isdir(base_dir):
        return paths
    for root, _dirs, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith((".txt", ".md")):
                paths.append(os.path.join(root, f))
    return paths


def _search_file_for_terms(
    path: str, terms: list[str], max_hits: int = 5, context_window: int = 4
) -> list[tuple[str, int, int]]:
    """
    Search file for terms with improved scoring and context extraction.

    Returns: List of (snippet_text, relevance_score, line_number)
    """
    snippets: list[tuple[str, int, int]] = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
    except Exception:
        return snippets

    lines = text.splitlines()
    if not lines:
        return snippets

    # Score each line based on term matches
    line_scores: list[tuple[int, int, list[str]]] = []  # (line_idx, score, matched_terms)
    for idx, ln in enumerate(lines):
        score = 0
        matched = []
        for t in terms:
            matches = len(re.findall(rf"\b{re.escape(t)}\b", ln, flags=re.IGNORECASE))
            if matches > 0:
                # Boost score for multiple occurrences and exact case matches
                score += matches * 2 if re.search(rf"\b{re.escape(t)}\b", ln) else matches
                matched.append(t)

        # Bonus for having multiple terms in same line (proximity bonus)
        if len(matched) > 1:
            score += len(matched) * 2

        if score > 0:
            line_scores.append((idx, score, matched))

    if not line_scores:
        return snippets

    # Sort by score descending
    line_scores.sort(key=lambda x: x[1], reverse=True)

    # Extract top lines with context window
    seen_ranges = set()
    for line_idx, score, matched_terms in line_scores[:max_hits]:
        if line_idx in seen_ranges:
            continue

        # Extract context window
        start_idx = max(0, line_idx - context_window)
        end_idx = min(len(lines), line_idx + context_window + 1)

        # Mark range as seen to avoid duplicates
        for i in range(start_idx, end_idx):
            seen_ranges.add(i)

        # Build context snippet
        context_lines = lines[start_idx:end_idx]
        snippet_text = " ".join(_normalize(ln) for ln in context_lines if ln.strip())

        # Boost score based on number of unique terms matched
        final_score = score + len(set(matched_terms)) * 3

        snippets.append((snippet_text, final_score, line_idx))

        if len(snippets) >= max_hits:
            break

    return snippets


def local_knowledge_context_retriever(
    cluster_genes: list[str],
    knowledge_dir: str | None = None,
    top_k: int = 10,
    min_relevance_score: int = 2,
) -> dict:
    """
    Build a ranked set of evidence snippets for a cluster. Currently works best for gene-centric documents that contain gene names as keywords.

    Returns a dict with fields:
    - snippets: [{text, source, meta, relevance_score}] (sorted by relevance)
    - citations: [{source, path_or_id}]
    - retrieval_metadata: {knowledge_dir, k, genes_queried, total_retrieved}
    """
    terms = [
        g for g in cluster_genes if isinstance(g, str) and g
    ]  # TODO: add gene aliases in addition to gene symbols after uniprot api proxy
    # update: both screen context and functional annotations will be added directly to evidence bundle
    all_snippets: list[dict] = []
    citations: list[dict] = []

    # Retrieve from local knowledge (scored by relevance)
    knowledge_count = 0
    if knowledge_dir:
        files = _iter_knowledge_files(knowledge_dir)
        for p in files:
            hits = _search_file_for_terms(p, terms, max_hits=5, context_window=4)
            for htext, score, line_num in hits:
                if score >= min_relevance_score:
                    all_snippets.append(
                        {
                            "text": htext,
                            "source": "knowledge_file",
                            "meta": {"path": os.path.basename(p), "score": score, "line": line_num},
                            "relevance_score": min(score, 90),
                        }
                    )
                    knowledge_count += 1

    # Sort all snippets by relevance score (descending)
    all_snippets.sort(key=lambda x: x["relevance_score"], reverse=True)

    # Take top k snippets
    top_snippets = all_snippets[:top_k]

    # Build citations from selected snippets (deduplicated)
    seen_citations = set()
    for snip in top_snippets:
        cit_key = ("knowledge_file", snip["meta"]["path"])

        if cit_key not in seen_citations:
            citations.append({"source": cit_key[0], "path_or_id": cit_key[1]})
            seen_citations.add(cit_key)

    return {
        "snippets": top_snippets,
        "citations": citations,
        "retrieval_metadata": {
            "knowledge_dir": knowledge_dir,
            "k": top_k,
            "genes_queried": len(terms),
            "total_retrieved": len(all_snippets),
            "knowledge_snippets_found": knowledge_count,
        },
    }
