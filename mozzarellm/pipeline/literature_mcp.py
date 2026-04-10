"""
Literature validation for MozzareLLM gene classifications.

Two modes:
- Mode B (Direct MCP): One LLM call with PubMed/bioRxiv MCP connectors. Unguided search.
- Mode A (Structured MCP): Two calls — Call 1 is standard CoT analysis, Call 2 is targeted
  MCP refinement on the flagged gene subset. See STRUCTURED_MCP_REFINEMENT_PROMPT.

Both modes take a per-cluster classification JSON and return an amended JSON
with a literature_validation field per gene.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from ..prompt_components import DIRECT_MCP_VALIDATION_PROMPT, LITERATURE_VALIDATION_OUTPUT_FORMAT

# MCP server URLs
# pubmed (pubmed.mcp.claude.com) — Anthropic-hosted, requires authorization_token=ANTHROPIC_API_KEY
# biorxiv (mcp.deepsense.ai) — NO keyword search, date/category only — not useful for gene queries
MCP_SERVERS = {
    "pubmed": "https://pubmed.mcp.claude.com/mcp",
    # biorxiv (mcp.deepsense.ai) excluded — server returns 500; no keyword search capability anyway
}


ANTHROPIC_STATUS_URL = "https://status.anthropic.com"


def check_mcp_servers(timeout: int = 5) -> dict[str, str]:
    """Probe each MCP server with a plain HTTP GET — no Anthropic API call.

    Uses httpx (already an anthropic SDK dependency) for cleaner error messages.
    Any HTTP response (including 4xx) means the server is reachable.
    A 5xx response or connection error means it's down.

    Args:
        timeout: Per-server connect+read timeout in seconds.

    Returns:
        Dict mapping server name → status string:
          "up (HTTP NNN)", "down (HTTP NNN)", or "down (<error>)".
    """
    import httpx

    statuses = {}
    for name, url in MCP_SERVERS.items():
        try:
            r = httpx.get(url, timeout=timeout, follow_redirects=True)
            code = r.status_code
        except httpx.TimeoutException:
            statuses[name] = "down (timeout)"
            continue
        except Exception as e:
            statuses[name] = f"down ({type(e).__name__}: {e})"
            continue

        if code < 500:
            statuses[name] = f"up (HTTP {code})"
        else:
            statuses[name] = f"down (HTTP {code})"

    return statuses


def check_anthropic_status(timeout: int = 5) -> str:
    """Check Anthropic's status page via HTTP GET — no API call.

    Returns a short status string: "operational (HTTP 200)", "degraded (HTTP NNN)",
    or "unreachable (<error>)".
    """
    import httpx

    try:
        r = httpx.get(ANTHROPIC_STATUS_URL, timeout=timeout, follow_redirects=True)
        if r.status_code == 200:
            return f"operational (HTTP {r.status_code})"
        return f"degraded (HTTP {r.status_code})"
    except Exception as e:
        return f"unreachable ({type(e).__name__}: {e})"


def get_available_mcp_servers(timeout: int = 5, verbose: bool = True) -> list[str]:
    """Probe MCP servers and Anthropic status — no API calls.

    Args:
        timeout: Per-server probe timeout in seconds.
        verbose: Print status of each server.

    Returns:
        List of server names that are up (subset of MCP_SERVERS keys).
    """
    anthropic_status = check_anthropic_status(timeout=timeout)
    mcp_statuses = check_mcp_servers(timeout=timeout)

    if verbose:
        icon = "✓" if "operational" in anthropic_status else "✗"
        print(f"  {icon} anthropic status page: {anthropic_status}")
        for name, status in mcp_statuses.items():
            icon = "✓" if status.startswith("up") else "✗"
            print(f"  {icon} {name} MCP: {status}")

    return [name for name, status in mcp_statuses.items() if status.startswith("up")]


def _extract_genes_to_validate(cluster_json: dict[str, Any]) -> list[str]:
    """Extract novel_role and uncharacterized gene symbols from a cluster result."""
    genes = []
    for g in cluster_json.get("novel_role_genes", []):
        gene = g.get("gene") or g
        if isinstance(gene, str):
            genes.append(gene)
    for g in cluster_json.get("uncharacterized_genes", []):
        gene = g.get("gene") or g
        if isinstance(gene, str):
            genes.append(gene)
    return genes


def _build_validation_prompt(
    pathway: str,
    cluster_json: dict[str, Any],
) -> str:
    """Build the Direct MCP validation prompt from the named template.

    Passes only novel_role_genes and uncharacterized_genes — established genes
    are excluded to avoid unnecessary token burn.
    """
    flagged_genes = {
        "novel_role_genes": cluster_json.get("novel_role_genes", []),
        "uncharacterized_genes": cluster_json.get("uncharacterized_genes", []),
    }
    return DIRECT_MCP_VALIDATION_PROMPT.format(
        flagged_genes_json=json.dumps(flagged_genes, indent=2),
        pathway=pathway,
        literature_validation_output_format=LITERATURE_VALIDATION_OUTPUT_FORMAT,
    )


def validate_and_amend_with_mcp(
    cluster_json: dict[str, Any],
    model: str = "claude-sonnet-4-5",
    max_tokens: int = 8000,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Mode A: Validate and amend classifications using LLM + MCP connectors.

    One LLM call that searches PubMed/bioRxiv via MCP tools and produces
    the amended classification JSON.

    Args:
        cluster_json: Per-cluster classification result (must have dominant_process,
                      novel_role_genes, uncharacterized_genes).
        model: Anthropic model to use.
        max_tokens: Max output tokens.
        max_retries: Retry count for MCP timeout errors.

    Returns:
        Amended cluster JSON with literature_validation per gene, plus metadata.
    """
    import anthropic

    genes = _extract_genes_to_validate(cluster_json)
    if not genes:
        return {**cluster_json, "_validation_metadata": {"mode": "mcp", "skipped": "no genes to validate"}}

    pathway = cluster_json.get("dominant_process", "")
    prompt = _build_validation_prompt(pathway, cluster_json)

    client = anthropic.Anthropic()
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    # authorization_token is only needed for Anthropic-hosted MCP servers (pubmed.mcp.claude.com)
    ANTHROPIC_HOSTED = {"pubmed.mcp.claude.com"}
    mcp_servers = [
        {"type": "url", "url": url, "name": name, **({"authorization_token": api_key} if any(h in url for h in ANTHROPIC_HOSTED) else {})}
        for name, url in MCP_SERVERS.items()
    ]
    tools = [{"type": "mcp_toolset", "mcp_server_name": name} for name in MCP_SERVERS]

    for attempt in range(max_retries):
        try:
            start = time.time()
            response = client.beta.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                mcp_servers=mcp_servers,
                tools=tools,
                betas=["mcp-client-2025-11-20"],
            )
            elapsed = time.time() - start
            break
        except (anthropic.BadRequestError, anthropic.InternalServerError) as e:
            msg = str(e).lower()
            retryable = any(kw in msg for kw in ("timed out", "504", "502", "503", "500", "connection error", "internal server"))
            if attempt < max_retries - 1 and retryable:
                time.sleep(15)
            else:
                raise

    # Extract text output
    output_text = ""
    tool_call_count = 0
    for block in response.content:
        if block.type == "text":
            output_text += block.text
        elif block.type == "mcp_tool_use":
            tool_call_count += 1

    # Parse JSON from response
    amended_json = _parse_json_from_text(output_text)

    # Estimate cost (Sonnet: $3/M input, $15/M output)
    cost = (response.usage.input_tokens * 3 + response.usage.output_tokens * 15) / 1_000_000

    if amended_json:
        amended_json["_validation_metadata"] = {
            "mode": "mcp",
            "model": model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cost_usd": round(cost, 4),
            "time_seconds": round(elapsed, 1),
            "tool_calls": tool_call_count,
            "genes_validated": genes,
        }
        return amended_json

    # Fallback: return original with metadata
    return {
        **cluster_json,
        "_validation_metadata": {
            "mode": "mcp",
            "error": "failed to parse amended JSON from LLM response",
            "raw_output": output_text[:500],
        },
    }


def validate_and_amend_without_mcp(
    cluster_json: dict[str, Any],
    literature_hits: list[dict[str, Any]],
    model: str = "claude-sonnet-4-5",
    max_tokens: int = 8000,
) -> dict[str, Any]:
    """Mode B: Amend classifications using pre-fetched literature search results.

    Takes raw Europe PMC search hits and uses an LLM call (no MCP) to
    interpret relevance and produce amended classifications.

    Args:
        cluster_json: Per-cluster classification result.
        literature_hits: Raw search results from LiteratureClient.search_gene_pathway_literature().
        model: Anthropic model to use.
        max_tokens: Max output tokens.

    Returns:
        Amended cluster JSON with literature_validation per gene.
    """
    import anthropic

    genes = _extract_genes_to_validate(cluster_json)
    if not genes:
        return {**cluster_json, "_validation_metadata": {"mode": "direct_api", "skipped": "no genes to validate"}}

    pathway = cluster_json.get("dominant_process", "")

    # Filter hits to only those mentioning our genes
    relevant_hits = [h for h in literature_hits if h.get("genes_mentioned")]

    prompt = f"""I need to validate gene-pathway associations based on literature search results and produce an amended classification.

## Original Classification
{json.dumps(cluster_json, indent=2)}

## Literature Search Results
The following papers were found by searching for the genes ({', '.join(genes)}) in relation to "{pathway}":

{json.dumps(relevant_hits, indent=2)}

## Task
Based on these search results, produce an amended version of the original classification JSON:
- Add a "literature_validation" field to each novel_role and uncharacterized gene entry
- If a paper directly demonstrates a gene's role in the pathway, suggest reclassification
- Keep the same JSON schema as the input

Return ONLY the amended JSON. The "literature_validation" field per gene should have:
- "literature_support": "none" | "weak" | "moderate" | "strong"
- "relevant_papers": [{{"title": "...", "year": "...", "key_finding": "..."}}]
- "suggested_reclassification": null or "established" or "novel_role"
- "rationale": "why reclassification is or isn't warranted"
"""

    client = anthropic.Anthropic()

    start = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.time() - start

    output_text = response.content[0].text if response.content else ""
    amended_json = _parse_json_from_text(output_text)

    cost = (response.usage.input_tokens * 3 + response.usage.output_tokens * 15) / 1_000_000

    if amended_json:
        amended_json["_validation_metadata"] = {
            "mode": "direct_api",
            "model": model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cost_usd": round(cost, 4),
            "time_seconds": round(elapsed, 1),
            "literature_hits_total": len(literature_hits),
            "literature_hits_relevant": len(relevant_hits),
            "genes_validated": genes,
        }
        return amended_json

    return {
        **cluster_json,
        "_validation_metadata": {
            "mode": "direct_api",
            "error": "failed to parse amended JSON from LLM response",
            "raw_output": output_text[:500],
        },
    }


def _parse_json_from_text(text: str) -> dict[str, Any] | None:
    """Extract JSON from LLM response text, handling markdown code blocks."""
    # Try extracting from ```json ... ``` block
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object boundaries
        brace_start = text.find("{")
        if brace_start == -1:
            return None
        depth = 0
        for i, c in enumerate(text[brace_start:], start=brace_start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start : i + 1])
                    except json.JSONDecodeError:
                        return None
    return None
