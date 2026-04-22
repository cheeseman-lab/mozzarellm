"""
Literature validation for MozzareLLM gene classifications.

Single LLM call constrained to exactly 2 MCP tool calls (search + metadata fetch).
Takes a per-cluster classification JSON and returns an amended JSON with a
literature_validation field per gene.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import anthropic

from ..prompt_components import LITERATURE_VALIDATION_OUTPUT_FORMAT, MCP_VALIDATION_PROMPT

# MCP server URLs
# pubmed.mcp.claude.com — Anthropic-hosted, requires authorization_token=ANTHROPIC_API_KEY.
MCP_SERVERS = {
    "pubmed": "https://pubmed.mcp.claude.com/mcp",
}

# Anthropic-hosted MCP servers require an authorization_token (the API key);
# third-party servers do not.
ANTHROPIC_HOSTED_MCP = {"pubmed.mcp.claude.com"}

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
    novel_role_subset: list[dict] | None = None,
    uncharacterized_subset: list[dict] | None = None,
) -> str:
    """Build the Direct MCP validation prompt from the named template.

    Passes only novel_role_genes and uncharacterized_genes — established genes
    are excluded to avoid unnecessary token burn. If subsets are provided, uses
    those instead of the full lists (for batching).
    """
    flagged_genes = {
        "novel_role_genes": novel_role_subset if novel_role_subset is not None else cluster_json.get("novel_role_genes", []),
        "uncharacterized_genes": uncharacterized_subset if uncharacterized_subset is not None else cluster_json.get("uncharacterized_genes", []),
    }
    return MCP_VALIDATION_PROMPT.format(
        flagged_genes_json=json.dumps(flagged_genes, indent=2),
        pathway=pathway,
        literature_validation_output_format=LITERATURE_VALIDATION_OUTPUT_FORMAT,
    )


def validate_and_amend_with_mcp(
    cluster_json: dict[str, Any],
    model: str = "claude-sonnet-4-5",
    max_tokens: int = 16000,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Validate and amend classifications using LLM + MCP connectors.

    Single API call constrained to exactly 2 MCP tool calls:
    - search_articles with one OR query covering all genes
    - get_article_metadata for the resulting PMIDs

    The model then validates relevance using the full cluster annotation.

    Args:
        cluster_json: Per-cluster classification result (must have dominant_process,
                      novel_role_genes, uncharacterized_genes).
        model: Anthropic model to use.
        max_tokens: Max output tokens.
        max_retries: Retry count for MCP timeout errors.

    Returns:
        Amended cluster JSON with literature_validation per gene, plus metadata.
    """
    genes = _extract_genes_to_validate(cluster_json)
    if not genes:
        return {**cluster_json, "_validation_metadata": {"mode": "mcp", "skipped": "no genes to validate"}}

    pathway = cluster_json.get("dominant_process", "")
    prompt = _build_validation_prompt(pathway, cluster_json)

    client = anthropic.Anthropic()
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    mcp_servers = [
        {"type": "url", "url": url, "name": name, **({"authorization_token": api_key} if any(h in url for h in ANTHROPIC_HOSTED_MCP) else {})}
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
                time.sleep(60)
            else:
                raise

    output_text = ""
    tool_call_count = 0
    for block in response.content:
        if block.type == "text":
            output_text += block.text
        elif block.type == "mcp_tool_use":
            tool_call_count += 1

    amended_json = _parse_json_from_text(output_text)
    cost = (response.usage.input_tokens * 3 + response.usage.output_tokens * 15) / 1_000_000

    meta = {
        "mode": "mcp",
        "model": model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost_usd": round(cost, 4),
        "time_seconds": round(elapsed, 1),
        "tool_calls": tool_call_count,
        "genes_validated": genes,
    }

    if not amended_json:
        return {
            **cluster_json,
            "_validation_metadata": {**meta, "error": "failed to parse amended JSON", "raw_output": output_text[:500]},
        }

    amended = {
        **cluster_json,
        "novel_role_genes": amended_json.get("novel_role_genes", cluster_json.get("novel_role_genes", [])),
        "uncharacterized_genes": amended_json.get("uncharacterized_genes", cluster_json.get("uncharacterized_genes", [])),
        "_validation_metadata": meta,
    }

    # Count reclassifications
    reclassifications = []
    for g in amended["novel_role_genes"] + amended["uncharacterized_genes"]:
        lit = g.get("literature_validation", {})
        if lit.get("suggested_reclassification"):
            reclassifications.append({"gene": g["gene"], "new_category": lit["suggested_reclassification"]})
    if reclassifications:
        amended["reclassifications"] = reclassifications

    return amended


def analyze_and_validate_unified(
    system_prompt: str,
    user_prompt: str,
    model: str = "claude-sonnet-4-5",
    max_tokens: int = 16000,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Single LLM call that performs CoT analysis AND constrained MCP literature validation.

    The system_prompt should be assembled with COT_STEPS_UNIFIED_MCP (which inserts
    the literature validation step between VERIFICATION and OUTPUT).

    Args:
        system_prompt: Full CoT system prompt including the literature validation step.
        user_prompt: Cluster bundle content.
        model: Anthropic model to use.
        max_tokens: Max output tokens (must accommodate CoT reasoning + final JSON).
        max_retries: Retry count for MCP timeout errors.

    Returns:
        Dict with parsed cluster result + _validation_metadata.
    """
    client = anthropic.Anthropic()
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    mcp_servers = [
        {"type": "url", "url": url, "name": name, **({"authorization_token": api_key} if any(h in url for h in ANTHROPIC_HOSTED_MCP) else {})}
        for name, url in MCP_SERVERS.items()
    ]
    tools = [{"type": "mcp_toolset", "mcp_server_name": name} for name in MCP_SERVERS]

    for attempt in range(max_retries):
        try:
            start = time.time()
            response = client.beta.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
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
                time.sleep(60)
            else:
                raise

    output_text = ""
    tool_call_count = 0
    for block in response.content:
        if block.type == "text":
            output_text += block.text
        elif block.type == "mcp_tool_use":
            tool_call_count += 1

    parsed = _parse_json_from_text(output_text)
    cost = (response.usage.input_tokens * 3 + response.usage.output_tokens * 15) / 1_000_000

    meta = {
        "mode": "unified_mcp",
        "model": model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost_usd": round(cost, 4),
        "time_seconds": round(elapsed, 1),
        "tool_calls": tool_call_count,
    }

    if not parsed:
        return {
            "_validation_metadata": {**meta, "error": "failed to parse JSON", "raw_output": output_text[:1000]},
        }

    parsed["_validation_metadata"] = meta
    return parsed


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
