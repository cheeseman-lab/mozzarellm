"""
Literature validation utilities for MozzareLLM gene classifications.

Provides the low-level Anthropic beta MCP call (`call_mcp`), MCP server
preflight (`get_available_mcp_servers`), and helpers for parsing/validating
MCP responses. Orchestration (one-shot, multi-turn stepwise) lives on
`AnthropicClient` in `mozzarellm.clients.llm_api_clients`.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import anthropic
from pydantic import ValidationError

from mozzarellm.schemas.mcp_schemas import (
    LiteraturePathwayRevision,
    LiteratureReclassification,
    LiteratureValidation,
)

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


def _build_mcp_servers_and_tools() -> tuple[list[dict], list[dict]]:
    """Construct mcp_servers + tools payloads for the Anthropic beta API."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    mcp_servers = [
        {
            "type": "url",
            "url": url,
            "name": name,
            **(
                {"authorization_token": api_key}
                if any(h in url for h in ANTHROPIC_HOSTED_MCP)
                else {}
            ),
        }
        for name, url in MCP_SERVERS.items()
    ]
    tools = [{"type": "mcp_toolset", "mcp_server_name": name} for name in MCP_SERVERS]
    return mcp_servers, tools


RETRYABLE_API_EXCEPTIONS: tuple = (
    anthropic.BadRequestError,
    anthropic.InternalServerError,
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
)

# Hard ceiling per call. Without this, the SDK falls back to a 600s default that
# combined with retries can run an MCP-spamming model for ~25 minutes per cluster.
PER_CALL_TIMEOUT_S = 300


def _is_retryable_api_error(exc: Exception) -> bool:
    if isinstance(exc, (anthropic.APITimeoutError, anthropic.APIConnectionError)):
        return True
    msg = str(exc).lower()
    return any(
        kw in msg
        for kw in (
            "timed out",
            "504",
            "502",
            "503",
            "500",
            "connection error",
            "internal server",
        )
    )


def call_mcp(
    *,
    system_prompt: str,
    messages: list[dict[str, Any]],
    model: str,
    max_tokens: int = 16000,
    max_retries: int = 3,
) -> tuple[Any, float]:
    """ONE place that talks to the Anthropic beta MCP endpoint. Returns (response, elapsed_s).

    All three execution modes (single+mcp, cot+mcp, stepwise+mcp at the literature step)
    funnel through here. Caller is responsible for shaping `messages` and parsing the
    response.
    """
    client = anthropic.Anthropic()
    mcp_servers, tools = _build_mcp_servers_and_tools()

    for attempt in range(max_retries):
        try:
            start = time.time()
            response = client.beta.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages,
                mcp_servers=mcp_servers,
                tools=tools,
                betas=["mcp-client-2025-11-20"],
                timeout=PER_CALL_TIMEOUT_S,
            )
            return response, time.time() - start
        except RETRYABLE_API_EXCEPTIONS as e:
            if attempt < max_retries - 1 and _is_retryable_api_error(e):
                time.sleep(60)
                continue
            raise


def _validate_literature_blocks(parsed: dict[str, Any]) -> list[str]:
    """Soft-validate literature-specific structures. Never raises."""
    warnings: list[str] = []

    rev = parsed.get("literature_informed_pathway_revision")
    if isinstance(rev, dict):
        try:
            LiteraturePathwayRevision.model_validate(rev)
        except ValidationError as e:
            warnings.append(f"pathway_revision: {e.errors()[0]['msg']}")

    for i, r in enumerate(parsed.get("literature_informed_reclassifications") or []):
        try:
            LiteratureReclassification.model_validate(r)
        except ValidationError as e:
            gene = r.get("gene", "?") if isinstance(r, dict) else "?"
            warnings.append(f"reclassification[{i}/{gene}]: {e.errors()[0]['msg']}")

    for category in ("novel_role_genes", "uncharacterized_genes"):
        for g in parsed.get(category) or []:
            lv = g.get("literature_validation") if isinstance(g, dict) else None
            if lv is None:
                continue
            try:
                LiteratureValidation.model_validate(lv)
            except ValidationError as e:
                gene = g.get("gene", "?")
                warnings.append(f"{category}/{gene}.literature_validation: {e.errors()[0]['msg']}")

    return warnings


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
