"""Per-cluster reasoning trace persistence.

Every code path that calls the LLM (standard, CoT, MCP, CoT+MCP, future
variants) writes a uniform trace record alongside the parsed cluster JSON.
The parsed result remains the function's default return value; this trace is
the unified raw-output record (response text + tool calls + metadata) that
captures everything needed to audit a run after the fact.

Trace shape is owned here only — `save_trace()` is the single entrypoint.
Each cluster gets one file: `<run_dir>/traces/cluster_<id>.json`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def save_trace(
    run_dir: Path,
    cluster_id: str,
    *,
    model: str,
    mode: str,
    raw_response: str,
    tool_calls: list[dict[str, Any]] | None = None,
    started_at: datetime | None = None,
    elapsed_s: float | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float | None = None,
    pricing_warning: str | None = None,
    schema_warnings: list[str] | None = None,
    error: str | None = None,
    steps: list[dict[str, Any]] | None = None,
    **extra: Any,
) -> Path:
    """Write a uniform trace JSON for one cluster call. Returns the file path.

    `tool_calls` is empty for the non-MCP path, populated by
    `extract_mcp_tool_calls()` for the MCP path.
    """
    traces_dir = run_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    trace: dict[str, Any] = {
        "cluster_id": str(cluster_id),
        "model": model,
        "mode": mode,
        "started_at": (started_at or datetime.now()).isoformat(timespec="seconds"),
        "elapsed_s": round(elapsed_s, 2) if elapsed_s is not None else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "pricing_warning": pricing_warning,
        "raw_response": raw_response,
        "tool_calls": tool_calls or [],
        "schema_warnings": schema_warnings or [],
        "error": error,
        "steps": steps or [],
    }
    if extra:
        trace.update(extra)

    out = traces_dir / f"cluster_{cluster_id}.json"
    out.write_text(json.dumps(trace, indent=2, ensure_ascii=False, default=str))
    return out


def extract_mcp_tool_calls(response_content: list[Any]) -> list[dict[str, Any]]:
    """Pull structured tool_use / tool_result blocks out of an Anthropic beta response.

    Pairs each `mcp_tool_use` with its matching `mcp_tool_result` by id, so the
    trace records both inputs and outputs in one entry per call.
    """
    uses: dict[str, dict[str, Any]] = {}
    results: dict[str, Any] = {}
    order: list[str] = []

    for block in response_content:
        btype = getattr(block, "type", None)
        if btype == "mcp_tool_use":
            tool_use_id = getattr(block, "id", None) or f"call_{len(order)}"
            uses[tool_use_id] = {
                "id": tool_use_id,
                "name": getattr(block, "name", None),
                "server_name": getattr(block, "server_name", None),
                "input": getattr(block, "input", None),
            }
            order.append(tool_use_id)
        elif btype == "mcp_tool_result":
            tool_use_id = getattr(block, "tool_use_id", None)
            content = getattr(block, "content", None)
            if isinstance(content, list):
                payload = []
                for c in content:
                    if hasattr(c, "model_dump"):
                        payload.append(c.model_dump())
                    elif hasattr(c, "text"):
                        payload.append({"type": "text", "text": c.text})
                    else:
                        payload.append(repr(c))
                results[tool_use_id] = {
                    "is_error": getattr(block, "is_error", False),
                    "content": payload,
                }
            else:
                results[tool_use_id] = {
                    "is_error": getattr(block, "is_error", False),
                    "content": content,
                }

    calls = []
    for tool_use_id in order:
        entry = dict(uses[tool_use_id])
        entry["output"] = results.get(tool_use_id)
        calls.append(entry)
    return calls
