"""Per-model token pricing for Anthropic Claude models.

Used by the benchmark and literature-validation pipelines to compute USD cost
without hard-coding Sonnet rates everywhere.
"""

from __future__ import annotations

# (input_$_per_M_tokens, output_$_per_M_tokens)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-7": (15.0, 75.0),
    "claude-opus-4-5": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

_FALLBACK_KEY = "claude-sonnet-4-6"


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> tuple[float, str | None]:
    """Compute USD cost for a model call.

    Returns:
        (cost_usd, warning) — warning is None on hit, otherwise a string explaining
        the fallback to Sonnet rates.
    """
    rates = MODEL_PRICING.get(model)
    warning: str | None = None
    if rates is None:
        rates = MODEL_PRICING[_FALLBACK_KEY]
        warning = f"unknown model '{model}'; cost computed at Sonnet rates"
    p_in, p_out = rates
    cost = (input_tokens * p_in + output_tokens * p_out) / 1_000_000
    return round(cost, 4), warning
