"""Per-model token pricing for Anthropic Claude models.

Used by the benchmark and literature-validation pipelines to compute USD cost
without hard-coding Sonnet rates everywhere.
"""

from __future__ import annotations

# (input_$_per_M_tokens, output_$_per_M_tokens)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-4-7": (15.0, 75.0),
    "claude-opus-4-5": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

_FALLBACK_KEY = "claude-sonnet-4-6"


def compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> tuple[float, str | None]:
    """Compute USD cost for a model call.

    ``input_tokens`` is the uncached input (the API reports cached tokens
    separately); cache writes bill at 1.25x the input rate and cache reads at
    0.1x, per the prompt-caching pricing rules.

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
    cost = (
        input_tokens * p_in
        + cache_creation_tokens * p_in * 1.25
        + cache_read_tokens * p_in * 0.1
        + output_tokens * p_out
    ) / 1_000_000
    return round(cost, 4), warning
