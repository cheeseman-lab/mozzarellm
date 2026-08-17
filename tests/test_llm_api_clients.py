"""Tests for Anthropic client behavior on newer models: structured
empty-response errors and upfront parameter resolution."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import mozzarellm.clients.llm_api_clients as llm
from mozzarellm.clients.llm_api_clients import (
    AnthropicClient,
    AnthropicNoTextError,
    _extract_anthropic_text,
)


def _response(stop_reason, texts=(), thinking=(), stop_details=None):
    content = [SimpleNamespace(type="thinking", thinking=t) for t in thinking]
    content += [SimpleNamespace(type="text", text=t) for t in texts]
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=content,
        usage=SimpleNamespace(input_tokens=10, output_tokens=len(texts)),
        stop_details=stop_details,
        _request_id="req_test",
    )


def test_text_blocks_joined_and_thinking_tolerated():
    resp = _response("end_turn", texts=("part one, ", "part two"), thinking=("hidden",))
    assert _extract_anthropic_text(resp) == "part one, part two"


def test_refusal_not_retryable_and_carries_diagnostics():
    details = SimpleNamespace(category="cyber", explanation="declined")
    with pytest.raises(AnthropicNoTextError) as excinfo:
        _extract_anthropic_text(_response("refusal", stop_details=details))
    err = excinfo.value
    assert err.retryable is False
    assert (err.category, err.explanation, err.request_id) == ("cyber", "declined", "req_test")
    assert "refusal" in str(err)


def test_empty_non_refusal_is_retryable():
    # e.g. the whole max_tokens budget spent on thinking: stochastic, so a
    # retry can succeed -- unlike a deterministic refusal.
    with pytest.raises(AnthropicNoTextError) as excinfo:
        _extract_anthropic_text(_response("max_tokens", thinking=("only thinking",)))
    assert excinfo.value.retryable is True


def _client(model, **kw):
    return AnthropicClient(model=model, api_key="test-key", **kw)


def test_sampling_dropped_on_locked_model():
    c = _client("claude-sonnet-5", temperature=0.2)
    assert "temperature" not in c._sampling_kwargs()
    assert c.resolved_params["dropped"] == ["temperature"]


def test_sampling_kept_on_older_model():
    c = _client("claude-sonnet-4-5", temperature=0.2)
    assert c._sampling_kwargs()["temperature"] == 0.2
    assert c.resolved_params["dropped"] == []


def test_thinking_respects_capability_lookup():
    with patch.object(llm, "_model_supports_enabled_thinking", return_value=False):
        assert _client("claude-sonnet-5", thinking=True)._thinking_kwarg() == {}
    with patch.object(llm, "_model_supports_enabled_thinking", return_value=True):
        kwarg = _client("claude-sonnet-4-5", thinking=True)._thinking_kwarg()
    assert kwarg["thinking"]["type"] == "enabled"


def test_capability_lookup_falls_back_offline():
    llm._THINKING_SUPPORT_CACHE.clear()
    with patch.object(llm.anthropic, "Anthropic", side_effect=RuntimeError("offline")):
        assert llm._model_supports_enabled_thinking("claude-sonnet-5", None) is False
        assert llm._model_supports_enabled_thinking("claude-sonnet-4-5", None) is True
    llm._THINKING_SUPPORT_CACHE.clear()

def test_resolved_params_records_the_full_outcome():
    c = _client(
        "claude-sonnet-5", temperature=0.2, top_p=0.9, top_k=40,
        stop_sequences=["END"], thinking=False,
    )
    c._resolve_params()
    assert c.resolved_params == {
        "sent": {"stop_sequences": ["END"]},
        "dropped": ["temperature", "top_p", "top_k"],
        "thinking": "disabled",
    }


def test_resolution_happens_once_and_is_stable():
    c = _client("claude-sonnet-4-5", temperature=0.2)
    assert c._sampling_kwargs()["temperature"] == 0.2
    c.temperature = 0.9  # post-resolution mutation must not change what is sent
    assert c._sampling_kwargs()["temperature"] == 0.2
    assert c.resolved_params["sent"]["temperature"] == 0.2


def test_thinking_capability_lookup_is_cached_per_model():
    llm._THINKING_SUPPORT_CACHE.clear()
    calls = []

    class _Models:
        def retrieve(self, model):
            calls.append(model)
            return SimpleNamespace(
                capabilities={"thinking": {"types": {"enabled": {"supported": True}}}}
            )

    with patch.object(llm.anthropic, "Anthropic", return_value=SimpleNamespace(models=_Models())):
        assert llm._model_supports_enabled_thinking("claude-sonnet-4-5", "k") is True
        assert llm._model_supports_enabled_thinking("claude-sonnet-4-5", "k") is True
    assert calls == ["claude-sonnet-4-5"]
    llm._THINKING_SUPPORT_CACHE.clear()


def test_enabled_thinking_budget_stays_within_max_tokens():
    with patch.object(llm, "_model_supports_enabled_thinking", return_value=True):
        c = _client("claude-sonnet-4-5", thinking=True, max_tokens=1500)
        budget = c._thinking_kwarg()["thinking"]["budget_tokens"]
    assert 1024 <= budget < 1500
    assert c.resolved_params["thinking"] == "enabled"

