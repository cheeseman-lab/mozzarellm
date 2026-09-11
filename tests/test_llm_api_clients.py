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



# ---------------------------------------------------------------------------
# _create_message: streaming for large outputs
# ---------------------------------------------------------------------------


class _FakeStream:
    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self._message


class _FakeMessages:
    def __init__(self):
        self.created_with = None
        self.streamed_with = None

    def create(self, **kwargs):
        self.created_with = kwargs
        return "created"

    def stream(self, **kwargs):
        self.streamed_with = kwargs
        return _FakeStream("streamed")


class _FakeAnthropic:
    def __init__(self):
        self.messages = _FakeMessages()


def _anthropic_client(max_tokens):
    from mozzarellm.clients.llm_api_clients import AnthropicClient

    return AnthropicClient(
        "claude-sonnet-5", 0.2, max_tokens, None, None, None, "test-key", False
    )


def test_benchmark_ceiling_stays_non_streaming():
    client = _anthropic_client(16000)
    fake = _FakeAnthropic()
    assert client._create_message(fake, {"max_tokens": 16000}) == "created"
    assert fake.messages.streamed_with is None


def test_large_outputs_stream_and_return_the_final_message():
    client = _anthropic_client(32000)
    fake = _FakeAnthropic()
    assert client._create_message(fake, {"max_tokens": 32000}) == "streamed"
    assert fake.messages.created_with is None
    assert fake.messages.streamed_with["max_tokens"] == 32000


def test_stepwise_uses_provided_turns():
    """Prebuilt turns (with overrides applied) are what the API actually receives."""
    c = _client("claude-sonnet-5")
    turns = [
        {"content": "STEP 1 - TUNED FIRST", "mcp": False},
        {"content": "STEP 2 - TUNED SECOND", "mcp": False},
    ]
    seen = []

    def fake_endpoint(*, system_prompt, messages, max_tokens, max_retries):
        seen.append([m["content"] for m in messages if m["role"] == "user"])
        return _response("end_turn", texts=('{"cluster_id": "1"}',)), 0.1

    with patch.object(c, "_call_messages_endpoint", side_effect=fake_endpoint):
        parsed, raw = c._analyze_stepwise(
            system_prompt="sys", user_prompt="bundle", mcp=False, max_retries=1, turns=turns
        )
    assert "TUNED FIRST" in seen[0][0]
    assert any("TUNED SECOND" in u for u in seen[-1])
    assert len(raw["steps"]) == 2


def test_analyze_rejects_turns_outside_stepwise():
    c = _client("claude-sonnet-5")
    with pytest.raises(ValueError, match="stepwise_turns"):
        c.analyze(
            system_prompt="sys",
            user_prompt="u",
            mode="cot",
            stepwise_turns=[{"content": "x", "mcp": False}],
        )


def test_stepwise_truncation_labeled_not_parse_failure():
    """A max_tokens stop with unparseable text is reported as truncation."""
    c = _client("claude-sonnet-5")

    def fake_endpoint(*, system_prompt, messages, max_tokens, max_retries):
        return _response("max_tokens", texts=('{"cluster_id": ',)), 0.1

    turns = [{"content": "STEP 1 - only step", "mcp": False}]
    with patch.object(c, "_call_messages_endpoint", side_effect=fake_endpoint):
        parsed, raw = c._analyze_stepwise(
            system_prompt="sys", user_prompt="bundle", mcp=False, max_retries=1, turns=turns
        )
    assert raw["error"] == "output truncated at max_tokens"
    assert raw["steps"][0]["stop_reason"] == "max_tokens"


# ---------------------------------------------------------------------------
# Provider-generic analyze (OpenAI / Gemini path)
# ---------------------------------------------------------------------------

_VALID_JSON = """{"cluster_id": "1", "dominant_process": "proteasome", "pathway_confidence": "High",
"established_genes": ["PSMA1"], "novel_role_genes": [], "uncharacterized_genes": []}"""


def _generic_client(response_text=_VALID_JSON, fail=False):
    from mozzarellm.clients.llm_api_clients import OpenAIClient

    class _Stub(OpenAIClient):
        def _make_api_call(self, system_prompt, user_prompt):
            if fail:
                raise RuntimeError("provider down")
            self._last_usage = {"input_tokens": 100, "output_tokens": 50}
            return response_text

    return _Stub("gpt-test", 0.2, 1000, None, None, None, "test-key")


def test_generic_analyze_parses_and_reports_usage():
    parsed, raw = _generic_client().analyze(system_prompt="s", user_prompt="u", mode="cot")
    assert parsed["dominant_process"] == "proteasome"
    assert raw["input_tokens"] == 100 and raw["output_tokens"] == 50
    assert raw["cost_usd"] is not None and raw["pricing_warning"]  # unknown model -> warned
    assert raw["error"] is None


def test_generic_analyze_rejects_anthropic_only_features():
    import pytest

    client = _generic_client()
    with pytest.raises(ValueError, match="Anthropic-only"):
        client.analyze(system_prompt="s", user_prompt="u", mcp=True)
    with pytest.raises(ValueError, match="stepwise is Anthropic-only"):
        client.analyze(system_prompt="s", user_prompt="u", mode="stepwise")


def test_generic_analyze_surfaces_provider_errors():
    parsed, raw = _generic_client(fail=True).analyze(
        system_prompt="s", user_prompt="u", max_retries=1
    )
    assert parsed is None
    assert "provider down" in raw["error"]
