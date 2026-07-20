"""
Unit tests for mozzarellm.clients.llm_api_clients backward-compatibility layer.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import anthropic
import httpx
import pytest

from mozzarellm.clients import llm_api_clients
from mozzarellm.clients.llm_api_clients import (
    AnthropicClient,
    AnthropicNoTextError,
    _extract_anthropic_text,
)


def _resp(stop_reason, blocks, in_tok=10, out_tok=5):
    """Fake messages response: blocks are (type, text) tuples."""
    content = [SimpleNamespace(type=t, text=x) for t, x in blocks]
    usage = SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok)
    return SimpleNamespace(stop_reason=stop_reason, content=content, usage=usage)


def _bad_request(message):
    resp = httpx.Response(400, request=httpx.Request("POST", "http://x"))
    return anthropic.BadRequestError(message, response=resp, body=None)


# =============================================================================
# _extract_anthropic_text
# =============================================================================


def test_extract_joins_text_blocks():
    r = _resp("end_turn", [("text", "hello "), ("text", "world")])
    assert _extract_anthropic_text(r) == "hello world"


def test_extract_skips_thinking_blocks():
    r = _resp("end_turn", [("thinking", "let me reason"), ("text", "answer")])
    assert _extract_anthropic_text(r) == "answer"


def test_extract_refusal_raises_nontext():
    r = _resp("refusal", [])
    with pytest.raises(AnthropicNoTextError) as ei:
        _extract_anthropic_text(r)
    assert ei.value.stop_reason == "refusal"
    assert ei.value.retryable is False


def test_extract_empty_content_raises():
    r = _resp("end_turn", [])
    with pytest.raises(AnthropicNoTextError):
        _extract_anthropic_text(r)


# =============================================================================
# temperature backward-compat (reactive strip + remember)
# =============================================================================


@pytest.fixture(autouse=True)
def _clear_reject_set():
    llm_api_clients._MODELS_REJECTING_TEMPERATURE.clear()
    llm_api_clients._MODELS_REJECTING_THINKING.clear()
    yield
    llm_api_clients._MODELS_REJECTING_TEMPERATURE.clear()
    llm_api_clients._MODELS_REJECTING_THINKING.clear()


def test_temperature_deprecated_is_stripped_and_remembered():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if "temperature" in kwargs:
            raise _bad_request("`temperature` is deprecated for this model.")
        return _resp("end_turn", [("text", "ok")])

    fake_client = Mock()
    fake_client.messages.create.side_effect = create
    with patch.object(anthropic, "Anthropic", return_value=fake_client):
        c = AnthropicClient(model="claude-sonnet-5", temperature=0.2, api_key="x")
        assert c._make_api_call("sys", "user") == "ok"

    # first attempt included temperature (rejected), second omitted it
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]
    assert "claude-sonnet-5" in llm_api_clients._MODELS_REJECTING_TEMPERATURE


def test_other_bad_request_is_not_swallowed():
    fake_client = Mock()
    fake_client.messages.create.side_effect = _bad_request("max_tokens too large")
    with patch.object(anthropic, "Anthropic", return_value=fake_client):
        c = AnthropicClient(model="claude-sonnet-4-5", temperature=0.2, api_key="x")
        with pytest.raises(anthropic.BadRequestError):
            c._make_api_call("sys", "user")


def test_temperature_error_retries_even_if_already_in_reject_set():
    # Concurrency race: another worker already recorded the model as rejecting
    # temperature between this request being built and its 400 arriving. The
    # error must still trigger a retry (which omits temperature), not a raise.
    llm_api_clients._MODELS_REJECTING_TEMPERATURE.add("claude-sonnet-5")
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise _bad_request("`temperature` is deprecated for this model.")
        return _resp("end_turn", [("text", "ok")])

    fake_client = Mock()
    fake_client.messages.create.side_effect = create
    with patch.object(anthropic, "Anthropic", return_value=fake_client):
        c = AnthropicClient(model="claude-sonnet-5", temperature=0.2, api_key="x")
        assert c._make_api_call("sys", "user") == "ok"
    assert len(calls) == 2  # retried rather than re-raising


def test_temperature_kept_for_accepting_model():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return _resp("end_turn", [("text", "ok")])

    fake_client = Mock()
    fake_client.messages.create.side_effect = create
    with patch.object(anthropic, "Anthropic", return_value=fake_client):
        c = AnthropicClient(model="claude-sonnet-4-5", temperature=0.2, api_key="x")
        c._make_api_call("sys", "user")
    assert calls[0]["temperature"] == 0.2


# =============================================================================
# thinking param (backward compatible)
# =============================================================================


def test_thinking_false_disables_and_is_sent():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return _resp("end_turn", [("text", "ok")])

    fake_client = Mock()
    fake_client.messages.create.side_effect = create
    with patch.object(anthropic, "Anthropic", return_value=fake_client):
        c = AnthropicClient(model="claude-sonnet-5", thinking=False, api_key="x")
        c._make_api_call("sys", "user")
    assert calls[0]["thinking"] == {"type": "disabled"}


def test_thinking_none_omits_param():
    c = AnthropicClient(model="claude-sonnet-4-5", thinking=None, api_key="x")
    assert c._thinking_kwarg() == {}


def test_thinking_true_enabled_with_budget_under_max_tokens():
    c = AnthropicClient(model="claude-sonnet-5", thinking=True, max_tokens=16000, api_key="x")
    kw = c._thinking_kwarg()
    assert kw["thinking"]["type"] == "enabled"
    assert kw["thinking"]["budget_tokens"] < 16000


def test_thinking_unsupported_is_stripped_and_remembered():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if "thinking" in kwargs:
            raise _bad_request("`thinking` is not supported for this model")
        return _resp("end_turn", [("text", "ok")])

    fake_client = Mock()
    fake_client.messages.create.side_effect = create
    with patch.object(anthropic, "Anthropic", return_value=fake_client):
        c = AnthropicClient(model="claude-legacy", thinking=False, api_key="x")
        assert c._make_api_call("sys", "user") == "ok"
    assert "thinking" in calls[0]
    assert "thinking" not in calls[1]
    assert "claude-legacy" in llm_api_clients._MODELS_REJECTING_THINKING


# =============================================================================
# query wrapper: refusal is non-retryable
# =============================================================================


def test_query_does_not_retry_refusal():
    fake_client = Mock()
    fake_client.messages.create.return_value = _resp("refusal", [], out_tok=1)
    with patch.object(anthropic, "Anthropic", return_value=fake_client):
        c = AnthropicClient(model="claude-sonnet-4-5", temperature=0.2, api_key="x")
        txt, err = c.query(system_prompt="sys", user_prompt="user", max_retries=3)
    assert txt is None
    assert "AnthropicNoTextError" in err
    # non-retryable → create called exactly once, not max_retries times
    assert fake_client.messages.create.call_count == 1
