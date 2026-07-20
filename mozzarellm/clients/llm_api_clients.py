"""
Unified LLM provider system for mozzarellm.

This module provides a consistent interface via client classes for querying different LLM providers
(OpenAI, Anthropic, Google Gemini) with automatic retry logic and error handling.
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import anthropic
from anthropic.types.messages.batch_create_params import Request

# NOTE: client specific imports (other than anthropic) are done in the methods to avoid import-time failures for optional dependencies


logger = logging.getLogger(__name__)


# Models that reject a non-default `temperature` (deprecated on newer Claude
# models). Populated reactively the first time a model rejects it, so new models
# are handled without a hardcoded list.
_MODELS_REJECTING_TEMPERATURE: set[str] = set()

# Models that reject an explicit `thinking` param. Populated reactively so old
# models that predate the param still work when thinking control is requested.
_MODELS_REJECTING_THINKING: set[str] = set()


class AnthropicNoTextError(RuntimeError):
    """Anthropic returned no usable text block (e.g. a safety-classifier refusal).

    Carries the API's stop_reason so callers can distinguish a refusal from other
    empty responses. Not retryable: the same prompt yields the same result.
    """

    retryable = False

    def __init__(
        self, stop_reason: str | None, input_tokens: int | None, output_tokens: int | None
    ):
        self.stop_reason = stop_reason
        hint = " (safety-classifier refusal)" if stop_reason == "refusal" else ""
        super().__init__(
            f"Anthropic returned no text{hint}: stop_reason={stop_reason!r}, "
            f"input_tokens={input_tokens}, output_tokens={output_tokens}"
        )


def _is_temperature_deprecated(exc: Exception) -> bool:
    """True if the error is the 'temperature is deprecated for this model' 400."""
    msg = str(exc).lower()
    return "temperature" in msg and "deprecat" in msg


def _is_thinking_unsupported(exc: Exception) -> bool:
    """True if the error complains about the `thinking` param (old models)."""
    return "thinking" in str(exc).lower()


def _extract_anthropic_text(response) -> str:
    """Join all text blocks from a messages response, tolerating thinking blocks.

    Raises AnthropicNoTextError on a refusal or any response with no text block,
    rather than blindly indexing content[0].
    """
    usage = getattr(response, "usage", None)
    in_tok = getattr(usage, "input_tokens", None)
    out_tok = getattr(usage, "output_tokens", None)
    if response.stop_reason == "refusal":
        raise AnthropicNoTextError(response.stop_reason, in_tok, out_tok)
    texts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    if not texts:
        raise AnthropicNoTextError(response.stop_reason, in_tok, out_tok)
    return "".join(texts)


class LLMClientBase(ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 16000,
        top_p: float | None = None,
        top_k: int | None = None,
        stop_sequences: list[str] | None = None,
        api_key: str | None = None,
        thinking: bool | None = None,
    ):
        """
        Initialize LLM provider.

        Args:
            model: Model identifier (e.g., "gpt-4o", "claude-sonnet-4-5-20250929")
            temperature: Temperature for generation (0.0-1.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter (0.0-1.0, optional)
            top_k: Top-K sampling parameter (optional, Claude/Gemini only)
            stop_sequences: List of stop sequences (optional)
            api_key: API key (if None, reads from environment)
            thinking: Extended-thinking control (Claude only). None = model default,
                False = disabled, True = enabled with a budget under max_tokens.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.stop_sequences = stop_sequences
        self.thinking = thinking
        self.api_key = api_key or self._get_api_key_from_env()
        self._validate_api_key()

    @abstractmethod
    def _get_api_key_from_env(self) -> str | None:
        """Get API key from environment variable."""
        pass

    @abstractmethod
    def _get_env_var_name(self) -> str:
        """Get name of environment variable for API key."""
        pass

    def _validate_api_key(self):
        """Validate that API key is available."""
        if not self.api_key:
            raise ValueError(f"{self._get_env_var_name()} not found in environment or constructor")

    @abstractmethod
    def _make_api_call(self, system_prompt: str, user_prompt: str) -> str:
        """Make the actual API call to the provider."""
        pass

    @abstractmethod
    def _make_batch_api_call(self, system_prompt: str, user_prompt: str) -> str:
        """Make a batched API call to the provider."""
        pass

    def query(
        self,
        *,
        max_retries: int = 3,
        batch: bool = False,
        screen_name: str | None = None,
        system_prompt: str,
        # use one but not both of the following parameters
        user_prompt: str | None = None,
        cluster_to_prompt_map: dict[str, str] | None = None,  # *****
    ) -> tuple[str | None, str | None]:
        """
        Main wrapper function for querying the LLM with retry logic.

        Args:
            system_prompt: System/context prompt
            user_prompt: User query prompt
            max_retries: Maximum number of retry attempts

        Returns:
            Tuple of (response_text, error_message)
            If successful: (text, None)
            If failed: (None, error_message)
        """
        for attempt in range(max_retries):
            try:
                if batch:
                    response = self._make_batch_api_call(
                        cluster_to_prompt_map,
                        screen_name,
                        system_prompt,
                    )
                else:
                    response = self._make_api_call(system_prompt, user_prompt)
                logger.info(
                    f"{self.__class__.__name__} call successful "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                return response, None

            except Exception as e:
                error_str = str(e)[:100]  # Truncate long errors
                error_msg = (
                    f"Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {error_str}"
                )
                logger.warning(f"{self.__class__.__name__}: {error_msg}")

                if not getattr(e, "retryable", True):
                    # Deterministic failure (e.g. safety refusal) — retrying the
                    # same prompt cannot help; return immediately.
                    return None, f"{type(e).__name__}: {e}"

                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    return None, f"API error after {max_retries} attempts: {str(e)}"

        return None, "Maximum retries exceeded"


class OpenAIClient(LLMClientBase):
    """OpenAI API provider (GPT-4, o4-mini, o3-mini, etc.)"""

    def _get_env_var_name(self) -> str:
        return "OPENAI_API_KEY"

    def _get_api_key_from_env(self) -> str | None:
        return os.environ.get(self._get_env_var_name())

    def _make_api_call(self, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Build kwargs with optional parameters
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
            "seed": 42,  # For reproducibility
        }

        # Add optional sampling parameters
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.stop_sequences:
            kwargs["stop"] = self.stop_sequences

        response = client.chat.completions.create(**kwargs)

        # Log token usage
        if hasattr(response, "usage"):
            tokens = response.usage.total_tokens
            logger.info(f"OpenAI tokens used: {tokens}")

        return response.choices[0].message.content

    def _make_batch_api_call(
        self,
        cluster_to_prompt_map: dict[str, str],
        screen_name: str,
        system_prompt: str,
    ) -> list[str]:
        """
        Makes a batch of requests to the OpenAI chat API.
        """
        pass  # TODO


class AnthropicClient(LLMClientBase):
    """Anthropic API provider (Claude models)"""

    def _get_env_var_name(self) -> str:
        return "ANTHROPIC_API_KEY"

    def _get_api_key_from_env(self) -> str | None:
        return os.environ.get(self._get_env_var_name())

    ### Helper functions for batch requests ###
    def _make_single_cluster_message_request(
        self, cluster_id: str, path_to_evidence_bundle: str, system_prompt: str
    ) -> Request:
        bundle_obj = json.loads(Path(path_to_evidence_bundle).read_text(encoding="utf-8"))
        bundle_text = json.dumps(
            bundle_obj, ensure_ascii=False
        )  # minify JSON; has no effect on readability for LLMs + saves tokens

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {  # caching to reduce costs
                        "type": "ephemeral",
                        "ttl": "5m",  # can set ttl to 1h if needed
                    },
                },
            ],
            # "thinking": {"type": "enabled"},  # can eventually set a token budget here
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Here is the evidence bundle JSON for cluster "
                            + cluster_id
                            + ":\n\n```json\n"
                            + bundle_text
                            + "\n```",
                        }
                    ],
                }
            ],
        }

        # Add optional sampling + thinking params (omitted for models that reject them)
        kwargs.update(self._sampling_kwargs())
        kwargs.update(self._thinking_kwarg())

        cluster_request = Request(
            custom_id=f"{cluster_id}_analysis_request",
            params=kwargs,
        )
        return cluster_request

    def _make_list_of_cluster_request_objs(
        self, cluster_to_prompt_map: dict[str, str], system_prompt: str
    ) -> list[str]:
        """
        Returns a list of Request objects for the Anthropic batch messages API.
        """
        requests = [
            # NOTE: user_prompt is unique to the cluster
            self._make_single_cluster_message_request(
                cluster_id, path_to_evidence_bundle, system_prompt
            )
            for cluster_id, path_to_evidence_bundle in cluster_to_prompt_map.items()
        ]
        return requests

    ### Endpoint access functions ###
    def _sampling_kwargs(self) -> dict:
        """Optional sampling params, omitting temperature for models that reject it."""
        kwargs: dict = {}
        if self.model not in _MODELS_REJECTING_TEMPERATURE:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.stop_sequences:
            kwargs["stop_sequences"] = self.stop_sequences
        return kwargs

    def _thinking_kwarg(self) -> dict:
        """Extended-thinking param, omitted when unset or unsupported by the model."""
        if self.thinking is None or self.model in _MODELS_REJECTING_THINKING:
            return {}
        if self.thinking is False:
            return {"thinking": {"type": "disabled"}}
        # enabled: budget must stay under max_tokens to leave room for the answer
        # (the max_tokens-1 cap protects small-budget callers from the 1024 floor)
        budget = min(max(1024, min(self.max_tokens // 2, 8000)), self.max_tokens - 1)
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}

    def _create_message(self, client, base: dict):
        """messages.create with reactive param compat (temperature / thinking)."""
        last_exc: Exception | None = None
        for _ in range(3):  # allow stripping temperature and/or thinking, then retry
            try:
                return client.messages.create(
                    **base, **self._sampling_kwargs(), **self._thinking_kwarg()
                )
            except anthropic.BadRequestError as e:
                last_exc = e
                # Retry whenever the error names a strippable param, even if another
                # thread already recorded the model (the next call omits it anyway);
                # gating the retry on set-membership races parallel workers into raises.
                if _is_temperature_deprecated(e):
                    if self.model not in _MODELS_REJECTING_TEMPERATURE:
                        _MODELS_REJECTING_TEMPERATURE.add(self.model)
                        logger.info(f"{self.model}: temperature deprecated; retrying without it")
                    continue
                if _is_thinking_unsupported(e):
                    if self.model not in _MODELS_REJECTING_THINKING:
                        _MODELS_REJECTING_THINKING.add(self.model)
                        logger.info(
                            f"{self.model}: thinking param unsupported; retrying without it"
                        )
                    continue
                raise
        raise last_exc  # pragma: no cover

    def _make_api_call(self, system_prompt: str, user_prompt: str) -> str:
        """
        Makes a single request to the Anthropic messages API.
        https://platform.claude.com/docs/en/api/python/messages/create

        Backward compatible across models: omits `temperature` / `thinking` for
        models that reject them (reactively, on the first 400), tolerates thinking
        blocks in the response, and raises AnthropicNoTextError on refusal / empty.
        """

        client = anthropic.Anthropic(api_key=self.api_key)

        base = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
        }

        response = self._create_message(client, base)

        # Store usage for cost tracking by callers
        if hasattr(response, "usage"):
            self.last_usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            logger.info(
                f"Anthropic tokens used: {response.usage.input_tokens + response.usage.output_tokens}"
            )
        else:
            self.last_usage = {}

        return _extract_anthropic_text(response)

    def _make_batch_api_call(
        self,
        cluster_to_prompt_map: dict[str, str],
        screen_name: str,
        system_prompt: str,
    ) -> list[str]:
        """
        Makes a batch of requests to the Anthropic messages API.
        https://platform.claude.com/docs/en/build-with-claude/batch-processing
        """

        client = anthropic.Anthropic(api_key=self.api_key)
        request_list = self._make_list_of_cluster_request_objs(cluster_to_prompt_map, system_prompt)
        message_batch = client.messages.batches.create(requests=request_list)
        batch_id = message_batch.id
        # saving the batch ID with a timestamp in a text file for reference
        Path(
            f"output/{screen_name}_analysis/intermediates/msg_batch_ID_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        ).write_text(batch_id)
        print(
            f"Batch {batch_id} created. Saving as {batch_id}.txt in output/{screen_name}_analysis/intermediates/"
        )
        # Polling for message batch completion
        while True:
            message_batch = client.messages.batches.retrieve(batch_id)
            if message_batch.processing_status == "ended":
                break
            print(f"Batch {batch_id} is still processing...")
            time.sleep(60)
        # Stream results file in memory-efficient chunks, processing one at a time
        errored_requests = []
        for result in client.messages.batches.results(batch_id):
            match result.result.type:
                case "succeeded":
                    path = Path(
                        f"output/{screen_name}_analysis/phase1_batch_cluster_LLM_analysis/{result.custom_id}_analysis_response.jsonl"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(str(result.result.message.content), encoding="utf-8")
                    print(f"{result.custom_id} succeeded. Saving response to {path}")
                case "errored":
                    error_obj = result.result.error
                    error_type = getattr(getattr(error_obj, "error", None), "type", "unknown")
                    error_msg = getattr(getattr(error_obj, "error", None), "message", None)
                    if error_type == "invalid_request_error":
                        print(f"Validation error {result.custom_id}: {error_type} -- {error_msg}")

                    else:
                        print(f"Server error {result.custom_id}: {error_type} -- {error_msg}")
                        errored_requests.append(result.custom_id)
                    # print(str(error_obj)) # DEBUG: uncomment for full error object
                case "expired":
                    print(f"Request expired {result.custom_id}")
                    errored_requests.append(result.custom_id)

        if errored_requests:
            print(f"\n{len(errored_requests)} request(s) failed: {errored_requests}")
        # TODO: log response metadata

    ### Unified analyze() — single entry point for all (mode × mcp × batch) combos ###

    def analyze(
        self,
        *,
        system_prompt: str,
        user_prompt: str | None = None,
        cluster_to_prompt_map: dict[str, str] | None = None,
        mode: str = "cot",
        mcp: bool = False,
        batch: bool = False,
        screen_name: str | None = None,
        max_retries: int = 3,
    ) -> tuple[dict | None, dict]:
        """Single entry point for cluster analysis.

        Routes on (mode, mcp, batch) — always returns (parsed, raw_outputs):

          mode in {standard, cot}, mcp=False  -> _analyze_plain
          mode in {standard, cot}, mcp=True   -> _analyze_mcp
          mode == "stepwise", mcp=any         -> _analyze_stepwise
          batch=True                          -> _analyze_batch (mcp/stepwise rejected)

        For batch, parsed is None (results write to disk during retrieval);
        raw_outputs carries batch_id and any errored custom_ids. raw_outputs has
        the uniform shape consumed by `mozzarellm.utils.trace.save_trace()`.
        """
        if mode not in ("standard", "cot", "stepwise"):
            raise ValueError(f"mode must be one of 'standard', 'cot', 'stepwise'; got {mode!r}")
        if batch and (mcp or mode == "stepwise"):
            raise ValueError(
                "batch=True is incompatible with mcp=True or mode='stepwise' "
                "(Anthropic batch API does not support beta MCP, and stepwise "
                "requires multi-turn conversation)."
            )

        if batch:
            if cluster_to_prompt_map is None or screen_name is None:
                raise ValueError("batch=True requires cluster_to_prompt_map and screen_name")
            return self._analyze_batch(
                system_prompt=system_prompt,
                cluster_to_prompt_map=cluster_to_prompt_map,
                screen_name=screen_name,
            )

        if user_prompt is None:
            raise ValueError("user_prompt is required for single/iterative dispatch")

        if mode == "stepwise":
            return self._analyze_stepwise(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                mcp=mcp,
                max_retries=max_retries,
            )

        if mcp:
            return self._analyze_mcp(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                mode=mode,
                max_retries=max_retries,
            )

        return self._analyze_plain(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            mode=mode,
            max_retries=max_retries,
        )

    def _mode_tag(self, mode: str, mcp: bool) -> str:
        return f"{mode}_mcp" if mcp else mode

    def _empty_raw_outputs(self) -> dict:
        return {
            "response_text": "",
            "tool_calls": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "elapsed_s": 0.0,
            "cost_usd": 0.0,
            "pricing_warning": None,
            "schema_warnings": [],
            "error": None,
            "steps": [],
        }

    def _analyze_plain(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        mode: str,
        max_retries: int,
    ) -> tuple[dict | None, dict]:
        """Sync, non-MCP path. Wraps `query()` + parses JSON + builds raw_outputs."""
        from mozzarellm.utils.llm_analysis_utils import process_cluster_response
        from mozzarellm.utils.pricing import compute_cost

        t0 = time.time()
        response_text, error = self.query(
            system_prompt=system_prompt, user_prompt=user_prompt, max_retries=max_retries
        )
        elapsed = time.time() - t0
        usage = getattr(self, "last_usage", {}) or {}
        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        cost, pricing_warning = compute_cost(self.model, in_tok, out_tok)

        raw_outputs = self._empty_raw_outputs()
        raw_outputs.update(
            {
                "response_text": response_text or "",
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "elapsed_s": elapsed,
                "cost_usd": cost,
                "pricing_warning": pricing_warning,
                "error": error,
            }
        )
        if error:
            return None, raw_outputs
        try:
            parsed = process_cluster_response(response_text)
        except Exception as e:
            raw_outputs["error"] = f"parse_error: {e}"
            return None, raw_outputs

        parsed["_validation_metadata"] = {
            "mode": self._mode_tag(mode, mcp=False),
            "model": self.model,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost_usd": cost,
            "time_seconds": round(elapsed, 1),
        }
        if pricing_warning:
            parsed["_validation_metadata"]["pricing_warning"] = pricing_warning
        return parsed, raw_outputs

    def _analyze_mcp(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        mode: str,
        max_retries: int,
        max_tokens: int = 16000,
    ) -> tuple[dict | None, dict]:
        """One-shot cluster analysis with PubMed MCP tools attached. Used by both
        (mode=standard, mcp=True) and (mode=cot, mcp=True) — they differ only in the
        system prompt's delivery format (squashed vs. STEP-numbered)."""
        from mozzarellm.pipeline.literature_mcp import (
            _parse_json_from_text,
            _validate_literature_blocks,
            call_mcp,
        )
        from mozzarellm.utils.pricing import compute_cost
        from mozzarellm.utils.trace import extract_mcp_tool_calls

        response, elapsed = call_mcp(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            model=self.model,
            max_tokens=max_tokens,
            max_retries=max_retries,
            client=anthropic.Anthropic(api_key=self.api_key),
        )

        output_text = "".join(
            b.text for b in response.content if getattr(b, "type", None) == "text"
        )
        tool_calls = extract_mcp_tool_calls(response.content)
        parsed = _parse_json_from_text(output_text)
        cost, pricing_warning = compute_cost(
            self.model, response.usage.input_tokens, response.usage.output_tokens
        )

        raw_outputs = self._empty_raw_outputs()
        raw_outputs.update(
            {
                "response_text": output_text,
                "tool_calls": tool_calls,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "elapsed_s": elapsed,
                "cost_usd": cost,
                "pricing_warning": pricing_warning,
            }
        )

        meta: dict = {
            "mode": self._mode_tag(mode, mcp=True),
            "model": self.model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cost_usd": cost,
            "time_seconds": round(elapsed, 1),
            "tool_calls": len(tool_calls),
        }
        if pricing_warning:
            meta["pricing_warning"] = pricing_warning

        if not parsed:
            meta_err = {**meta, "error": "failed to parse JSON", "raw_output": output_text[:1000]}
            raw_outputs["error"] = "failed to parse JSON"
            return ({"_validation_metadata": meta_err}, raw_outputs)

        schema_warnings = _validate_literature_blocks(parsed)
        if schema_warnings:
            meta["schema_warnings"] = schema_warnings
            raw_outputs["schema_warnings"] = schema_warnings

        parsed["_validation_metadata"] = meta
        return parsed, raw_outputs

    def _call_messages_endpoint(
        self,
        *,
        system_prompt: str,
        messages: list[dict],
        max_tokens: int,
        max_retries: int,
    ):
        """Plain messages endpoint with 5xx-aware retry. Used by stepwise non-MCP steps."""
        from mozzarellm.pipeline.literature_mcp import (
            PER_CALL_TIMEOUT_S,
            RETRYABLE_API_EXCEPTIONS,
            _is_retryable_api_error,
        )

        client = anthropic.Anthropic(api_key=self.api_key)
        for attempt in range(max_retries):
            try:
                start = time.time()
                response = client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=messages,
                    timeout=PER_CALL_TIMEOUT_S,
                )
                return response, time.time() - start
            except RETRYABLE_API_EXCEPTIONS as e:
                if attempt < max_retries - 1 and _is_retryable_api_error(e):
                    backoff = min(30 * (2**attempt), 120)
                    time.sleep(backoff)
                    continue
                raise

    def _analyze_stepwise(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        mcp: bool,
        max_retries: int,
        max_tokens: int = 16000,
    ) -> tuple[dict | None, dict]:
        """Run the canonical CoT chain as N sequential, multi-turn API calls.

        Each step is a separate API call; prior assistant responses are appended
        to `messages` so step N sees steps 1..N-1's outputs. Steps in the MCP
        index set get PubMed tools attached; others use the plain endpoint.

        On any step failure, iteration aborts and a partial trace is returned
        with steps 1..N-1 preserved.
        """
        from mozzarellm.pipeline.literature_mcp import (
            _parse_json_from_text,
            _validate_literature_blocks,
            call_mcp,
        )
        from mozzarellm.utils.pricing import compute_cost
        from mozzarellm.utils.prompt_factory import compose_stepwise_user_turns
        from mozzarellm.utils.trace import extract_mcp_tool_calls

        # Per-turn user content + MCP routing decided by prompt_factory.
        turns = compose_stepwise_user_turns(mcp)

        messages: list[dict] = []
        step_records: list[dict] = []
        total_in = 0
        total_out = 0
        total_cost = 0.0
        total_elapsed = 0.0
        pricing_warnings: list[str] = []

        for i, turn in enumerate(turns):
            # Prepend the cluster bundle (per-cluster content) to the first turn only.
            user_content = f"{user_prompt}\n\n{turn['content']}" if i == 0 else turn["content"]
            messages.append({"role": "user", "content": user_content})
            use_mcp = turn["mcp"]

            try:
                if use_mcp:
                    response, elapsed = call_mcp(
                        system_prompt=system_prompt,
                        messages=messages,
                        model=self.model,
                        max_tokens=max_tokens,
                        max_retries=max_retries,
                        client=anthropic.Anthropic(api_key=self.api_key),
                    )
                else:
                    response, elapsed = self._call_messages_endpoint(
                        system_prompt=system_prompt,
                        messages=messages,
                        max_tokens=max_tokens,
                        max_retries=max_retries,
                    )
            except Exception as e:
                step_records.append(
                    {"step_index": i, "use_mcp": use_mcp, "error": f"{type(e).__name__}: {e}"}
                )
                raw_outputs = self._empty_raw_outputs()
                raw_outputs.update(
                    {
                        "input_tokens": total_in,
                        "output_tokens": total_out,
                        "elapsed_s": total_elapsed,
                        "cost_usd": total_cost,
                        "pricing_warning": "; ".join(pricing_warnings) or None,
                        "error": f"step {i} failed: {type(e).__name__}: {e}",
                        "steps": step_records,
                    }
                )
                return None, raw_outputs

            text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
            tool_calls = extract_mcp_tool_calls(response.content) if use_mcp else []
            in_tok = response.usage.input_tokens
            out_tok = response.usage.output_tokens
            cost, warning = compute_cost(self.model, in_tok, out_tok)
            if warning:
                pricing_warnings.append(warning)

            step_records.append(
                {
                    "step_index": i,
                    "use_mcp": use_mcp,
                    "assistant_text": text,
                    "tool_calls": tool_calls,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "elapsed_s": round(elapsed, 2),
                    "cost_usd": cost,
                    "error": None,
                }
            )

            total_in += in_tok
            total_out += out_tok
            total_cost += cost
            total_elapsed += elapsed

            # Append plain text as assistant turn; tool blocks were resolved server-side.
            messages.append({"role": "assistant", "content": text})

        final_text = step_records[-1]["assistant_text"] if step_records else ""
        parsed = _parse_json_from_text(final_text)

        raw_outputs = self._empty_raw_outputs()
        raw_outputs.update(
            {
                "response_text": final_text,
                "tool_calls": [tc for r in step_records for tc in r.get("tool_calls", [])],
                "input_tokens": total_in,
                "output_tokens": total_out,
                "elapsed_s": total_elapsed,
                "cost_usd": total_cost,
                "pricing_warning": "; ".join(pricing_warnings) or None,
                "steps": step_records,
            }
        )

        meta = {
            "mode": self._mode_tag("stepwise", mcp),
            "model": self.model,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "cost_usd": total_cost,
            "time_seconds": round(total_elapsed, 1),
            "tool_calls": sum(len(r.get("tool_calls", [])) for r in step_records),
            "n_steps": len(step_records),
        }
        if pricing_warnings:
            meta["pricing_warning"] = "; ".join(pricing_warnings)

        if not parsed:
            meta_err = {**meta, "error": "failed to parse JSON", "raw_output": final_text[:1000]}
            raw_outputs["error"] = "failed to parse JSON"
            return ({"_validation_metadata": meta_err}, raw_outputs)

        if mcp:
            schema_warnings = _validate_literature_blocks(parsed)
            if schema_warnings:
                meta["schema_warnings"] = schema_warnings
                raw_outputs["schema_warnings"] = schema_warnings

        parsed["_validation_metadata"] = meta
        return parsed, raw_outputs

    def _analyze_batch(
        self,
        *,
        system_prompt: str,
        cluster_to_prompt_map: dict[str, str],
        screen_name: str,
    ) -> tuple[None, dict]:
        """Submit + poll + retrieve a message batch. Per-cluster results write
        to disk; returns (None, raw_outputs) where raw_outputs carries batch_id
        and any errored custom_ids."""
        client = anthropic.Anthropic(api_key=self.api_key)
        request_list = self._make_list_of_cluster_request_objs(cluster_to_prompt_map, system_prompt)
        message_batch = client.messages.batches.create(requests=request_list)
        batch_id = message_batch.id

        intermediates = Path(f"output/{screen_name}_analysis/intermediates")
        intermediates.mkdir(parents=True, exist_ok=True)
        (intermediates / f"msg_batch_ID_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt").write_text(
            batch_id
        )
        print(f"Batch {batch_id} created. ID saved in {intermediates}/")

        while True:
            message_batch = client.messages.batches.retrieve(batch_id)
            if message_batch.processing_status == "ended":
                break
            print(f"Batch {batch_id} is still processing...")
            time.sleep(60)

        errored_requests: list[str] = []
        for result in client.messages.batches.results(batch_id):
            match result.result.type:
                case "succeeded":
                    path = Path(
                        f"output/{screen_name}_analysis/phase1_batch_cluster_LLM_analysis/{result.custom_id}_analysis_response.jsonl"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(str(result.result.message.content), encoding="utf-8")
                case "errored":
                    error_obj = result.result.error
                    error_type = getattr(getattr(error_obj, "error", None), "type", "unknown")
                    error_msg = getattr(getattr(error_obj, "error", None), "message", None)
                    print(f"{result.custom_id}: {error_type} -- {error_msg}")
                    errored_requests.append(result.custom_id)
                case "expired":
                    print(f"Request expired {result.custom_id}")
                    errored_requests.append(result.custom_id)

        if errored_requests:
            print(f"\n{len(errored_requests)} request(s) failed: {errored_requests}")

        raw_outputs = self._empty_raw_outputs()
        raw_outputs.update({"batch_id": batch_id, "errored_requests": errored_requests})
        return None, raw_outputs


# Error object for reference: ErrorResponse(error=InvalidRequestError(message='max_tokens: must be greater than or equal to 1',
# type='invalid_request_error', details={'error_visibility': 'user_facing'}), request_id=None, type='error')


class GeminiClient(LLMClientBase):
    """Google Gemini API provider"""

    def _get_env_var_name(self) -> str:
        return "GOOGLE_API_KEY"

    def _get_api_key_from_env(self) -> str | None:
        return os.environ.get(self._get_env_var_name())

    def _make_api_call(self, system_prompt: str, user_prompt: str) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)

        # Build config with optional parameters (no hardcoded values!)
        config_kwargs = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "system_instruction": system_prompt,
        }

        # Add optional sampling parameters
        if self.top_p is not None:
            config_kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            config_kwargs["top_k"] = self.top_k
        if self.stop_sequences:
            config_kwargs["stop_sequences"] = self.stop_sequences

        config = types.GenerateContentConfig(**config_kwargs)

        response = client.models.generate_content(
            model=self.model, contents=user_prompt, config=config
        )

        return response.text

    def _make_batch_api_call(
        self,
        cluster_to_prompt_map: dict[str, str],
        screen_name: str,
        system_prompt: str,
    ) -> list[str]:
        """
        Makes a batch of requests to the Google Gemini API.
        """
        pass  # TODO


def create_client(
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 16000,
    top_p: float | None = None,
    top_k: int | None = None,
    stop_sequences: list[str] | None = None,
    api_key: str | None = None,
    thinking: bool | None = None,
):
    """
    Factory function to create the appropriate client based on model name.

    Args:
        model: Model identifier
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter (0.0-1.0, optional)
        top_k: Top-K sampling parameter (optional, Claude/Gemini only)
        stop_sequences: List of stop sequences (optional)
        api_key: Optional API key (reads from env if None)
        thinking: Extended-thinking control (Claude only; None/False/True)

    Returns:
        Appropriate LLMProvider instance

    Raises:
        ValueError: If model prefix is not recognized
    """
    model_lower = model.lower()

    # OpenAI models
    if any(model_lower.startswith(prefix) for prefix in ["gpt", "o4", "o3", "o1"]):
        return OpenAIClient(model, temperature, max_tokens, top_p, top_k, stop_sequences, api_key)

    # Anthropic models
    elif model_lower.startswith("claude"):
        return AnthropicClient(
            model, temperature, max_tokens, top_p, top_k, stop_sequences, api_key, thinking
        )

    # Google models
    elif model_lower.startswith("gemini"):
        return GeminiClient(model, temperature, max_tokens, top_p, top_k, stop_sequences, api_key)

    else:
        raise ValueError(
            f"Unknown model prefix: {model}. "
            "Supported prefixes: gpt*, o4*, o3*, o1*, claude*, gemini*"
        )
