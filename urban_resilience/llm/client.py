"""LLM client protocols, mock client for offline testing, and environment-configured provider adapter."""

from __future__ import annotations

from collections import deque
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMClient(Protocol):
    """Protocol defining the standard structured LLM completion interface."""

    def complete_json(
        self,
        messages: List[Dict[str, str]],
        schema: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Sends chat messages to an LLM provider and returns a structured JSON dictionary."""
        ...


def default_mock_generator(
    messages: List[Dict[str, str]], schema: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Provides a safe, domain-aware default mock response for ReAct subagents and Supervisor review."""
    user_msg = messages[-1].get("content", "") if messages else ""
    system_content = messages[0].get("content", "").lower() if messages else ""

    if "review:" in user_msg.lower():
        return {
            "action": "stop",
            "reasoning": "Evidence collection complete; no further delegation needed.",
            "tasks": [],
        }

    user_msg_lower = user_msg.lower()
    is_follow_up = (
        "targeted follow-up" in user_msg_lower
        or "follow_up_source" in user_msg_lower
        or "targeted inquiry" in user_msg_lower
        or "query_refined_evidence" in user_msg_lower
        or "refined source" in user_msg_lower
    )

    if is_follow_up:
        tool_calls_count = sum(
            1
            for m in messages
            if m.get("role") == "assistant" and "call tool" in m.get("content", "").lower()
        )
        if tool_calls_count == 0:
            import re
            m = re.search(r"region '([^']+)'", user_msg)
            t_region = m.group(1) if m else "SZ_FT_CBD"
            target_domain = "temporal"
            for d in ["temporal", "flexibility", "predictability", "spatial", "functional"]:
                if d in system_content:
                    target_domain = d
                    break
            return {
                "action": "call_tool",
                "reasoning": "Calling query_refined_evidence for targeted follow-up secondary window inquiry",
                "tool_call": {
                    "call_id": "c1",
                    "tool": "query_refined_evidence",
                    "args": {"region_id": t_region, "domain": target_domain},
                },
            }
        else:
            return {
                "action": "final",
                "reasoning": f"Refined follow-up analysis complete after {tool_calls_count} tool observation(s)",
                "feature_report": None,
            }

    # Domain-specific canonical tool execution chains
    domain_tools_map = {
        "temporal": ["compute_hourly_entropy", "compute_peak_concentration"],
        "flexibility": ["compute_peak_shift_elasticity", "compute_offpeak_adaptation"],
        "predictability": ["compute_autocorr_lags", "compute_rhythm_regularity"],
        "spatial": ["compute_destination_diversity", "compute_network_route_redundancy"],
        "functional": ["compute_poi_diversity_metrics", "compute_land_use_balance"],
    }

    target_tools = ["compute_hourly_entropy", "compute_peak_concentration"]
    first_line = messages[0].get("content", "").splitlines()[0].lower() if messages else ""
    matched = False
    for d_name, d_tools in domain_tools_map.items():
        if d_name in first_line:
            target_tools = d_tools
            matched = True
            break
    if not matched:
        for d_name, d_tools in domain_tools_map.items():
            if d_name in system_content:
                target_tools = d_tools
                break

    # Count how many tool calls have already been executed in this conversation
    tool_calls_count = sum(
        1
        for m in messages
        if m.get("role") == "assistant" and "call tool" in m.get("content", "").lower()
    )

    if tool_calls_count < len(target_tools):
        next_tool = target_tools[tool_calls_count]
        call_id = f"c{tool_calls_count + 1}"
        return {
            "action": "call_tool",
            "reasoning": f"Calling domain tool {next_tool} (step {tool_calls_count + 1}/{len(target_tools)})",
            "tool_call": {"call_id": call_id, "tool": next_tool, "args": {}},
        }

    return {
        "action": "final",
        "reasoning": f"Domain analysis completed after {tool_calls_count} tool observation(s)",
        "feature_report": None,
    }


class MockLLMClient:
    """Mock LLM client executing scripted responses for deterministic, offline testing."""

    def __init__(
        self,
        model: str = "mock-model-v1",
        default_latency: float = 0.01,
        tokens_per_call: int = 150,
        auto_fallback: bool = False,
    ):
        self.model = model
        self.default_latency = default_latency
        self.tokens_per_call = tokens_per_call
        self.auto_fallback = auto_fallback
        self._response_queue: deque[Any] = deque()
        self._generator: Optional[Callable[[List[Dict[str, str]], Optional[Dict[str, Any]]], Dict[str, Any]]] = None

        self.call_count: int = 0
        self.history: List[Dict[str, Any]] = []
        self.token_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.total_latency_seconds: float = 0.0

    def enqueue_response(self, response: Dict[str, Any] | str) -> None:
        """Appends a scripted response to the FIFO response queue."""
        self._response_queue.append(response)

    def set_response_generator(
        self,
        generator: Callable[[List[Dict[str, str]], Optional[Dict[str, Any]]], Dict[str, Any]],
    ) -> None:
        """Sets a dynamic generator callback to produce responses based on input messages."""
        self._generator = generator

    set_generator = set_response_generator

    def reset(self) -> None:
        """Resets queue, history, and usage statistics."""
        self._response_queue.clear()
        self._generator = None
        self.call_count = 0
        self.history.clear()
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.total_latency_seconds = 0.0

    def complete_json(
        self,
        messages: List[Dict[str, str]],
        schema: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Returns the next scripted or generated response dictionary."""
        start_time = time.perf_counter()
        active_model = model or self.model
        self.call_count += 1

        # Simulate approximate token usage
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        prompt_tokens = max(10, prompt_chars // 4)
        completion_tokens = self.tokens_per_call
        self.token_usage["prompt_tokens"] += prompt_tokens
        self.token_usage["completion_tokens"] += completion_tokens
        self.token_usage["total_tokens"] += (prompt_tokens + completion_tokens)

        # Retrieve response
        if self._generator is not None:
            raw = self._generator(messages, schema)
        elif self._response_queue:
            raw = self._response_queue.popleft()
        elif self.auto_fallback:
            raw = default_mock_generator(messages, schema)
        else:
            raise RuntimeError(
                f"MockLLMClient queue is empty! No scripted response available for call #{self.call_count}."
            )

        elapsed = time.perf_counter() - start_time + self.default_latency
        self.total_latency_seconds += elapsed

        # Handle raw string JSON or dictionary
        if isinstance(raw, str):
            try:
                result = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"MockLLMClient produced invalid JSON string: {exc}") from exc
        elif isinstance(raw, dict):
            result = dict(raw)
        else:
            raise TypeError(f"Mock response must be dict or JSON str, got {type(raw)}")

        self.history.append({
            "call_index": self.call_count,
            "model": active_model,
            "messages": messages,
            "schema": schema,
            "response": result,
            "latency_seconds": elapsed,
        })
        return result


class EnvironmentConfiguredLLMClient:
    """Production provider client reading credentials strictly from environment variables."""

    def __init__(
        self,
        api_key_env_var: str = "URBAN_LLM_API_KEY",
        base_url_env_var: str = "URBAN_LLM_BASE_URL",
        model_env_var: str = "URBAN_LLM_MODEL",
        default_model: str = "gpt-4o-mini",
    ):
        self.api_key_env_var = api_key_env_var
        self.base_url_env_var = base_url_env_var
        self.model_env_var = model_env_var
        self.default_model = default_model

        self.token_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.total_latency_seconds: float = 0.0

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env_var, "").strip()
        if not key:
            raise RuntimeError(
                f"LLM API key environment variable '{self.api_key_env_var}' is not configured. "
                f"Real LLM invocation is disabled. For testing, use MockLLMClient."
            )
        return key

    @property
    def base_url(self) -> str:
        return os.environ.get(self.base_url_env_var, "https://api.openai.com/v1").rstrip("/")

    @property
    def default_model_name(self) -> str:
        return os.environ.get(self.model_env_var, self.default_model)

    def complete_json(
        self,
        messages: List[Dict[str, str]],
        schema: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Issues an HTTP POST request to an OpenAI-compatible /chat/completions endpoint."""
        start_time = time.perf_counter()
        target_model = model or self.default_model_name
        url = f"{self.base_url}/chat/completions"

        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response_schema", "schema": schema},
            }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP Error {exc.code}: {err_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM Connection Error: {exc.reason}") from exc

        elapsed = time.perf_counter() - start_time
        self.total_latency_seconds += elapsed

        data = json.loads(raw_body)
        usage = data.get("usage", {})
        self.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        self.token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        self.token_usage["total_tokens"] += usage.get("total_tokens", 0)

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"LLM response contained no choices: {raw_body}")

        content_str = choices[0].get("message", {}).get("content", "{}")
        try:
            return json.loads(content_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse LLM completion content as JSON: {content_str}") from exc


def get_llm_client(
    mode: str = "mock",
    model: Optional[str] = None,
    api_key_env_var: str = "URBAN_LLM_API_KEY",
    base_url_env_var: str = "URBAN_LLM_BASE_URL",
    model_env_var: str = "URBAN_LLM_MODEL",
    auto_fallback: bool = True,
    **kwargs: Any,
) -> LLMClient:
    """Factory creating an LLMClient instance according to the requested mode."""
    mode_lower = (mode or "mock").lower()
    if mode_lower in ("mock", "test"):
        return MockLLMClient(model=model or "mock-model-v1", auto_fallback=auto_fallback, **kwargs)
    elif mode_lower in ("replay", "playback"):
        from urban_resilience.llm.replay import ReplayLLMClient
        replay_file = kwargs.pop("replay_file", None) or kwargs.pop("cache_file_path", None)
        if not replay_file:
            raise ValueError("Replay mode requires 'replay_file' path parameter.")
        return ReplayLLMClient(replay_file_path=replay_file, model=model or "mock-replay-v1", **kwargs)
    elif mode_lower in ("record", "caching", "cache"):
        from urban_resilience.llm.replay import CachingLLMClient
        underlying = kwargs.pop("underlying_client", None)
        cache_file = kwargs.pop("cache_file_path", None) or kwargs.pop("replay_file", None)
        return CachingLLMClient(underlying_client=underlying, cache_file_path=cache_file, **kwargs)
    elif mode_lower in ("env", "environment", "openai", "deepseek", "anthropic", "remote"):
        return EnvironmentConfiguredLLMClient(
            api_key_env_var=api_key_env_var,
            base_url_env_var=base_url_env_var,
            model_env_var=model_env_var,
            default_model=model or "gpt-4o-mini",
        )
    else:
        raise ValueError(f"Unknown LLM client mode: '{mode}'. Expected 'mock', 'env', 'replay', 'record'.")

