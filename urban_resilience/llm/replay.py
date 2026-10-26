"""Record and replay LLM client harnesses for deterministic experiment reproducibility.

Allows recording full multi-agent LLM conversations to JSONL files and replaying them
strictly offline with zero external network connectivity or non-deterministic variance.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from urban_resilience.llm.client import LLMClient, MockLLMClient


def compute_request_hash(
    messages: List[Dict[str, str]],
    schema: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    """Computes a deterministic SHA-256 hash for an LLM invocation request."""
    canonical = {
        "messages": messages,
        "schema": schema or {},
        "model": model or "",
        "temperature": round(float(temperature), 4),
    }
    dumped = json.dumps(canonical, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


class CachingLLMClient:
    """LLM client decorator that persists all request/response pairs to a JSONL log."""

    def __init__(
        self,
        underlying_client: Optional[LLMClient] = None,
        cache_file_path: Optional[Path | str] = None,
        random_seed: int = 42,
    ):
        self.underlying_client = underlying_client or MockLLMClient(auto_fallback=True)
        self.cache_file_path = Path(cache_file_path) if cache_file_path else None
        self.random_seed = random_seed
        self.call_count: int = 0
        self.recorded_count: int = 0
        self.model = getattr(self.underlying_client, "model", "mock-caching-v1")

        if self.cache_file_path:
            self.cache_file_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def token_usage(self) -> Dict[str, int]:
        return getattr(self.underlying_client, "token_usage", {"total_tokens": 0})

    @property
    def total_latency_seconds(self) -> float:
        return getattr(self.underlying_client, "total_latency_seconds", 0.0)

    def complete_json(
        self,
        messages: List[Dict[str, str]],
        schema: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Delegates completion to underlying client and records the pair."""
        resp = self.underlying_client.complete_json(
            messages=messages,
            schema=schema,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        self.call_count += 1

        req_hash = compute_request_hash(messages, schema, model, temperature)
        record = {
            "step": self.call_count,
            "request_hash": req_hash,
            "messages": messages,
            "schema": schema,
            "model": model or self.model,
            "temperature": temperature,
            "random_seed": self.random_seed,
            "response": resp,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if self.cache_file_path:
            with open(self.cache_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.recorded_count += 1

        return resp


class ReplayLLMClient:
    """Strict offline replay client retrieving pre-recorded responses from a JSONL file."""

    def __init__(
        self,
        replay_file_path: Optional[Path | str] = None,
        model: str = "mock-replay-v1",
        default_latency: float = 0.005,
        tokens_per_call: int = 150,
        cache_file_path: Optional[Path | str] = None,
    ):
        target_path = replay_file_path or cache_file_path
        if not target_path:
            raise ValueError("ReplayLLMClient requires either replay_file_path or cache_file_path.")
        self.replay_file_path = Path(target_path)
        self.model = model
        self.default_latency = default_latency
        self.tokens_per_call = tokens_per_call

        self.call_count: int = 0
        self.history: List[Dict[str, Any]] = []
        self.token_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.total_latency_seconds: float = 0.0

        # Load records into hash-based response queues
        self._records_by_hash: Dict[str, deque[Dict[str, Any]]] = {}
        self._load_records()

    def _load_records(self) -> None:
        if not self.replay_file_path.exists():
            raise FileNotFoundError(f"Replay cache file does not exist: {self.replay_file_path}")

        with open(self.replay_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                req_hash = record.get("request_hash")
                if not req_hash and "messages" in record:
                    req_hash = compute_request_hash(
                        record["messages"],
                        record.get("schema"),
                        record.get("model"),
                        record.get("temperature", 0.0),
                    )
                if req_hash not in self._records_by_hash:
                    self._records_by_hash[req_hash] = deque()
                self._records_by_hash[req_hash].append(record["response"])

    def complete_json(
        self,
        messages: List[Dict[str, str]],
        schema: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Retrieves matching pre-recorded response without making any network calls."""
        req_hash = compute_request_hash(messages, schema, model, temperature)
        q = self._records_by_hash.get(req_hash)

        if not q:
            raise RuntimeError(
                f"No recorded response found for prompt hash '{req_hash}' in replay cache {self.replay_file_path}. "
                f"Replay client is strictly offline and cannot synthesize novel responses."
            )

        resp = q.popleft()
        self.call_count += 1
        time.sleep(self.default_latency)
        self.total_latency_seconds += self.default_latency

        self.token_usage["prompt_tokens"] += self.tokens_per_call // 2
        self.token_usage["completion_tokens"] += self.tokens_per_call // 2
        self.token_usage["total_tokens"] += self.tokens_per_call

        self.history.append({
            "step": self.call_count,
            "request_hash": req_hash,
            "response": resp,
        })
        return resp
