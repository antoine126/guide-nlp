"""A small, typed client for a local Ollama server (chapters 32 to 47).

Every LLM number in the book was measured with Ollama, ``temperature=0`` and a
fixed seed. Answers can be cached on disk so that a script run twice gives the
same output without recomputing it.

Usage: start Ollama, pull the model (``ollama pull qwen2.5:7b-instruct-q4_K_M``),
then ``chat(...)`` from any chapter.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from guide_nlp.config import settings

Message = dict[str, Any]


class ChatResult(BaseModel):
    """What one call returns: the text, the token counts and the timings."""

    content: str
    thinking: str = ""
    prompt_tokens: int = 0
    eval_tokens: int = 0
    eval_s: float = 0.0
    total_s: float = 0.0
    wall_s: float = 0.0
    logprobs: list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None

    @property
    def tokens_per_second(self) -> float:
        return self.eval_tokens / self.eval_s if self.eval_s else 0.0


@dataclass
class Ollama:
    """``POST /api/chat`` with retries and an optional JSONL cache."""

    url: str = settings.ollama_url
    timeout: float = 1800.0
    cache_path: Path | None = None
    _cache: dict[str, ChatResult] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = httpx.Client(timeout=self.timeout)
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.open(encoding="utf-8"):
                entry = json.loads(line)
                self._cache[entry["key"]] = ChatResult.model_validate(entry["value"])

    def chat(
        self,
        model: str,
        messages: list[Message],
        *,
        fmt: str | dict[str, Any] | None = None,
        temperature: float = 0.0,
        seed: int = 0,
        max_tokens: int = 512,
        think: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
        options: dict[str, Any] | None = None,
    ) -> ChatResult:
        """One chat completion. ``fmt`` is ``"json"`` or a JSON schema."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "seed": seed,
                "num_predict": max_tokens,
                **(options or {}),
            },
        }
        if fmt is not None:
            payload["format"] = fmt
        if think is not None:
            payload["think"] = think
        if tools is not None:
            payload["tools"] = tools
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = top_logprobs
        key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        start = time.perf_counter()
        for attempt in range(4):  # a busy server answers 500 while it swaps models
            response = self._client.post(f"{self.url}/api/chat", json=payload)
            if response.status_code < 500:
                break
            time.sleep(20 * (attempt + 1))
        response.raise_for_status()
        data = response.json()
        message = data["message"]
        result = ChatResult(
            content=message.get("content", ""),
            thinking=message.get("thinking", "") or "",
            prompt_tokens=data.get("prompt_eval_count", 0),
            eval_tokens=data.get("eval_count", 0),
            eval_s=data.get("eval_duration", 0) / 1e9,
            total_s=data.get("total_duration", 0) / 1e9,
            wall_s=time.perf_counter() - start,
            logprobs=data.get("logprobs"),
            tool_calls=message.get("tool_calls"),
        )
        if self.cache_path is not None:
            self._cache[key] = result
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as sink:
                entry = {"key": key, "value": result.model_dump()}
                sink.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return result

    def loaded_memory_gb(self, model: str) -> float:
        """Memory used by a loaded model, as reported by ``/api/ps``."""
        for loaded in self._client.get(f"{self.url}/api/ps").json().get("models", []):
            if model in (loaded["name"], loaded["model"]):
                return loaded.get("size", 0) / 1e9
        return 0.0


_default: Ollama | None = None


def default_client() -> Ollama:
    """A shared client, cached in ``data/interim/llm_cache.jsonl``."""
    global _default
    if _default is None:
        _default = Ollama(cache_path=settings.interim_dir / "llm_cache.jsonl")
    return _default


def chat(messages: list[Message], model: str | None = None, **kwargs: Any) -> ChatResult:
    """Shortcut: ``chat([{"role": "user", "content": "..."}])``."""
    return default_client().chat(model or settings.llm_model, messages, **kwargs)
