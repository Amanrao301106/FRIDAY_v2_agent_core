"""
FRIDAY Voice Assistant — Ollama LLM Client.

Wraps the ``ollama`` Python library with:

* Configurable model, host, and timeout
* LRU caching for repeated identical commands
* Retry logic for transient connection failures
* Performance timing for latency monitoring
* Abstract ``LLMBackend`` protocol for future cloud offloading

Usage::

    from intelligence.llm_client import OllamaClient
    from config import load_config

    cfg = load_config()
    client = OllamaClient(cfg)
    raw = client.query("open chrome")
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import List, Optional, Protocol

import ollama

from config import FridayConfig
from utils.logger import get_logger

log = get_logger("intelligence.llm_client")


# ── Abstract Backend (for future cloud offloading) ──────────

class LLMBackend(Protocol):
    """Protocol for LLM backends.  Implement this to add a cloud-based
    LLM (e.g. OpenAI, Anthropic) as an alternative to local Ollama."""

    def chat(
        self,
        system_prompt: str,
        user_message: str,
    ) -> str:
        """Send a chat request and return the raw response text."""
        ...


# ── LRU-Cached Query Function ──────────────────────────────

def _make_cached_query(cache_size: int):
    """
    Create a cached query function with the specified LRU cache size.

    We use a module-level factory because ``lru_cache`` doesn't work
    cleanly as a method decorator (``self`` pollutes the cache key).
    """

    @lru_cache(maxsize=cache_size)
    def _cached_query(
        model: str,
        host: str,
        timeout: int,
        system_prompt: str,
        user_message: str,
        temperature: float,
    ) -> str:
        """Execute an Ollama chat request (cached by all arguments)."""
        client = ollama.Client(host=host, timeout=timeout)
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            options={"temperature": temperature},
        )
        return response["message"]["content"]

    return _cached_query


# ── Ollama Client ───────────────────────────────────────────

class OllamaClient:
    """
    Ollama/Llama3 client with caching, retries, and timing.

    Args:
        cfg: FRIDAY configuration.
    """

    def __init__(self, cfg: FridayConfig) -> None:
        self._model = cfg.llm.model
        self._host = cfg.llm.host
        self._timeout = cfg.llm.timeout
        self._temperature = cfg.llm.temperature
        self._max_retries = 2

        self._cached_query = _make_cached_query(cfg.llm.cache_size)

        log.info(
            "OllamaClient initialized — model=%s, host=%s, cache=%d",
            self._model,
            self._host,
            cfg.llm.cache_size,
        )

    def query(
        self,
        system_prompt: str,
        user_message: str,
        use_cache: bool = True,
    ) -> str:
        """
        Send a query to the LLM and return the raw response.

        Args:
            system_prompt: The system prompt defining FRIDAY's behavior.
            user_message: The user's voice command.
            use_cache: Whether to use the LRU cache.  Set to False
                       for context-dependent queries where the same
                       text might need different responses.

        Returns:
            The raw LLM response string.

        Raises:
            ConnectionError: If Ollama is unreachable after retries.
            TimeoutError: If the request exceeds the configured timeout.
        """
        start = time.perf_counter()

        for attempt in range(1, self._max_retries + 1):
            try:
                if use_cache:
                    result = self._cached_query(
                        self._model,
                        self._host,
                        self._timeout,
                        system_prompt,
                        user_message,
                        self._temperature,
                    )
                else:
                    result = self._uncached_query(
                        system_prompt, user_message
                    )

                elapsed = time.perf_counter() - start
                log.info(
                    "LLM response received in %.2fs (attempt %d, cached=%s)",
                    elapsed,
                    attempt,
                    use_cache,
                )
                return result

            except Exception as exc:
                elapsed = time.perf_counter() - start
                if attempt < self._max_retries:
                    log.warning(
                        "LLM request failed (attempt %d/%.0fs): %s — retrying",
                        attempt,
                        elapsed,
                        exc,
                    )
                    time.sleep(1.0 * attempt)  # Simple linear backoff
                else:
                    log.error(
                        "LLM request failed after %d attempts (%.0fs): %s",
                        attempt,
                        elapsed,
                        exc,
                    )
                    raise ConnectionError(
                        f"Cannot reach Ollama at {self._host}: {exc}"
                    ) from exc

        # Unreachable, but satisfies type checker
        raise ConnectionError("LLM query failed")

    def clear_cache(self) -> None:
        """Clear the LRU cache (e.g. after config reload)."""
        self._cached_query.cache_clear()
        log.info("LLM cache cleared")

    def cache_info(self) -> str:
        """Return cache statistics as a human-readable string."""
        info = self._cached_query.cache_info()
        return (
            f"hits={info.hits}, misses={info.misses}, "
            f"size={info.currsize}/{info.maxsize}"
        )

    # ── Internal ────────────────────────────────────────────

    def _uncached_query(
        self, system_prompt: str, user_message: str
    ) -> str:
        """Execute an uncached Ollama chat request."""
        client = ollama.Client(host=self._host, timeout=self._timeout)
        response = client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            options={"temperature": self._temperature},
        )
        return response["message"]["content"]
