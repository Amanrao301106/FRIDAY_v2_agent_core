"""
FRIDAY Voice Assistant — Input Validation & Command Sanitization.

Provides a ``SecurityValidator`` that checks every action before
execution.  It enforces:

* Input length limits and control-character stripping
* Executable allowlisting (only configured apps can be launched/killed)
* URL scheme validation (http/https only)
* Blocked-pattern detection (prevents dangerous shell commands)
* Rate limiting between commands

All rejections are logged at WARNING level and produce a human-readable
reason string suitable for speaking back to the user.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlparse

from config import FridayConfig
from utils.logger import get_logger

log = get_logger("security.validator")


# ── Result Type ─────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Outcome of a security check."""
    allowed: bool
    reason: Optional[str] = None     # Human-readable rejection reason
    sanitized_value: Optional[str] = None  # Cleaned value (if applicable)


# ── Validator ───────────────────────────────────────────────

class SecurityValidator:
    """
    Stateful validator that checks actions against the security policy
    defined in :class:`FridayConfig`.

    Instantiate once and reuse across the lifetime of the application.
    """

    def __init__(self, cfg: FridayConfig) -> None:
        self._cfg = cfg
        self._sec = cfg.security
        self._last_command_time: float = 0.0

        # Pre-compile blocked patterns for performance
        self._blocked_re: List[re.Pattern[str]] = [
            re.compile(re.escape(p), re.IGNORECASE)
            for p in self._sec.blocked_patterns
        ]

        log.info(
            "SecurityValidator initialized — %d allowed executables, "
            "%d blocked patterns",
            len(self._sec.allowed_executables),
            len(self._blocked_re),
        )

    # ── Public API ──────────────────────────────────────────

    def sanitize_input(self, text: str) -> ValidationResult:
        """
        Sanitize user voice input *before* sending it to the LLM.

        Strips control characters, enforces length limits, and checks
        for blocked patterns.

        Args:
            text: Raw transcribed text from speech recognition.

        Returns:
            A :class:`ValidationResult` with ``sanitized_value`` set
            to the cleaned text on success.
        """
        if not text or not text.strip():
            return ValidationResult(allowed=False, reason="Empty input")

        # Strip control characters (keep printable + whitespace)
        cleaned = re.sub(r"[^\x20-\x7E\u00A0-\uFFFF]", "", text)

        # Enforce length limit
        if len(cleaned) > self._sec.max_input_length:
            log.warning("Input too long (%d chars), truncating", len(cleaned))
            cleaned = cleaned[: self._sec.max_input_length]

        # Check for blocked patterns in the input itself
        for pattern in self._blocked_re:
            if pattern.search(cleaned):
                reason = f"Input contains blocked pattern: {pattern.pattern}"
                log.warning(reason)
                return ValidationResult(allowed=False, reason=reason)

        return ValidationResult(
            allowed=True,
            sanitized_value=cleaned.strip(),
        )

    def validate_executable(self, exe_name: str) -> ValidationResult:
        """
        Check whether an executable is on the allowlist.

        Args:
            exe_name: The executable name (e.g. ``'chrome.exe'``).

        Returns:
            :class:`ValidationResult` indicating whether execution is
            permitted.
        """
        if not exe_name:
            return ValidationResult(
                allowed=False, reason="Empty executable name"
            )

        # Wildcard — allow anything
        if "*" in self._sec.allowed_executables:
            return ValidationResult(allowed=True, sanitized_value=exe_name)

        # Case-insensitive match
        if exe_name.lower() in [e.lower() for e in self._sec.allowed_executables]:
            return ValidationResult(allowed=True, sanitized_value=exe_name)

        reason = f"Executable '{exe_name}' is not on the allowlist"
        log.warning(reason)
        return ValidationResult(allowed=False, reason=reason)

    def validate_url(self, url: str) -> ValidationResult:
        """
        Ensure the URL uses an allowed scheme (http/https).

        Args:
            url: The URL to validate.

        Returns:
            :class:`ValidationResult`.
        """
        if not url or not url.strip():
            return ValidationResult(allowed=False, reason="Empty URL")

        cleaned = url.strip()
        if "://" not in cleaned:
            cleaned = f"https://{cleaned}"

        try:
            parsed = urlparse(cleaned)
        except Exception:
            return ValidationResult(
                allowed=False, reason=f"Malformed URL: {url}"
            )

        if parsed.scheme not in self._sec.allowed_url_schemes:
            reason = (
                f"URL scheme '{parsed.scheme}' not allowed "
                f"(allowed: {self._sec.allowed_url_schemes})"
            )
            log.warning(reason)
            return ValidationResult(allowed=False, reason=reason)

        if not parsed.netloc:
            return ValidationResult(
                allowed=False, reason=f"Malformed URL: {url}"
            )

        return ValidationResult(allowed=True, sanitized_value=cleaned)

    def validate_command_value(self, value: str) -> ValidationResult:
        """
        Generic check on any string value that will be used in a
        system command.  Rejects values containing blocked patterns.

        Args:
            value: The command parameter to validate.

        Returns:
            :class:`ValidationResult`.
        """
        if not value:
            return ValidationResult(allowed=True, sanitized_value="")

        for pattern in self._blocked_re:
            if pattern.search(value):
                reason = f"Value contains blocked pattern: {pattern.pattern}"
                log.warning(reason)
                return ValidationResult(allowed=False, reason=reason)

        return ValidationResult(allowed=True, sanitized_value=value)

    def check_chain_length(self, length: int) -> ValidationResult:
        """
        Ensure an action chain doesn't exceed the configured maximum.

        Args:
            length: Number of actions in the chain.

        Returns:
            :class:`ValidationResult`.
        """
        if length > self._sec.max_chain_length:
            reason = (
                f"Action chain too long ({length} actions, "
                f"max {self._sec.max_chain_length})"
            )
            log.warning(reason)
            return ValidationResult(allowed=False, reason=reason)
        return ValidationResult(allowed=True)

    def check_rate_limit(self) -> ValidationResult:
        """
        Enforce minimum delay between consecutive command executions.

        Returns:
            :class:`ValidationResult`.  The caller should respect the
            cooldown if ``allowed`` is False.
        """
        now = time.time()
        elapsed = now - self._last_command_time
        if elapsed < self._sec.command_cooldown:
            remaining = self._sec.command_cooldown - elapsed
            return ValidationResult(
                allowed=False,
                reason=f"Rate limited — wait {remaining:.1f}s",
            )
        self._last_command_time = now
        return ValidationResult(allowed=True)
