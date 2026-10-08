"""
FRIDAY Voice Assistant — Retry Engine.

Implements the failure recovery strategy:

1. **Automatic remediation**: Retry with exponential backoff
   (up to ``max_attempts`` total tries).
2. **Selective retry**: Transient failures get silent retries.
3. **Fail-fast**: Irreversible actions (e.g. ``send_whatsapp``)
   escalate immediately to the user.
4. **User decision point**: After retries are exhausted, ask the
   user to continue, skip, or abort.

All retry attempts, failures, and user decisions are recorded in
the audit log.
"""

from __future__ import annotations

import time
from typing import Callable, TYPE_CHECKING

from config import RetryConfig
from execution.handlers.base import ActionResult
from intelligence.response_parser import ActionRequest
from security.audit import AuditLogger
from utils.logger import get_logger

if TYPE_CHECKING:
    from speech.tts import TextToSpeech
    from execution.handlers import HandlerRegistry

log = get_logger("execution.retry")


class RetryEngine:
    """
    Wraps action execution with retry logic and user decision points.

    Args:
        cfg: Retry configuration.
        tts: TTS engine for speaking to the user at decision points.
        registry: Handler registry to check fail-fast status.
        audit: Audit logger for recording retry attempts.
    """

    def __init__(
        self,
        cfg: RetryConfig,
        tts: "TextToSpeech",
        registry: "HandlerRegistry",
        audit: AuditLogger,
    ) -> None:
        self._max_attempts = cfg.max_attempts
        self._backoff_base = cfg.backoff_base
        self._backoff_max = cfg.backoff_max
        self._fail_fast_actions = set(cfg.fail_fast_actions)
        self._tts = tts
        self._registry = registry
        self._audit = audit

    def execute_with_retry(
        self,
        action: ActionRequest,
        execute_fn: Callable[[ActionRequest], ActionResult],
    ) -> ActionResult:
        """
        Execute an action with retry logic and failure recovery.

        The strategy depends on the action's ``fail_strategy`` field
        and whether the handler marks it as fail-fast:

        * ``fail_fast`` or handler-declared fail-fast → execute once,
          escalate on failure.
        * ``skip`` → execute once, return success even if it fails.
        * ``ask_user`` → execute once, then ask user on failure.
        * ``retry`` (default) → retry with exponential backoff, then
          ask user if all retries fail.

        Args:
            action: The action to execute.
            execute_fn: The callable that actually executes the action
                        (typically ``handler.execute``).

        Returns:
            The final :class:`ActionResult`.
        """
        action_desc = action.description or f"{action.action}({action.value})"
        is_fail_fast = (
            action.fail_strategy == "fail_fast"
            or action.action in self._fail_fast_actions
            or self._registry.is_fail_fast(action.action)
        )

        # ── Fail-fast path ──────────────────────────────────
        if is_fail_fast:
            log.info("Fail-fast action: %s — no retry", action_desc)
            result = execute_fn(action)
            self._audit.record(
                action=action.action,
                params=action.to_dict(),
                result="success" if result.success else "failure",
                attempt=1,
                error=result.error,
            )
            if not result.success:
                self._tts.speak(
                    f"Action failed: {action_desc}. {result.error or ''}"
                )
            return result

        # ── Skip path ───────────────────────────────────────
        if action.fail_strategy == "skip":
            log.info("Skip-on-fail action: %s", action_desc)
            result = execute_fn(action)
            self._audit.record(
                action=action.action,
                params=action.to_dict(),
                result="success" if result.success else "skipped",
                attempt=1,
                error=result.error,
            )
            if not result.success:
                log.warning("Skipping failed action: %s", action_desc)
                result.success = True  # Mark as "success" to continue chain
                result.message = f"Skipped: {action_desc}"
            return result

        # ── Retry path (default) ────────────────────────────
        last_result: ActionResult = ActionResult(success=False)
        for attempt in range(1, self._max_attempts + 1):
            result = execute_fn(action)

            if result.success:
                self._audit.record(
                    action=action.action,
                    params=action.to_dict(),
                    result="success",
                    attempt=attempt,
                )
                return result

            last_result = result
            log.warning(
                "Action failed (attempt %d/%d): %s — %s",
                attempt,
                self._max_attempts,
                action_desc,
                result.error,
            )

            self._audit.record(
                action=action.action,
                params=action.to_dict(),
                result="failure",
                attempt=attempt,
                error=result.error,
            )

            # Don't sleep after the last attempt
            if attempt < self._max_attempts:
                delay = min(
                    self._backoff_base * (2 ** (attempt - 1)),
                    self._backoff_max,
                )
                log.info("Retrying in %.1fs…", delay)
                time.sleep(delay)

        # ── All retries exhausted → user decision point ─────
        return self._ask_user_decision(action, last_result, action_desc)

    def _ask_user_decision(
        self,
        action: ActionRequest,
        last_result: ActionResult,
        action_desc: str,
    ) -> ActionResult:
        """
        Ask the user how to proceed after all retries are exhausted.

        Options:
        * **continue** → return the failure but let the chain continue
        * **skip** → mark as success and continue
        * **abort** → return failure to stop the chain

        Returns:
            An :class:`ActionResult` reflecting the user's choice.
        """
        prompt = (
            f"Step failed after retries: {action_desc}. "
            f"Error: {last_result.error or 'unknown'}. "
            "Say continue, skip, or abort."
        )

        decision = self._tts.confirm(
            prompt, options=["continue", "skip", "abort"]
        )

        self._audit.record(
            action=action.action,
            params=action.to_dict(),
            result="user_decision",
            error=last_result.error,
            user_decision=decision,
        )

        if "skip" in decision:
            log.info("User chose to skip: %s", action_desc)
            return ActionResult(
                success=True, message=f"Skipped by user: {action_desc}"
            )
        elif "continue" in decision:
            log.info("User chose to continue despite failure: %s", action_desc)
            return ActionResult(
                success=True,
                message=f"Continuing despite failure: {action_desc}",
            )
        else:
            # Default to abort
            log.info("User chose to abort: %s", action_desc)
            return ActionResult(
                success=False,
                error=f"Aborted by user: {action_desc}",
            )
