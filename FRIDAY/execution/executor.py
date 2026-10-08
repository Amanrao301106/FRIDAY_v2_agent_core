"""Action execution orchestration for FRIDAY."""

from __future__ import annotations

import time
from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from config import FridayConfig, resolve_app
from execution.handlers import HandlerRegistry
from execution.retry import RetryEngine
from intelligence.context import ContextManager
from intelligence.response_parser import ActionRequest, ParsedResponse
from security.audit import AuditLogger
from security.validator import SecurityValidator
from utils.logger import get_logger

if TYPE_CHECKING:
    from speech.tts import TextToSpeech

log = get_logger("execution.executor")


@dataclass(frozen=True)
class ExecutionReport:
    """Machine-readable result of executing an action chain."""

    success: bool
    message: str = ""
    error: Optional[str] = None
    actions_completed: int = 0


class ActionExecutor:
    """Execute parsed LLM responses through validation, handlers, and retry."""

    def __init__(
        self,
        cfg: FridayConfig,
        registry: HandlerRegistry,
        retry_engine: RetryEngine,
        validator: SecurityValidator,
        audit: AuditLogger,
        context: ContextManager,
        tts: "TextToSpeech",
    ) -> None:
        self._cfg = cfg
        self._registry = registry
        self._retry = retry_engine
        self._validator = validator
        self._audit = audit
        self._context = context
        self._tts = tts

    def execute_chain(self, response: ParsedResponse, user_input: str) -> ExecutionReport:
        """Execute all actions in a parsed response sequentially."""
        actions = response.actions
        meta = response.meta

        log.info(
            "Processing command: intent=%s, confidence=%.2f, optimized=%s, actions=%d",
            meta.intent,
            meta.confidence,
            meta.optimized,
            len(actions),
        )

        if meta.confidence < self._cfg.automation.min_confidence:
            if self._is_speak_only_response(response):
                clarification = response.spoken_response or actions[0].value
                self._tts.speak(clarification or "Please clarify your request.")
                log.info(
                    "Low-confidence clarification spoken: %.2f < %.2f",
                    meta.confidence,
                    self._cfg.automation.min_confidence,
                )
                self._audit.record(
                    action="chain",
                    params={"intent": meta.intent, "confidence": meta.confidence},
                    result="clarification_spoken",
                )
                return ExecutionReport(True, "Clarification spoken", actions_completed=0)

            self._tts.speak("Please clarify your request.")
            log.warning(
                "Command rejected for low confidence: %.2f < %.2f",
                meta.confidence,
                self._cfg.automation.min_confidence,
            )
            self._audit.record(
                action="chain",
                params={"intent": meta.intent, "confidence": meta.confidence},
                result="rejected_low_confidence",
            )
            return ExecutionReport(False, "Low-confidence command", "Please clarify your request.")

        chain_check = self._validator.check_chain_length(len(actions))
        if not chain_check.allowed:
            self._tts.speak(
                f"Sorry, that command has too many steps. {chain_check.reason}"
            )
            log.warning("Chain rejected: %s", chain_check.reason)
            return ExecutionReport(False, "Chain rejected", chain_check.reason)

        if response.requires_confirmation:
            desc = ", then ".join(a.description or a.action for a in actions)
            decision = self._tts.confirm(
                f"I'm about to: {desc}. Should I proceed? Say yes or no."
            )
            if not any(word in decision for word in ("yes", "yeah", "sure")):
                self._tts.speak("Okay, cancelled.")
                log.info("User cancelled confirmed action chain")
                self._audit.record(
                    action="chain",
                    params={"actions": [a.action for a in actions]},
                    result="cancelled_by_user",
                    user_decision="no",
                )
                return ExecutionReport(False, "Cancelled by user", "User cancelled")

        if response.spoken_response:
            self._tts.speak_async(response.spoken_response)

        all_success = True
        actions_summary: list[str] = []
        completed_count = 0
        app_opened = ""
        app_closed = ""

        for i, action in enumerate(actions):
            if action.value == "context_app":
                last_app = self._context.last_app_opened
                if last_app:
                    log.info("Resolved context_app to '%s'", last_app)
                    action.value = last_app
                else:
                    self._tts.speak("I don't know which app you mean.")
                    log.warning("context_app used but no app in context")
                    all_success = False
                    actions_summary.append(f"{action.action}(NO_CONTEXT)")
                    continue

            log.info("Executing action %d/%d: %s", i + 1, len(actions), action.action)

            rate_check = self._validator.check_rate_limit()
            if not rate_check.allowed:
                wait_time = self._cfg.security.command_cooldown
                log.debug("Rate limited; waiting %.1fs", wait_time)
                time.sleep(wait_time)

            if not self._validate_action(action):
                all_success = False
                actions_summary.append(f"{action.action}(BLOCKED)")
                continue

            handler = self._registry.get(action.action)
            if handler is None:
                log.error("No handler for action type: %s", action.action)
                self._tts.speak(f"I don't know how to do: {action.action}")
                all_success = False
                actions_summary.append(f"{action.action}(NO_HANDLER)")
                continue

            if self._cfg.automation.dry_run:
                log.info("Dry run: validated action %s", action.to_dict())
                self._audit.record(
                    action=action.action,
                    params=action.to_dict(),
                    result="dry_run",
                )
                actions_summary.append(
                    f"{action.action}({action.value or action.name})"
                )
                continue

            result = self._retry.execute_with_retry(action, handler.execute)
            actions_summary.append(f"{action.action}({action.value or action.name})")

            if result.app_opened:
                app_opened = result.app_opened
            if result.app_closed:
                app_closed = result.app_closed

            if not result.success:
                all_success = False
                log.error("Action chain stopped at step %d: %s", i + 1, result.error)
                break

            completed_count += 1

            if action.delay_after > 0 and i < len(actions) - 1:
                log.debug("Inter-action delay: %.1fs", action.delay_after)
                time.sleep(action.delay_after)

        self._context.record_command(
            user_input=user_input,
            actions_summary=actions_summary,
            success=all_success,
            app_opened=app_opened,
            app_closed=app_closed,
        )

        if all_success:
            log.info("Command chain completed successfully (%d actions)", len(actions))
            return ExecutionReport(True, "All actions completed", actions_completed=completed_count)

        log.warning("Command chain completed with errors")
        return ExecutionReport(
            False,
            "Action chain failed",
            "One or more actions failed",
            actions_completed=completed_count,
        )

    def _validate_action(self, action: ActionRequest) -> bool:
        """Run security checks appropriate for the action type."""
        if action.action in ("open_app", "close_app"):
            if not self._validate_app_action(action):
                return False

        if action.action == "open_website":
            url_check = self._validator.validate_url(action.value)
            if not url_check.allowed:
                log.warning("URL blocked: %s", url_check.reason)
                self._tts.speak(f"I can't open that URL - {url_check.reason}")
                self._audit.record(
                    action=action.action,
                    params=action.to_dict(),
                    result="blocked",
                    error=url_check.reason,
                )
                return False
            action.value = url_check.sanitized_value or action.value

        value_check = self._validator.validate_command_value(action.value)
        if not value_check.allowed:
            log.warning("Value blocked: %s", value_check.reason)
            self._audit.record(
                action=action.action,
                params=action.to_dict(),
                result="blocked",
                error=value_check.reason,
            )
            return False

        return True

    def _is_speak_only_response(self, response: ParsedResponse) -> bool:
        """Return True when a low-confidence response only asks the user something."""
        return bool(response.actions) and all(
            action.action == "speak" for action in response.actions
        )

    def _validate_app_action(self, action: ActionRequest) -> bool:
        """Validate app actions against the configured executable allowlist."""
        value = action.value.strip()
        entry = resolve_app(self._cfg, value.replace(".exe", ""))

        if entry:
            exe_name = "explorer.exe" if entry.is_uwp else Path(entry.executable).name
            if action.action == "close_app":
                exe_name = Path(entry.process_name).name
        else:
            exe_name = Path(value).name
            if action.action == "open_app" and not Path(exe_name).suffix:
                exe_name = f"{exe_name}.exe"

        val_check = self._validator.validate_command_value(value)
        exe_check = self._validator.validate_executable(exe_name)
        if val_check.allowed and exe_check.allowed:
            return True

        reason = val_check.reason or exe_check.reason or "Action is not allowed"
        log.warning("Action blocked by security: %s - %s", action.action, reason)
        self._tts.speak(f"I can't do that - {reason}")
        self._audit.record(
            action=action.action,
            params=action.to_dict(),
            result="blocked",
            error=reason,
        )
        return False
