"""Conversation context for recent commands and app references."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Set

from config import FridayConfig
from utils.logger import get_logger

log = get_logger("intelligence.context")


@dataclass
class CommandRecord:
    """A single command history entry."""

    user_input: str
    actions_summary: List[str]
    success: bool
    app_opened: str = ""
    app_closed: str = ""


class ContextManager:
    """Track recent command history and app state for pronoun resolution."""

    def __init__(self, cfg: FridayConfig) -> None:
        self._window_size = cfg.context.window_size
        self._track_apps = cfg.context.track_open_apps
        self._history: Deque[CommandRecord] = deque(maxlen=self._window_size)
        self._open_apps: Set[str] = set()

        log.info(
            "ContextManager initialized (window=%d, track_apps=%s)",
            self._window_size,
            self._track_apps,
        )

    def record_command(
        self,
        user_input: str,
        actions_summary: List[str],
        success: bool,
        app_opened: str = "",
        app_closed: str = "",
    ) -> None:
        """Record a completed command and update tracked app state."""
        self._history.append(
            CommandRecord(
                user_input=user_input,
                actions_summary=actions_summary,
                success=success,
                app_opened=app_opened,
                app_closed=app_closed,
            )
        )

        if self._track_apps:
            if app_opened:
                self._open_apps.add(app_opened.lower())
                log.debug("App opened: %s (tracking)", app_opened)
            if app_closed:
                self._open_apps.discard(app_closed.lower())
                log.debug("App closed: %s (tracking)", app_closed)

    def get_history_strings(self) -> List[str]:
        """Return recent command history formatted for prompt context."""
        result: List[str] = []
        for record in self._history:
            status = "OK" if record.success else "FAILED"
            actions_str = ", ".join(record.actions_summary)
            result.append(f"User: '{record.user_input}' -> {actions_str} {status}")
        return result

    @property
    def last_app_opened(self) -> str:
        """Return the most recently opened app name, or an empty string."""
        for record in reversed(self._history):
            if record.app_opened:
                return record.app_opened
        return ""

    @property
    def last_app_closed(self) -> str:
        """Return the most recently closed app name, or an empty string."""
        for record in reversed(self._history):
            if record.app_closed:
                return record.app_closed
        return ""

    @property
    def open_apps(self) -> Set[str]:
        """Return a copy of currently tracked open apps."""
        return self._open_apps.copy()

    def clear(self) -> None:
        """Clear all tracked context."""
        self._history.clear()
        self._open_apps.clear()
        log.info("Context cleared")
