"""
FRIDAY Voice Assistant — Handler Registry.

Central registry that maps action types to handler instances.
New handlers are added here — no changes needed to the executor.

The registry resolves handlers in registration order.  Specialized
handlers (e.g. ``AppHandler``) are registered before generic ones
so they take priority.

Usage::

    from execution.handlers import HandlerRegistry
    registry = HandlerRegistry(cfg, tts)
    handler = registry.get("open_app")
    result = handler.execute(action)
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

from config import FridayConfig
from execution.handlers.base import ActionHandler
from execution.handlers.app_handler import AppHandler, WhatsAppHandler
from execution.handlers.generic_handler import GenericAppHandler
from execution.handlers.input_handler import InputHandler
from execution.handlers.web_handler import WebHandler
from execution.handlers.speak_handler import SpeakHandler
from utils.logger import get_logger

if TYPE_CHECKING:
    from speech.tts import TextToSpeech

log = get_logger("execution.handlers")


class HandlerRegistry:
    """
    Maps action types to handler instances.

    Handlers are checked in registration order — the first handler
    that declares support for an action type wins.

    Args:
        cfg: FRIDAY configuration.
        tts: The shared TTS engine (needed by SpeakHandler).
    """

    def __init__(self, cfg: FridayConfig, tts: "TextToSpeech") -> None:
        self._handlers: List[ActionHandler] = []
        self._cache: Dict[str, ActionHandler] = {}  # action_type → handler

        # ── Register handlers (order matters!) ──────────────
        # Specialized handlers first, generic fallback last.
        self._register(AppHandler(cfg))
        self._register(WhatsAppHandler(cfg))
        self._register(InputHandler())
        self._register(WebHandler())
        self._register(SpeakHandler(tts))
        self._register(GenericAppHandler())  # Fallback — must be last

        log.info(
            "HandlerRegistry initialized with %d handlers covering %d action types",
            len(self._handlers),
            len(self._cache),
        )

    def _register(self, handler: ActionHandler) -> None:
        """Register a handler.  First registration for an action type wins."""
        self._handlers.append(handler)
        for action_type in handler.handled_actions:
            if action_type not in self._cache:
                self._cache[action_type] = handler
                log.debug(
                    "Registered %s → %s",
                    action_type,
                    handler.__class__.__name__,
                )

    def get(self, action_type: str) -> Optional[ActionHandler]:
        """
        Look up the handler for a given action type.

        Args:
            action_type: The action type string (e.g. ``'open_app'``).

        Returns:
            The handler instance, or ``None`` if no handler is
            registered for this type.
        """
        return self._cache.get(action_type)

    def is_fail_fast(self, action_type: str) -> bool:
        """
        Check whether an action type is designated as fail-fast
        (no retry, escalate immediately).

        Args:
            action_type: The action type string.

        Returns:
            ``True`` if the action should never be retried.
        """
        handler = self.get(action_type)
        if handler:
            return action_type in handler.fail_fast_actions
        return False
