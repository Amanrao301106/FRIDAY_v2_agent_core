"""
FRIDAY Voice Assistant — Speak Handler.

Handles the ``speak`` action by delegating to the TTS engine.
This handler receives a reference to the TTS instance at
construction time (dependency injection).
"""

from __future__ import annotations

from typing import Set, TYPE_CHECKING

from execution.handlers.base import ActionHandler, ActionResult
from intelligence.response_parser import ActionRequest
from utils.logger import get_logger

if TYPE_CHECKING:
    from speech.tts import TextToSpeech

log = get_logger("execution.handlers.speak")


class SpeakHandler(ActionHandler):
    """
    Speaks text to the user via the TTS engine.

    Args:
        tts: The shared TTS engine instance.
    """

    def __init__(self, tts: "TextToSpeech") -> None:
        self._tts = tts

    @property
    def handled_actions(self) -> Set[str]:
        return {"speak"}

    def execute(self, action: ActionRequest) -> ActionResult:
        text = action.value.strip()
        if not text:
            return ActionResult(success=False, error="No text to speak")

        log.info("Speaking: '%s'", text[:60])
        self._tts.speak(text)
        return ActionResult(success=True, message=f"Spoke: {text[:40]}")
