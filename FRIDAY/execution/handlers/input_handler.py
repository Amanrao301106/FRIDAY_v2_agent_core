"""
FRIDAY Voice Assistant — Input Handlers.

Handles keyboard and mouse actions:
- ``type_text`` — Type text via keyboard
- ``press_key`` — Press a single key
- ``hotkey``    — Press a key combination
- ``click``     — Click the mouse
- ``scroll``    — Scroll the mouse wheel

All actions delegate to ``pyautogui`` with proper error handling.
"""

from __future__ import annotations

from typing import Set

import pyautogui

from execution.handlers.base import ActionHandler, ActionResult
from intelligence.response_parser import ActionRequest
from utils.logger import get_logger

log = get_logger("execution.handlers.input")


class InputHandler(ActionHandler):
    """
    Handles keyboard and mouse input actions via ``pyautogui``.

    pyautogui's failsafe (move mouse to corner to abort) is left
    enabled for safety.
    """

    @property
    def handled_actions(self) -> Set[str]:
        return {"type_text", "press_key", "hotkey", "click", "scroll"}

    def execute(self, action: ActionRequest) -> ActionResult:
        try:
            if action.action == "type_text":
                return self._type_text(action)
            elif action.action == "press_key":
                return self._press_key(action)
            elif action.action == "hotkey":
                return self._hotkey(action)
            elif action.action == "click":
                return self._click(action)
            elif action.action == "scroll":
                return self._scroll(action)
            else:
                return ActionResult(
                    success=False,
                    error=f"Unexpected action: {action.action}",
                )
        except pyautogui.FailSafeException:
            log.warning("pyautogui failsafe triggered!")
            return ActionResult(
                success=False,
                error="Action aborted — mouse moved to screen corner (failsafe)",
            )

    # ── Individual Actions ──────────────────────────────────

    def _type_text(self, action: ActionRequest) -> ActionResult:
        text = action.value
        if not text:
            return ActionResult(success=False, error="No text provided")

        log.info("Typing text: '%s'", text[:60])
        pyautogui.write(text, interval=0.03)
        return ActionResult(success=True, message=f"Typed: {text[:40]}")

    def _press_key(self, action: ActionRequest) -> ActionResult:
        key = action.value.strip().lower()
        if not key:
            return ActionResult(success=False, error="No key specified")
        if key not in pyautogui.KEYBOARD_KEYS:
            return ActionResult(success=False, error=f"Unsupported key: {key}")

        log.info("Pressing key: %s", key)
        pyautogui.press(key)
        return ActionResult(success=True, message=f"Pressed {key}")

    def _hotkey(self, action: ActionRequest) -> ActionResult:
        keys = action.keys
        if not keys:
            return ActionResult(success=False, error="No keys specified for hotkey")
        invalid = [key for key in keys if key not in pyautogui.KEYBOARD_KEYS]
        if invalid:
            return ActionResult(
                success=False,
                error=f"Unsupported hotkey key(s): {', '.join(invalid)}",
            )

        log.info("Hotkey: %s", "+".join(keys))
        pyautogui.hotkey(*keys)
        return ActionResult(success=True, message=f"Hotkey {'+'.join(keys)}")

    def _click(self, action: ActionRequest) -> ActionResult:
        log.info("Clicking mouse")
        pyautogui.click()
        return ActionResult(success=True, message="Clicked")

    def _scroll(self, action: ActionRequest) -> ActionResult:
        try:
            amount = int(action.value) if action.value else -300
        except (ValueError, TypeError):
            amount = -300

        log.info("Scrolling: %d", amount)
        pyautogui.scroll(amount)
        return ActionResult(success=True, message=f"Scrolled {amount}")
