from __future__ import annotations
import time
from .models import ScreenTarget

class SafeScreenController:
    """Mouse/keyboard actions with confidence and safety gates."""

    def __init__(self, min_confidence: float = 0.80, move_duration: float = 0.15):
        self.min_confidence = min_confidence
        self.move_duration = move_duration

    def click_target(self, target: ScreenTarget, *, confirm: bool = False) -> bool:
        if target.confidence < self.min_confidence:
            return False
        if confirm:
            return False  # Caller must obtain explicit human approval first.

        import pyautogui
        pyautogui.moveTo(target.x, target.y, duration=self.move_duration)
        pyautogui.click()
        return True

    def type_text(self, text: str, *, confirm: bool = False) -> bool:
        if confirm:
            return False
        import pyautogui
        pyautogui.write(text, interval=0.01)
        return True

    def press(self, key: str, *, confirm: bool = False) -> bool:
        if confirm:
            return False
        import pyautogui
        pyautogui.press(key)
        return True
