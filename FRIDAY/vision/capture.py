from __future__ import annotations
from pathlib import Path
from typing import Optional
import time

class ScreenCapture:
    """Capture the desktop without performing any input action."""

    def __init__(self, output_dir: str = "data/screenshots"):
        self.output_dir = Path(output_dir)

    def capture(self, filename: Optional[str] = None):
        try:
            import pyautogui
        except ImportError as exc:
            raise RuntimeError("PyAutoGUI is required for screen capture.") from exc

        self.output_dir.mkdir(parents=True, exist_ok=True)
        name = filename or f"screen_{int(time.time() * 1000)}.png"
        path = self.output_dir / name
        image = pyautogui.screenshot()
        image.save(path)
        return path
