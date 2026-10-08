from __future__ import annotations
from vision.capture import ScreenCapture
from vision.analyzer import ScreenAnalyzer
from vision.models import ScreenObservation

class FridayEyes:
    """High-level safe screen observation API."""

    def __init__(self, analyzer: ScreenAnalyzer, screenshot_dir: str = "data/screenshots"):
        self.capture = ScreenCapture(screenshot_dir)
        self.analyzer = analyzer

    def observe(self, instruction: str) -> ScreenObservation:
        path = self.capture.capture()
        return self.analyzer.analyze(str(path), instruction)
