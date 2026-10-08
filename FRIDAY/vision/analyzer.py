from __future__ import annotations
from .models import ScreenObservation, ScreenTarget
from .client import VisionBackend, parse_target_response

SCREEN_PROMPT = """Analyze this Windows desktop screenshot for a computer agent.
Return ONLY JSON:
{
  "summary": "short description",
  "targets": [
    {"label":"...", "x":123, "y":456, "confidence":0.0, "description":"..."}
  ]
}
Coordinates must be pixel coordinates in the supplied screenshot.
Only include visible, actionable targets relevant to the instruction.
Never invent a target. If uncertain, use low confidence."""

class ScreenAnalyzer:
    def __init__(self, backend: VisionBackend):
        self.backend = backend

    def analyze(self, image_path: str, instruction: str) -> ScreenObservation:
        raw = self.backend.analyze(image_path, SCREEN_PROMPT + "\nInstruction: " + instruction)
        data = parse_target_response(raw)
        targets = []
        for item in data.get("targets", []):
            try:
                targets.append(ScreenTarget(
                    label=str(item["label"]),
                    x=int(item["x"]),
                    y=int(item["y"]),
                    confidence=float(item.get("confidence", 0.0)),
                    description=str(item.get("description", "")),
                ))
            except (KeyError, TypeError, ValueError):
                continue

        try:
            from PIL import Image
            width, height = Image.open(image_path).size
        except Exception:
            width = height = 0

        return ScreenObservation(
            width=width,
            height=height,
            targets=targets,
            summary=str(data.get("summary", "")),
            raw=raw,
            model=getattr(self.backend, "model", None),
        )
