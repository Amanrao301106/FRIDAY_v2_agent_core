from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class ScreenTarget:
    """A target identified on screen."""
    label: str
    x: int
    y: int
    confidence: float = 0.0
    description: str = ""

@dataclass(frozen=True)
class ScreenObservation:
    """Structured observation of the current desktop."""
    width: int
    height: int
    targets: list[ScreenTarget] = field(default_factory=list)
    summary: str = ""
    raw: str = ""
    model: Optional[str] = None
