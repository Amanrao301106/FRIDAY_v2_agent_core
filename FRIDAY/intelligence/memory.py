"""Persistent alias and preference memory for FRIDAY."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from config import FridayConfig, PROJECT_ROOT
from intelligence.response_parser import ActionRequest, ParsedResponse, ResponseMeta
from utils.logger import get_logger

log = get_logger("intelligence.memory")


class MemoryManager:
    """Small JSON-backed memory store for aliases and learned preferences."""

    def __init__(self, cfg: FridayConfig) -> None:
        self._enabled = cfg.memory.enabled
        raw_path = Path(cfg.memory.file)
        self._path = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path
        self._data: Dict[str, Any] = {"aliases": {}, "corrections": []}
        if self._enabled:
            self._load()

    @property
    def aliases(self) -> Dict[str, str]:
        """Return alias -> value mappings."""
        aliases = self._data.get("aliases", {})
        if not isinstance(aliases, dict):
            return {}
        result: Dict[str, str] = {}
        for key, payload in aliases.items():
            if isinstance(payload, dict) and isinstance(payload.get("value"), str):
                result[str(key)] = payload["value"]
        return result

    def alias_policy(self) -> Dict[str, Dict[str, Any]]:
        """Return alias policy entries including learned scores."""
        aliases = self._data.get("aliases", {})
        if not isinstance(aliases, dict):
            return {}
        result: Dict[str, Dict[str, Any]] = {}
        for key, payload in aliases.items():
            if isinstance(payload, dict) and isinstance(payload.get("value"), str):
                result[str(key)] = {
                    "value": payload["value"],
                    "score": float(payload.get("score", 1.0) or 0.0),
                    "frequency": int(payload.get("frequency", 0) or 0),
                }
        return result

    def prompt_lines(self) -> list[str]:
        """Return compact memory lines suitable for prompt injection."""
        lines: list[str] = []
        for alias, payload in sorted(self.alias_policy().items()):
            lines.append(
                f"{alias} -> {payload['value']} "
                f"(score={payload['score']:.2f}, frequency={payload['frequency']})"
            )
        return lines

    def learn_from_command(self, command: str) -> Optional[ParsedResponse]:
        """Learn alias mappings from explicit user preference commands."""
        if not self._enabled:
            return None

        learned = self._extract_learning(command)
        if not learned:
            return None

        alias, value = learned
        self.set_alias(alias, value)
        return ParsedResponse(
            actions=[
                ActionRequest(
                    action="speak",
                    value=f"Remembered {alias}.",
                )
            ],
            spoken_response="Remembered.",
            requires_confirmation=False,
            meta=ResponseMeta(
                intent="learn_alias",
                confidence=1.0,
                plan_complexity="low",
                requires_learning=False,
                learning_hint=f"mapped {alias} to {value}",
                policy_used=False,
                learning_mode=False,
                reward_prediction=0.8,
                optimized=True,
            ),
            raw="memory_learning",
        )

    def feedback_from_command(
        self,
        command: str,
        last_app: str = "",
    ) -> Optional[ParsedResponse]:
        """Detect simple feedback and update policy scores."""
        if not self._enabled:
            return None

        text = re.sub(r"\s+", " ", command.strip().lower())
        positive = {
            "good",
            "good job",
            "that worked",
            "correct",
            "right",
            "yes that is right",
            "perfect",
        }
        negative = {
            "wrong",
            "that was wrong",
            "incorrect",
            "not that",
            "bad",
            "no that is wrong",
        }

        if text in positive:
            self._record_feedback("positive", last_app)
            return self._feedback_response("Feedback saved.", 0.7)
        if text in negative:
            self._record_feedback("negative", last_app)
            return self._feedback_response("Feedback saved.", 0.2)
        return None

    def set_alias(self, alias: str, value: str) -> None:
        """Persist an alias mapping."""
        normalized_alias = _normalize_alias(alias)
        normalized_value = value.strip().lower()
        if not normalized_alias or not normalized_value:
            return

        aliases = self._data.setdefault("aliases", {})
        if not isinstance(aliases, dict):
            aliases = {}
            self._data["aliases"] = aliases

        current = aliases.get(normalized_alias, {})
        frequency = 0
        score = 1.0
        if isinstance(current, dict):
            frequency = int(current.get("frequency", 0) or 0)
            score = float(current.get("score", 1.0) or 0.0)

        aliases[normalized_alias] = {
            "value": normalized_value,
            "frequency": frequency + 1,
            "score": min(score + 0.5, 10.0),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save()
        log.info("Learned alias: %s -> %s", normalized_alias, normalized_value)

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._data.update(data)
        except Exception as exc:
            log.warning("Failed to load memory file %s: %s", self._path, exc)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def _record_feedback(self, kind: str, last_app: str) -> None:
        feedback = self._data.setdefault("feedback", [])
        if isinstance(feedback, list):
            feedback.append(
                {
                    "kind": kind,
                    "last_app": last_app,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        if last_app:
            delta = 0.25 if kind == "positive" else -0.25
            aliases = self._data.get("aliases", {})
            if isinstance(aliases, dict):
                for payload in aliases.values():
                    if not isinstance(payload, dict):
                        continue
                    if str(payload.get("value", "")).lower() == last_app.lower():
                        score = float(payload.get("score", 1.0) or 0.0)
                        payload["score"] = max(0.0, min(score + delta, 10.0))
                        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def _feedback_response(
        self,
        spoken: str,
        reward_prediction: float,
    ) -> ParsedResponse:
        return ParsedResponse(
            actions=[ActionRequest(action="speak", value=spoken)],
            spoken_response="Saved.",
            requires_confirmation=False,
            meta=ResponseMeta(
                intent="feedback",
                confidence=1.0,
                plan_complexity="low",
                requires_learning=False,
                learning_hint="",
                policy_used=True,
                learning_mode=False,
                reward_prediction=reward_prediction,
                optimized=True,
            ),
            raw="memory_feedback",
        )

    def _extract_learning(self, command: str) -> Optional[tuple[str, str]]:
        text = re.sub(r"\s+", " ", command.strip().lower())
        patterns = [
            r"^(?:remember\s+)?my\s+(.+?)\s+(?:is|means|=)\s+(.+)$",
            r"^(?:remember\s+)?(.+?)\s+(?:is|means|=)\s+(.+)$",
            r"^use\s+(.+?)\s+as\s+my\s+(.+)$",
        ]

        for index, pattern in enumerate(patterns):
            match = re.match(pattern, text)
            if not match:
                continue
            if index == 2:
                value, alias = match.groups()
            else:
                alias, value = match.groups()
            return _normalize_alias(alias), value.strip()
        return None


def _normalize_alias(alias: str) -> str:
    cleaned = re.sub(r"\s+", " ", alias.strip().lower())
    if cleaned.startswith("my "):
        cleaned = cleaned[3:].strip()
    return cleaned
