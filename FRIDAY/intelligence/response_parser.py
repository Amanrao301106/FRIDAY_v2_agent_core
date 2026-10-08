"""
FRIDAY Voice Assistant — LLM Response Parser.

Extracts and validates JSON from raw LLM output.  Handles common
failure modes:

* Markdown code fences (```json ... ```)
* Preamble/postamble text around JSON
* Missing fields (applies defaults)
* Completely malformed responses

All parsed actions are returned as typed ``ActionRequest`` dataclass
instances.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("intelligence.response_parser")


# ── Data Types ──────────────────────────────────────────────

@dataclass
class ActionRequest:
    """A single validated action to execute."""
    action: str
    value: str = ""
    keys: List[str] = field(default_factory=list)
    name: str = ""
    message: str = ""
    delay_after: float = 0.0
    fail_strategy: str = "retry"  # retry | ask_user | fail_fast | skip
    description: str = ""
    fallback: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dict for audit logging."""
        d: Dict[str, Any] = {"action": self.action}
        if self.value:
            d["value"] = self.value
        if self.keys:
            d["keys"] = self.keys
        if self.name:
            d["name"] = self.name
        if self.message:
            d["message"] = self.message
        if self.fallback:
            d["fallback"] = self.fallback
        return d


@dataclass
class ResponseMeta:
    """Diagnostic metadata from the LLM."""
    intent: str = "unknown"
    confidence: float = 0.0
    plan_complexity: str = "low"
    requires_learning: bool = False
    learning_hint: str = ""
    policy_used: bool = False
    learning_mode: bool = False
    reward_prediction: float = 0.0
    optimized: bool = False


@dataclass
class ParsedResponse:
    """The fully parsed LLM response."""
    actions: List[ActionRequest]
    spoken_response: str = ""
    requires_confirmation: bool = False
    meta: ResponseMeta = field(default_factory=ResponseMeta)
    raw: str = ""  # Original LLM output for debugging


class ParseError(Exception):
    """Raised when the LLM response cannot be parsed into valid actions."""


# ── Valid Action Types ──────────────────────────────────────

VALID_ACTIONS = frozenset({
    "open_app",
    "close_app",
    "open_website",
    "type_text",
    "press_key",
    "hotkey",
    "click",
    "scroll",
    "send_whatsapp",
    "speak",
})

VALID_FAIL_STRATEGIES = frozenset({"retry", "ask_user", "fail_fast", "skip"})
VALID_PLAN_COMPLEXITIES = frozenset({"low", "medium", "high"})


def _as_float(value: Any, default: float = 0.0) -> float:
    """Convert parser metadata values to float without leaking ValueError."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_keys(value: Any) -> List[str]:
    """Normalize hotkey values into a list of key names."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(k).strip().lower() for k in value if str(k).strip()]
    if isinstance(value, str):
        parts = re.split(r"[,+\s]+", value)
        return [part.strip().lower() for part in parts if part.strip()]
    return [str(value).strip().lower()] if str(value).strip() else []


def _build_action(action_type: str, raw_action: Dict[str, Any]) -> ActionRequest:
    """Build an ActionRequest with normalized optional fields."""
    fail_strategy = str(raw_action.get("fail_strategy", "retry")).strip().lower()
    if fail_strategy not in VALID_FAIL_STRATEGIES:
        log.warning("Invalid fail_strategy '%s'; using retry", fail_strategy)
        fail_strategy = "retry"

    raw_fallback = raw_action.get("fallback")
    fallback = raw_fallback if isinstance(raw_fallback, dict) else None

    return ActionRequest(
        action=action_type,
        value=str(raw_action.get("value", "")),
        keys=_as_keys(raw_action.get("keys", [])),
        name=str(raw_action.get("name", "")),
        message=str(raw_action.get("message", "")),
        delay_after=max(0.0, _as_float(raw_action.get("delay_after", 0.0))),
        fail_strategy=fail_strategy,
        description=str(raw_action.get("description", "")),
        fallback=fallback,
    )


# ── JSON Extraction ─────────────────────────────────────────

def _extract_json(raw: str) -> str:
    """
    Extract a JSON string from potentially messy LLM output.

    Handles:
      1. Raw JSON (ideal case)
      2. Markdown fenced code blocks: ```json { ... } ```
      3. JSON embedded in prose text

    Args:
        raw: The raw LLM response string.

    Returns:
        The extracted JSON string.

    Raises:
        ParseError: If no valid JSON can be found.
    """
    text = raw.strip()

    # 1) Try raw JSON directly
    if text.startswith("{") or text.startswith("["):
        return text

    # 2) Try extracting from markdown code fences
    fence_match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL
    )
    if fence_match:
        return fence_match.group(1).strip()

    # 3) Try finding a JSON object anywhere in the text
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    raise ParseError(f"No JSON found in LLM response: {text[:200]}")


# ── Parser ──────────────────────────────────────────────────

def parse_response(raw: str) -> ParsedResponse:
    """
    Parse and validate a raw LLM response into a
    :class:`ParsedResponse`.

    Args:
        raw: The raw string returned by the LLM.

    Returns:
        A validated :class:`ParsedResponse` with one or more
        :class:`ActionRequest` objects.

    Raises:
        ParseError: If the response is malformed, missing required
                    fields, or contains invalid action types.
    """
    if not raw or not raw.strip():
        raise ParseError("Empty LLM response")

    # Extract JSON
    json_str = _extract_json(raw)

    # Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ParseError(
            f"Invalid JSON: {exc} — raw: {json_str[:200]}"
        ) from exc

    if not isinstance(data, dict):
        raise ParseError(f"Expected JSON object, got {type(data).__name__}")

    # ── Handle legacy single-action format ──────────────────
    # The old system returned {"action": "...", "value": "..."}
    # without an "actions" array.  Support this for compatibility.
    if "action" in data and "actions" not in data:
        log.debug("Legacy single-action format detected — converting")
        data = {
            "actions": [data],
            "spoken_response": data.get("spoken_response", ""),
            "meta": {
                "intent": str(data.get("action", "legacy_action")),
                "confidence": 1.0,
                "plan_complexity": "low",
                "requires_learning": False,
                "learning_hint": "",
                "policy_used": False,
                "learning_mode": False,
                "reward_prediction": 0.0,
                "optimized": False,
            },
        }

    # ── Parse actions array ─────────────────────────────────
    raw_actions = data.get("actions")
    if not raw_actions or not isinstance(raw_actions, list):
        raise ParseError("Missing or empty 'actions' array in response")

    actions: List[ActionRequest] = []
    for i, raw_action in enumerate(raw_actions):
        if not isinstance(raw_action, dict):
            log.warning("Action %d is not a dict — skipping", i)
            continue

        action_type = raw_action.get("action", "").strip().lower()
        if action_type not in VALID_ACTIONS:
            log.warning(
                "Unknown action type '%s' at index %d — skipping",
                action_type, i,
            )
            continue

        actions.append(_build_action(action_type, raw_action))

    if not actions:
        raise ParseError("No valid actions found in LLM response")

    # ── Parse meta (v3.0) ───────────────────────────────────
    raw_meta = data.get("meta", {})
    if isinstance(raw_meta, dict):
        plan_complexity = str(raw_meta.get("plan_complexity", "low")).lower()
        if plan_complexity not in VALID_PLAN_COMPLEXITIES:
            log.warning("Invalid plan_complexity '%s'; using low", plan_complexity)
            plan_complexity = "low"
        meta = ResponseMeta(
            intent=str(raw_meta.get("intent", "unknown")),
            confidence=_as_float(raw_meta.get("confidence", 0.0)),
            plan_complexity=plan_complexity,
            requires_learning=bool(raw_meta.get("requires_learning", False)),
            learning_hint=str(raw_meta.get("learning_hint", "")),
            policy_used=bool(raw_meta.get("policy_used", False)),
            learning_mode=bool(raw_meta.get("learning_mode", False)),
            reward_prediction=_as_float(raw_meta.get("reward_prediction", 0.0)),
            optimized=bool(raw_meta.get("optimized", False)),
        )
    else:
        meta = ResponseMeta()

    result = ParsedResponse(
        actions=actions,
        spoken_response=str(data.get("spoken_response", "")),
        requires_confirmation=bool(data.get("requires_confirmation", False)),
        meta=meta,
        raw=raw,
    )

    log.info(
        "Parsed %d action(s): %s [intent=%s, confidence=%.2f]",
        len(actions),
        [a.action for a in actions],
        meta.intent,
        meta.confidence,
    )
    return result
