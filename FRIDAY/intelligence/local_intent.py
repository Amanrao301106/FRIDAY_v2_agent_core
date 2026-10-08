"""Deterministic parser for common desktop commands.

This module handles high-confidence commands locally so FRIDAY can respond
quickly and can still perform basic tasks when the LLM is unavailable.
"""

from __future__ import annotations

import re
from urllib.parse import quote_plus

from intelligence.response_parser import ActionRequest, ParsedResponse, ResponseMeta

_HOTKEYS: dict[str, list[str]] = {
    "copy": ["ctrl", "c"],
    "paste": ["ctrl", "v"],
    "cut": ["ctrl", "x"],
    "select all": ["ctrl", "a"],
    "save": ["ctrl", "s"],
    "undo": ["ctrl", "z"],
    "redo": ["ctrl", "y"],
    "new tab": ["ctrl", "t"],
    "close tab": ["ctrl", "w"],
    "refresh": ["ctrl", "r"],
}

_WEBSITE_ALIASES: dict[str, str] = {
    "google": "https://google.com",
    "youtube": "https://youtube.com",
    "gmail": "https://mail.google.com",
    "github": "https://github.com",
    "chatgpt": "https://chatgpt.com",
}


_AMBIGUOUS_ALIAS_WORDS = {"app", "editor", "browser", "terminal", "music app"}


def parse_local_command(
    command: str,
    aliases: dict[str, str] | None = None,
    policy: dict[str, dict] | None = None,
) -> ParsedResponse | None:
    """Return a parsed response for common commands, or None if no rule matches."""
    text = _normalize(command)
    aliases = aliases or {}
    policy = policy or {}
    if not text:
        return None

    for phrase, keys in _HOTKEYS.items():
        if text == phrase or text == f"press {phrase}":
            return _response(
                [ActionRequest(action="hotkey", keys=keys)],
                intent="hotkey",
                spoken="Done.",
            )

    if text in {"click", "left click"}:
        return _response([ActionRequest(action="click")], intent="click", spoken="Done.")

    if text.startswith("scroll "):
        return _parse_scroll(text)

    for prefix in ("type ", "write "):
        if text.startswith(prefix):
            value = command.strip()[len(prefix):].strip()
            if value:
                return _response(
                    [ActionRequest(action="type_text", value=value)],
                    intent="type_text",
                    spoken="Typing.",
                )

    if text.startswith("press "):
        key = text.removeprefix("press ").strip()
        if key and " " not in key:
            return _response(
                [ActionRequest(action="press_key", value=key)],
                intent="press_key",
                spoken="Done.",
            )

    close_match = re.match(r"^(?:close|quit|exit)\s+(.+)$", text)
    if close_match:
        target = close_match.group(1).strip()
        value = "context_app" if target in {"it", "that", "app"} else target
        return _response(
            [ActionRequest(action="close_app", value=value)],
            intent="close_app",
            spoken="Closing.",
        )

    open_match = re.match(r"^(?:open|launch|start)\s+(.+)$", text)
    if open_match:
        target = open_match.group(1).strip()
        alias_result = _resolve_or_clarify_alias(
            target,
            aliases,
            policy,
            intent="open_app",
        )
        if alias_result:
            return alias_result
        url = _website_url(target)
        if url:
            return _response(
                [ActionRequest(action="open_website", value=url)],
                intent="open_website",
                spoken="Opening.",
            )
        return _response(
            [ActionRequest(action="open_app", value=target)],
            intent="open_app",
            spoken="Opening.",
        )

    go_match = re.match(r"^(?:go to|open website|visit)\s+(.+)$", text)
    if go_match:
        target = go_match.group(1).strip()
        return _response(
            [ActionRequest(action="open_website", value=_website_url(target) or target)],
            intent="open_website",
            spoken="Opening.",
        )

    search_match = re.match(r"^(?:search|google)\s+(.+)$", text)
    if search_match:
        query = search_match.group(1).strip()
        if query:
            return _response(
                [
                    ActionRequest(
                        action="open_website",
                        value=f"https://www.google.com/search?q={quote_plus(query)}",
                    )
                ],
                intent="web_search",
                spoken="Searching.",
            )

    return None


def _normalize(command: str) -> str:
    return re.sub(r"\s+", " ", command.strip().lower())


def _website_url(target: str) -> str:
    cleaned = target.strip().lower()
    cleaned = re.sub(r"^(?:the\s+)?", "", cleaned)
    if cleaned in _WEBSITE_ALIASES:
        return _WEBSITE_ALIASES[cleaned]
    if "." in cleaned and " " not in cleaned:
        return cleaned if "://" in cleaned else f"https://{cleaned}"
    return ""


def _resolve_or_clarify_alias(
    target: str,
    aliases: dict[str, str],
    policy: dict[str, dict],
    intent: str,
) -> ParsedResponse | None:
    alias = _alias_key(target)
    if not alias:
        return None
    if alias in policy:
        payload = policy[alias]
        value = str(payload.get("value", aliases.get(alias, "")))
        score = float(payload.get("score", 1.0) or 0.0)
        return _response(
            [ActionRequest(action="open_app", value=value)],
            intent=intent,
            spoken="Opening.",
            confidence=0.95 if score >= 1.0 else 0.75,
            policy_used=True,
            reward_prediction=max(0.1, min(score / 5.0, 0.95)),
        )
    if alias in aliases:
        return _response(
            [ActionRequest(action="open_app", value=aliases[alias])],
            intent=intent,
            spoken="Opening.",
            confidence=0.9,
            policy_used=True,
            reward_prediction=0.75,
        )
    if target.startswith("my ") or alias in _AMBIGUOUS_ALIAS_WORDS:
        noun = alias.split()[-1] if alias else "app"
        question = f"Which {noun}?"
        return _response(
            [ActionRequest(action="speak", value=question)],
            intent=intent,
            spoken=question,
            confidence=0.4,
            requires_learning=True,
            learning_hint=f"map {alias} to preferred app",
            learning_mode=True,
            reward_prediction=0.3,
        )
    return None


def _alias_key(target: str) -> str:
    cleaned = _normalize(target)
    if cleaned.startswith("my "):
        return cleaned[3:].strip()
    if cleaned in _AMBIGUOUS_ALIAS_WORDS:
        return cleaned
    return ""


def _parse_scroll(text: str) -> ParsedResponse | None:
    direction = text.removeprefix("scroll ").strip()
    if direction in {"up", "upward", "upwards"}:
        amount = 500
    elif direction in {"down", "downward", "downwards"}:
        amount = -500
    else:
        try:
            amount = int(direction)
        except ValueError:
            return None
    return _response(
        [ActionRequest(action="scroll", value=str(amount))],
        intent="scroll",
        spoken="Done.",
    )


def _response(
    actions: list[ActionRequest],
    intent: str,
    spoken: str,
    confidence: float = 1.0,
    requires_learning: bool = False,
    learning_hint: str = "",
    policy_used: bool = False,
    learning_mode: bool = False,
    reward_prediction: float = 0.0,
) -> ParsedResponse:
    return ParsedResponse(
        actions=actions,
        spoken_response=spoken,
        requires_confirmation=False,
        meta=ResponseMeta(
            intent=intent,
            confidence=confidence,
            plan_complexity="low",
            requires_learning=requires_learning,
            learning_hint=learning_hint,
            policy_used=policy_used,
            learning_mode=learning_mode,
            reward_prediction=reward_prediction,
            optimized=True,
        ),
        raw="local_intent",
    )
