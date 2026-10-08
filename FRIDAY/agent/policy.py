"""Controlled-autonomy policy for FRIDAY actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Set

from config import FridayConfig
from intelligence.response_parser import ActionRequest


@dataclass(frozen=True)
class PolicyDecision:
    """Result of evaluating an action against the autonomy policy."""

    allowed: bool
    requires_confirmation: bool = False
    reason: str = ""


class ActionPolicy:
    """Decide whether an action can run automatically."""

    def __init__(self, cfg: FridayConfig) -> None:
        self._cfg = cfg
        self._confirmation_actions: Set[str] = {
            action.lower() for action in cfg.automation.confirmation_actions
        }
        self._blocked_actions: Set[str] = {
            action.lower() for action in cfg.automation.blocked_actions
        }

    def evaluate(self, action: ActionRequest) -> PolicyDecision:
        action_name = action.action.lower().strip()

        if action_name in self._blocked_actions:
            return PolicyDecision(False, reason=f"Action '{action_name}' is blocked by policy")

        if action_name in self._confirmation_actions:
            return PolicyDecision(
                True,
                requires_confirmation=True,
                reason=f"Action '{action_name}' requires confirmation",
            )

        if action.fail_strategy == "ask_user":
            return PolicyDecision(True, requires_confirmation=True, reason="Action requested user confirmation")

        return PolicyDecision(True)

    def requires_confirmation_for_any(self, actions: Iterable[ActionRequest]) -> bool:
        return any(self.evaluate(action).requires_confirmation for action in actions)
