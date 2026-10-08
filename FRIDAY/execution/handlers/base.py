"""
FRIDAY Voice Assistant — Abstract Action Handler.

Defines the :class:`ActionHandler` base class and the
:class:`ActionResult` dataclass that all handlers must use.

New capabilities are added by subclassing ``ActionHandler`` and
registering the subclass in the handler registry
(``execution/handlers/__init__.py``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Set

from intelligence.response_parser import ActionRequest


@dataclass
class ActionResult:
    """Outcome of executing a single action."""
    success: bool
    message: str = ""           # Human-readable result description
    error: Optional[str] = None # Error details (if failed)
    app_opened: str = ""        # App name opened (for context tracking)
    app_closed: str = ""        # App name closed (for context tracking)


class ActionHandler(ABC):
    """
    Abstract base class for action handlers.

    Each handler declares which action types it can handle and
    provides an ``execute`` method.  Handlers are stateless — all
    context comes from the ``ActionRequest``.

    Subclass checklist:

    1. Override :meth:`handled_actions` to return the set of action
       type strings this handler supports.
    2. Override :meth:`execute` to perform the action.
    3. Optionally override :attr:`fail_fast_actions` to declare
       which of your actions should never be retried.
    """

    @property
    @abstractmethod
    def handled_actions(self) -> Set[str]:
        """Return the set of action type strings this handler supports."""
        ...

    @abstractmethod
    def execute(self, action: ActionRequest) -> ActionResult:
        """
        Execute a single action.

        Args:
            action: The validated action request.

        Returns:
            An :class:`ActionResult` describing the outcome.
        """
        ...

    @property
    def fail_fast_actions(self) -> Set[str]:
        """
        Return action types that should escalate immediately on
        failure (no retry).  Override in subclasses as needed.

        Returns:
            A set of action type strings.  Default: empty set.
        """
        return set()

    def can_handle(self, action_type: str) -> bool:
        """Check if this handler supports the given action type."""
        return action_type in self.handled_actions
