"""Task state models used by the FRIDAY agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from intelligence.response_parser import ActionRequest


class TaskStatus(str, Enum):
    PLANNING = "planning"
    RUNNING = "running"
    WAITING_CONFIRMATION = "waiting_confirmation"
    PAUSED = "paused"
    FAILED = "failed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class StepResult:
    """Outcome of one action in a task."""

    action: str
    success: bool
    message: str = ""
    error: str = ""


@dataclass
class AgentTask:
    """Mutable state for one user goal."""

    goal: str
    status: TaskStatus = TaskStatus.PLANNING
    actions: List[ActionRequest] = field(default_factory=list)
    current_index: int = 0
    completed_steps: List[StepResult] = field(default_factory=list)
    replans: int = 0
    last_error: str = ""
    max_steps: int = 10
    max_replans: int = 3

    @property
    def remaining_actions(self) -> List[ActionRequest]:
        return self.actions[self.current_index :]

    @property
    def is_finished(self) -> bool:
        return self.status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }

    def advance(self, result: StepResult) -> None:
        self.completed_steps.append(result)
        if result.success:
            self.current_index += 1
        else:
            self.last_error = result.error or result.message

    def mark_completed(self) -> None:
        self.status = TaskStatus.COMPLETED

    def mark_failed(self, reason: str) -> None:
        self.last_error = reason
        self.status = TaskStatus.FAILED

    def mark_cancelled(self) -> None:
        self.status = TaskStatus.CANCELLED
