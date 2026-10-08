"""Controlled autonomous execution loop for FRIDAY."""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from config import FridayConfig
from intelligence.response_parser import ParsedResponse
from agent.models import AgentTask, StepResult, TaskStatus
from agent.planner import TaskPlanner
from agent.policy import ActionPolicy
from utils.logger import get_logger

if TYPE_CHECKING:
    from execution.executor import ActionExecutor

log = get_logger("agent.orchestrator")


class AgentOrchestrator:
    """Plan, execute, verify, and re-plan a user goal."""

    def __init__(
        self,
        cfg: FridayConfig,
        planner: TaskPlanner,
        executor: ActionExecutor,
        policy: ActionPolicy,
        confirm: Callable[[str], str],
        speak: Callable[[str], None],
    ) -> None:
        self._cfg = cfg
        self._planner = planner
        self._executor = executor
        self._policy = policy
        self._confirm = confirm
        self._speak = speak

    def run(self, goal: str) -> AgentTask:
        task = AgentTask(
            goal=goal,
            max_steps=self._cfg.security.max_chain_length,
            max_replans=self._cfg.automation.max_replans,
        )
        task.status = TaskStatus.PLANNING

        try:
            planned = self._planner.plan(goal)
            task.actions = planned.actions
            task.status = TaskStatus.RUNNING

            while task.current_index < len(task.actions):
                if task.current_index >= task.max_steps:
                    task.mark_failed("Task exceeded maximum step limit")
                    break

                action = task.actions[task.current_index]
                decision = self._policy.evaluate(action)
                if not decision.allowed:
                    task.mark_failed(decision.reason)
                    self._speak(f"I can't do that. {decision.reason}")
                    break

                if decision.requires_confirmation:
                    task.status = TaskStatus.WAITING_CONFIRMATION
                    answer = self._confirm(
                        f"FRIDAY wants to {action.description or action.action}. Should I continue? Say yes or no."
                    )
                    if not self._is_yes(answer):
                        task.mark_cancelled()
                        self._speak("Okay, cancelled.")
                        break
                    task.status = TaskStatus.RUNNING

                single = ParsedResponse(
                    actions=[action],
                    spoken_response="",
                    requires_confirmation=False,
                )
                report = self._executor.execute_chain(single, goal)
                result = StepResult(
                    action=action.action,
                    success=report.success,
                    message=report.message,
                    error=report.error or "",
                )
                task.advance(result)

                if not report.success:
                    if task.replans >= task.max_replans:
                        task.mark_failed(report.error or "Action failed")
                        self._speak("I couldn't complete the task.")
                        break

                    task.replans += 1
                    task.status = TaskStatus.PLANNING
                    context = self._replan_context(task)
                    replacement = self._planner.plan(task.goal, context=context)
                    task.actions = task.actions[: task.current_index] + replacement.actions
                    task.status = TaskStatus.RUNNING
                    continue

            if not task.is_finished:
                task.mark_completed()
                self._speak("Task completed.")
        except Exception as exc:
            log.exception("Agent task failed")
            task.mark_failed(str(exc))
            self._speak("I couldn't complete that task.")

        return task

    @staticmethod
    def _is_yes(answer: str) -> bool:
        words = set(answer.lower().split())
        return bool(words & {"yes", "yeah", "yep", "sure", "okay", "ok", "continue"})

    @staticmethod
    def _replan_context(task: AgentTask) -> str:
        completed = "\n".join(
            f"- {item.action}: {'success' if item.success else 'failed'} {item.error or item.message}"
            for item in task.completed_steps
        ) or "- none"
        return (
            f"Goal: {task.goal}\n"
            f"Replan number: {task.replans}\n"
            f"Previous results:\n{completed}\n"
            f"Last error: {task.last_error or 'none'}"
        )
