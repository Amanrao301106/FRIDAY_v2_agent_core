"""
FRIDAY Voice Assistant — Generic App Handler (Fallback).

Handles ``open_app`` and ``close_app`` for applications that are
NOT in the config registry.  Uses Windows ``start`` command for
opening and ``taskkill`` for closing.

This handler is only reached when :class:`AppHandler` cannot
resolve the app name to a registered entry.  It is registered
with lower priority in the handler registry.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Set

from execution.handlers.base import ActionHandler, ActionResult
from intelligence.response_parser import ActionRequest
from utils.logger import get_logger

log = get_logger("execution.handlers.generic")


class GenericAppHandler(ActionHandler):
    """
    Fallback handler for unregistered applications.

    Uses ``subprocess.Popen`` with argument lists (no shell
    interpolation) for safety.
    """

    @property
    def handled_actions(self) -> Set[str]:
        return {"open_app", "close_app"}

    def execute(self, action: ActionRequest) -> ActionResult:
        if action.action == "open_app":
            return self._open(action)
        elif action.action == "close_app":
            return self._close(action)
        return ActionResult(
            success=False, error=f"Unexpected action: {action.action}"
        )

    def _open(self, action: ActionRequest) -> ActionResult:
        app = action.value.strip()
        if not app:
            return ActionResult(success=False, error="No app name provided")

        log.info("Opening app via generic handler: %s", app)
        app_target = app if Path(app).suffix else f"{app}.exe"
        try:
            subprocess.Popen(
                ["cmd", "/c", "start", "", app_target],
                shell=False,
            )
            return ActionResult(
                success=True,
                message=f"Opened {app_target} (generic)",
                app_opened=app,
            )
        except Exception as exc:
            return ActionResult(
                success=False,
                error=f"Failed to open {app}: {exc}",
            )

    def _close(self, action: ActionRequest) -> ActionResult:
        process = action.value.strip()
        if not process:
            return ActionResult(
                success=False, error="No process name provided"
            )

        log.info("Closing app via generic handler: %s", process)
        try:
            result = subprocess.run(
                ["taskkill", "/f", "/im", process],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return ActionResult(
                    success=True,
                    message=f"Closed {process} (generic)",
                    app_closed=process.replace(".exe", "").lower(),
                )
            else:
                return ActionResult(
                    success=False,
                    error=f"Could not close {process}: {result.stderr.strip()}",
                )
        except subprocess.TimeoutExpired:
            return ActionResult(
                success=False, error=f"Timeout closing {process}"
            )
        except Exception as exc:
            return ActionResult(
                success=False, error=f"Failed to close {process}: {exc}"
            )
