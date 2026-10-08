"""
FRIDAY Voice Assistant — Application Handlers.

Handles ``open_app`` and ``close_app`` for the five hardcoded apps:
Chrome, VS Code, Notepad, Calculator, WhatsApp.

Uses ``subprocess.run()`` with argument lists (not string interpolation)
to prevent shell injection attacks.  WhatsApp gets special handling
as a UWP app launched via ``explorer shell:AppsFolder\\...``.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Set

import pyautogui

from config import FridayConfig, resolve_app
from execution.handlers.base import ActionHandler, ActionResult
from intelligence.response_parser import ActionRequest
from utils.logger import get_logger

log = get_logger("execution.handlers.app")


class AppHandler(ActionHandler):
    """
    Handles ``open_app`` and ``close_app`` for registered applications.

    Falls through to the generic handler if the app is not in the
    config registry.

    Args:
        cfg: FRIDAY configuration.
    """

    def __init__(self, cfg: FridayConfig) -> None:
        self._cfg = cfg

    @property
    def handled_actions(self) -> Set[str]:
        return {"open_app", "close_app"}

    def execute(self, action: ActionRequest) -> ActionResult:
        if action.action == "open_app":
            return self._open(action)
        elif action.action == "close_app":
            return self._close(action)
        return ActionResult(success=False, error=f"Unexpected action: {action.action}")

    # ── Open App ────────────────────────────────────────────

    def _open(self, action: ActionRequest) -> ActionResult:
        """Open an application by name."""
        app_name = action.value.strip().lower()
        entry = resolve_app(self._cfg, app_name)

        if entry is None:
            # No registered app — fall through to generic handler
            return self._generic_open(app_name)

        log.info("Opening registered app: %s → %s", app_name, entry.executable)

        try:
            if entry.is_uwp:
                # UWP apps need the explorer shell:AppsFolder syntax
                target = entry.executable
                if target.lower().startswith("explorer "):
                    target = target.split(" ", 1)[1]
                subprocess.Popen(
                    ["explorer", target],
                    shell=False,
                )
            else:
                subprocess.Popen(
                    ["cmd", "/c", "start", "", entry.executable],
                    shell=False,
                )

            # Wait for the app to start
            if entry.launch_delay > 0:
                log.debug("Waiting %.1fs for app startup", entry.launch_delay)
                time.sleep(entry.launch_delay)

            return ActionResult(
                success=True,
                message=f"Opened {app_name}",
                app_opened=app_name,
            )

        except Exception as exc:
            log.error("Failed to open %s: %s", app_name, exc)
            return ActionResult(
                success=False,
                error=f"Failed to open {app_name}: {exc}",
            )

    def _generic_open(self, app_name: str) -> ActionResult:
        """Open an unregistered app using Windows ``start`` command."""
        log.info("Opening unregistered app via generic handler: %s", app_name)
        app_target = app_name if Path(app_name).suffix else f"{app_name}.exe"
        try:
            subprocess.Popen(
                ["cmd", "/c", "start", "", app_target],
                shell=False,
            )
            return ActionResult(
                success=True,
                message=f"Opened {app_target} (generic)",
                app_opened=app_name,
            )
        except Exception as exc:
            log.error("Generic open failed for %s: %s", app_name, exc)
            return ActionResult(
                success=False,
                error=f"Failed to open {app_name}: {exc}",
            )

    # ── Close App ───────────────────────────────────────────

    def _close(self, action: ActionRequest) -> ActionResult:
        """Close an application by process name."""
        process_name = action.value.strip()

        if not process_name:
            return ActionResult(
                success=False, error="No process name provided for close_app"
            )

        # Try to resolve to a registered app's process name
        entry = resolve_app(self._cfg, process_name.replace(".exe", ""))
        if entry:
            process_name = entry.process_name

        log.info("Closing app: %s", process_name)

        try:
            result = subprocess.run(
                ["taskkill", "/f", "/im", process_name],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                return ActionResult(
                    success=True,
                    message=f"Closed {process_name}",
                    app_closed=process_name.replace(".exe", "").lower(),
                )
            else:
                # taskkill returns non-zero if process not found
                stderr = result.stderr.strip()
                log.warning("taskkill returned %d: %s", result.returncode, stderr)
                return ActionResult(
                    success=False,
                    error=f"Could not close {process_name}: {stderr}",
                )

        except subprocess.TimeoutExpired:
            return ActionResult(
                success=False,
                error=f"Timeout closing {process_name}",
            )
        except Exception as exc:
            log.error("Failed to close %s: %s", process_name, exc)
            return ActionResult(
                success=False,
                error=f"Failed to close {process_name}: {exc}",
            )


class WhatsAppHandler(ActionHandler):
    """
    Handles ``send_whatsapp`` — opens WhatsApp, searches for a
    contact, and sends a message using ``pyautogui`` UI automation.

    This handler is inherently fragile because it relies on fixed
    UI timing.  Delays are configurable via ``config.yaml``.
    """

    def __init__(self, cfg: FridayConfig) -> None:
        self._cfg = cfg
        self._entry = resolve_app(cfg, "whatsapp")

    @property
    def handled_actions(self) -> Set[str]:
        return {"send_whatsapp"}

    @property
    def fail_fast_actions(self) -> Set[str]:
        # Messages are irreversible — never retry silently
        return {"send_whatsapp"}

    def execute(self, action: ActionRequest) -> ActionResult:
        name = action.name.strip()
        message = action.message.strip()

        if not name or not message:
            return ActionResult(
                success=False,
                error="send_whatsapp requires both 'name' and 'message'",
            )

        log.info("Sending WhatsApp message to '%s': '%s'", name, message[:50])

        try:
            # 1. Open WhatsApp
            if self._entry:
                target = self._entry.executable
                if target.lower().startswith("explorer "):
                    target = target.split(" ", 1)[1]
                subprocess.Popen(
                    ["explorer", target],
                    shell=False,
                )
                delay = self._entry.launch_delay or 5.0
            else:
                subprocess.Popen(
                    ["explorer", "shell:AppsFolder\\5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App"],
                    shell=False,
                )
                delay = 5.0

            time.sleep(delay)

            # 2. Search for contact
            pyautogui.hotkey("ctrl", "f")
            time.sleep(1.0)

            pyautogui.write(name, interval=0.05)
            time.sleep(2.0)

            pyautogui.press("enter")
            time.sleep(2.0)

            # 3. Type and send message
            pyautogui.write(message, interval=0.03)
            time.sleep(0.5)

            pyautogui.press("enter")

            return ActionResult(
                success=True,
                message=f"Sent WhatsApp message to {name}",
            )

        except Exception as exc:
            log.error("WhatsApp automation failed: %s", exc)
            return ActionResult(
                success=False,
                error=f"WhatsApp error: {exc}",
            )
