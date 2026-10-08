"""
FRIDAY Voice Assistant — Website Handler.

Handles ``open_website`` by opening URLs in the default browser
via :func:`webbrowser.open`.  URLs are validated by the security
layer before reaching this handler.
"""

from __future__ import annotations

import webbrowser
from typing import Set

from execution.handlers.base import ActionHandler, ActionResult
from intelligence.response_parser import ActionRequest
from utils.logger import get_logger

log = get_logger("execution.handlers.web")


class WebHandler(ActionHandler):
    """Opens URLs in the system default browser."""

    @property
    def handled_actions(self) -> Set[str]:
        return {"open_website"}

    def execute(self, action: ActionRequest) -> ActionResult:
        url = action.value.strip()
        if not url:
            return ActionResult(success=False, error="No URL provided")

        # Ensure the URL has a scheme
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        log.info("Opening website: %s", url)

        try:
            webbrowser.open(url)
            return ActionResult(
                success=True,
                message=f"Opened {url}",
            )
        except Exception as exc:
            log.error("Failed to open URL %s: %s", url, exc)
            return ActionResult(
                success=False,
                error=f"Failed to open {url}: {exc}",
            )
