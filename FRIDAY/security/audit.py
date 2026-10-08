"""
FRIDAY Voice Assistant — Audit Logger.

Records every executed action in a structured, append-only audit log
for compliance, debugging, and post-mortem analysis.

Each audit entry includes:
- ISO-8601 timestamp
- Action type and parameters
- Execution result (success / failure / skipped)
- Retry attempt count
- User decisions (for interactive failure recovery)

The audit log is separate from the application log to ensure it
captures *only* security-relevant events without noise.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from utils.logger import get_logger

log = get_logger("security.audit")


class AuditLogger:
    """
    Append-only audit logger for action execution events.

    Uses a dedicated file handler separate from the application logger
    so audit entries are never mixed with debug output.
    """

    def __init__(self, audit_file: str, max_size_mb: int = 10, backup_count: int = 5) -> None:
        """
        Initialize the audit logger.

        Args:
            audit_file: Path to the audit log file.
            max_size_mb: Maximum file size before rotation.
            backup_count: Number of rotated files to keep.
        """
        self._logger = logging.getLogger("friday.audit")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False  # Don't leak to root logger
        for existing in list(self._logger.handlers):
            self._logger.removeHandler(existing)
            existing.close()

        # Ensure directory exists
        directory = os.path.dirname(audit_file)
        if directory:
            os.makedirs(directory, exist_ok=True)

        handler = logging.handlers.RotatingFileHandler(
            audit_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)

        log.info("Audit logger initialized → %s", audit_file)

    def record(
        self,
        action: str,
        params: Dict[str, Any],
        result: str,
        attempt: int = 1,
        error: Optional[str] = None,
        user_decision: Optional[str] = None,
    ) -> None:
        """
        Write a single audit entry.

        Args:
            action: The action type (e.g. ``'open_app'``).
            params: Action parameters as a dict.
            result: Outcome — ``'success'``, ``'failure'``, ``'skipped'``,
                    ``'aborted'``.
            attempt: Which attempt number this was (1 = first try).
            error: Error message if the action failed.
            user_decision: The user's choice at a decision point
                           (``'continue'``, ``'skip'``, ``'abort'``).
        """
        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "params": params,
            "result": result,
            "attempt": attempt,
        }
        if error:
            entry["error"] = error
        if user_decision:
            entry["user_decision"] = user_decision

        self._logger.info(json.dumps(entry, ensure_ascii=False))
