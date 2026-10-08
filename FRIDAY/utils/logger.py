"""
FRIDAY Voice Assistant — Structured Logging.

Provides per-module loggers with JSON-formatted file output and
colored console output. All loggers share a common 'friday.*'
namespace for unified configuration.

Usage:
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Something happened", extra={"detail": "value"})
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from colorama import Fore, Style, init as colorama_init

    colorama_init(autoreset=True)
    _HAS_COLORAMA = True
except ImportError:
    _HAS_COLORAMA = False


# ── JSON Formatter (file logs) ──────────────────────────────

class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for machine parsing."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge any extra fields the caller passed
        if hasattr(record, "detail"):
            log_entry["detail"] = record.detail  # type: ignore[attr-defined]
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


# ── Colored Console Formatter ───────────────────────────────

class ColoredFormatter(logging.Formatter):
    """Human-readable colored console output (falls back to plain if
    colorama is unavailable)."""

    _LEVEL_COLORS = {
        "DEBUG": Fore.CYAN if _HAS_COLORAMA else "",
        "INFO": Fore.GREEN if _HAS_COLORAMA else "",
        "WARNING": Fore.YELLOW if _HAS_COLORAMA else "",
        "ERROR": Fore.RED if _HAS_COLORAMA else "",
        "CRITICAL": Fore.RED + Style.BRIGHT if _HAS_COLORAMA else "",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self._LEVEL_COLORS.get(record.levelname, "")
        reset = Style.RESET_ALL if _HAS_COLORAMA else ""
        timestamp = datetime.now().strftime("%H:%M:%S")
        # Shorten logger name: 'friday.speech.recognizer' -> 'speech.recognizer'
        short_name = record.name.replace("friday.", "", 1) if record.name.startswith("friday.") else record.name
        msg = f"{Fore.WHITE if _HAS_COLORAMA else ''}{timestamp}{reset} {color}{record.levelname:<8}{reset} [{short_name}] {record.getMessage()}"
        if record.exc_info and record.exc_info[1]:
            msg += f"\n{self.formatException(record.exc_info)}"
        return msg


# ── Logger Factory ──────────────────────────────────────────

_initialized = False


def _ensure_log_dir(path: str) -> None:
    """Create the directory for a log file if it doesn't exist."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def setup_logging(
    level: str = "INFO",
    console: bool = True,
    log_file: Optional[str] = None,
    max_file_size_mb: int = 10,
    backup_count: int = 5,
) -> None:
    """
    Configure the root 'friday' logger with console and/or file handlers.

    This should be called once at startup from main.py.
    Subsequent calls are no-ops.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        console: Whether to add a colored console handler.
        log_file: Path to the rotating log file (None = no file logging).
        max_file_size_mb: Max size per log file before rotation.
        backup_count: Number of rotated files to keep.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    root_logger = logging.getLogger("friday")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.propagate = False

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter())
        root_logger.addHandler(console_handler)

    # File handler (rotating)
    if log_file:
        _ensure_log_dir(log_file)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger under the 'friday' namespace.

    Args:
        name: Module name (typically ``__name__``). Will be prefixed
              with 'friday.' if not already.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    if not name.startswith("friday."):
        # Convert module paths: 'speech.recognizer' -> 'friday.speech.recognizer'
        name = f"friday.{name}"
    return logging.getLogger(name)
