"""
FRIDAY Voice Assistant — Configuration Loader.

Loads ``config.yaml`` into typed dataclass objects. Falls back to
sensible defaults when the YAML file is missing or incomplete.

Usage:
    from config import load_config
    cfg = load_config()           # loads from project root
    cfg = load_config("my.yaml")  # loads from custom path
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from utils.logger import get_logger

log = get_logger("config")

# Project root directory (where main.py lives)
PROJECT_ROOT: Path = Path(__file__).resolve().parent


# ── Dataclass Hierarchy ─────────────────────────────────────

@dataclass(frozen=True)
class LLMConfig:
    """Settings for the Ollama / Llama3 backend."""
    model: str = "llama3"
    host: str = "http://localhost:11434"
    timeout: int = 30
    temperature: float = 0.1
    cache_size: int = 128


@dataclass(frozen=True)
class SpeechConfig:
    """Speech recognition settings."""
    wake_word: str = "hey friday"
    ambient_noise_duration: float = 1.0
    phrase_time_limit: int = 5
    wake_phrase_time_limit: int = 3
    listen_timeout: int = 7
    energy_threshold: Optional[int] = None
    pause_threshold: float = 0.8
    dynamic_energy: bool = True


@dataclass(frozen=True)
class TTSConfig:
    """Text-to-speech settings."""
    rate: int = 180
    volume: float = 0.9
    voice_index: int = 0


@dataclass(frozen=True)
class AppEntry:
    """A single registered application."""
    executable: str
    process_name: str
    aliases: List[str] = field(default_factory=list)
    is_uwp: bool = False
    launch_delay: float = 0.0


@dataclass(frozen=True)
class SecurityConfig:
    """Security and sandboxing settings."""
    allowed_executables: List[str] = field(default_factory=lambda: ["*"])
    blocked_patterns: List[str] = field(default_factory=list)
    max_chain_length: int = 10
    command_cooldown: float = 0.5
    max_input_length: int = 500
    allowed_url_schemes: List[str] = field(default_factory=lambda: ["http", "https"])


@dataclass(frozen=True)
class RetryConfig:
    """Retry / failure-recovery settings."""
    max_attempts: int = 3
    backoff_base: float = 1.0
    backoff_max: float = 8.0
    fail_fast_actions: List[str] = field(default_factory=lambda: ["send_whatsapp"])


@dataclass(frozen=True)
class LoggingConfig:
    """Logging settings."""
    level: str = "INFO"
    console: bool = True
    file: str = "logs/friday.log"
    audit_file: str = "logs/audit.log"
    max_file_size_mb: int = 10
    backup_count: int = 5


@dataclass(frozen=True)
class ContextConfig:
    """Context / memory settings."""
    window_size: int = 10
    track_open_apps: bool = True


@dataclass(frozen=True)
class AutomationConfig:
    """Execution policy and local intent handling settings."""
    direct_commands_enabled: bool = True
    dry_run: bool = False
    queue_size: int = 50
    min_confidence: float = 0.5
    agent_enabled: bool = True
    max_replans: int = 3
    confirmation_actions: List[str] = field(default_factory=lambda: ["send_whatsapp"])
    blocked_actions: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class MemoryConfig:
    """Persistent learning and alias memory settings."""
    enabled: bool = True
    file: str = "data/memory.json"


@dataclass(frozen=True)
class FridayConfig:
    """Top-level configuration object grouping all sub-configs."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    apps: Dict[str, AppEntry] = field(default_factory=dict)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    automation: AutomationConfig = field(default_factory=AutomationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


# ── Builders ────────────────────────────────────────────────

def _build_apps(raw: Dict[str, Any]) -> Dict[str, AppEntry]:
    """Parse the apps section of config.yaml into ``AppEntry`` objects."""
    apps: Dict[str, AppEntry] = {}
    for key, val in raw.items():
        if not isinstance(val, dict):
            continue
        apps[key] = AppEntry(
            executable=val.get("executable", ""),
            process_name=val.get("process_name", ""),
            aliases=val.get("aliases", []),
            is_uwp=val.get("is_uwp", False),
            launch_delay=val.get("launch_delay", 0.0),
        )
    return apps


def _safe_get(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Return a sub-dict or empty dict if key is missing/not a dict."""
    val = data.get(key)
    return val if isinstance(val, dict) else {}


def load_config(path: Optional[str] = None) -> FridayConfig:
    """
    Load and validate configuration from a YAML file.

    Args:
        path: Path to the YAML file. Defaults to ``config.yaml``
              in the project root.

    Returns:
        A fully-populated :class:`FridayConfig` instance.
        Missing sections fall back to defaults.
    """
    config_path = Path(path) if path else PROJECT_ROOT / "config.yaml"

    raw: Dict[str, Any] = {}
    if config_path.exists():
        log.info("Loading configuration from %s", config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    else:
        log.warning(
            "Config file not found at %s — using defaults", config_path
        )

    try:
        cfg = FridayConfig(
            llm=LLMConfig(**_safe_get(raw, "llm")),
            speech=SpeechConfig(**_safe_get(raw, "speech")),
            tts=TTSConfig(**_safe_get(raw, "tts")),
            apps=_build_apps(raw.get("apps", {})),
            security=SecurityConfig(**_safe_get(raw, "security")),
            retry=RetryConfig(**_safe_get(raw, "retry")),
            logging=LoggingConfig(**_safe_get(raw, "logging")),
            context=ContextConfig(**_safe_get(raw, "context")),
            automation=AutomationConfig(**_safe_get(raw, "automation")),
            memory=MemoryConfig(**_safe_get(raw, "memory")),
        )
    except TypeError as exc:
        log.error("Invalid config key: %s — falling back to defaults", exc)
        cfg = FridayConfig()

    log.info(
        "Configuration loaded — model=%s, wake_word='%s', %d registered apps",
        cfg.llm.model,
        cfg.speech.wake_word,
        len(cfg.apps),
    )
    return cfg


def resolve_app(cfg: FridayConfig, name: str) -> Optional[AppEntry]:
    """
    Look up an app by name or alias in the config registry.

    Args:
        cfg: The loaded configuration.
        name: The app name or alias to search for (case-insensitive).

    Returns:
        The matching :class:`AppEntry`, or ``None`` if not found
        (caller should fall back to the generic handler).
    """
    name_lower = name.lower().strip()
    for _key, entry in cfg.apps.items():
        if name_lower in [a.lower() for a in entry.aliases]:
            return entry
        if name_lower == _key.lower():
            return entry
    return None
