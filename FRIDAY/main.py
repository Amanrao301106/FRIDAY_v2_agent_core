"""
FRIDAY Voice Assistant — Main Entry Point.

Orchestrates the complete voice assistant lifecycle:

1. Load configuration from ``config.yaml``
2. Initialize logging, TTS, speech recognition, LLM, and execution
3. Start the command queue consumer thread
4. Enter the main loop: wake word → command → LLM → execute
5. Handle graceful shutdown on ``stop`` or Ctrl+C

Run with::

    python main.py

Architecture overview::

    ┌──────────────────────────────────────────┐
    │                Main Loop                  │
    │  WakeWord → Command → LLM → Parse → Queue│
    └────────────────────┬─────────────────────┘
                         │
                  ┌──────▼──────┐
                  │  CMD Queue   │  (sequential consumer)
                  │  Consumer    │
                  └──────┬──────┘
                         │
              ┌──────────▼──────────┐
              │  ActionExecutor      │
              │  → Security Check    │
              │  → Handler Dispatch  │
              │  → Retry Engine      │
              │  → Audit Log         │
              │  → Context Update    │
              └─────────────────────┘
"""

from __future__ import annotations

import signal
import sys
from typing import Optional

from config import load_config, FridayConfig
from utils.logger import setup_logging, get_logger


def main() -> None:
    """FRIDAY voice assistant entry point."""

    # ── 1. Load Configuration ───────────────────────────────
    cfg = load_config()

    # ── 2. Initialize Logging ───────────────────────────────
    setup_logging(
        level=cfg.logging.level,
        console=cfg.logging.console,
        log_file=cfg.logging.file,
        max_file_size_mb=cfg.logging.max_file_size_mb,
        backup_count=cfg.logging.backup_count,
    )
    log = get_logger("main")
    log.info("=" * 60)
    log.info("FRIDAY Voice Assistant starting up")
    log.info("=" * 60)

    # ── 3. Initialize Components ────────────────────────────

    # Import here to ensure logging is configured first
    from speech.tts import TextToSpeech
    from speech.recognizer import WakeWordListener, CommandListener
    from intelligence.llm_client import OllamaClient
    from intelligence.prompt_builder import build_system_prompt
    from intelligence.response_parser import parse_response, ParseError
    from intelligence.local_intent import parse_local_command
    from intelligence.memory import MemoryManager
    from intelligence.context import ContextManager
    from security.validator import SecurityValidator
    from security.audit import AuditLogger
    from execution.handlers import HandlerRegistry
    from execution.retry import RetryEngine
    from execution.executor import ActionExecutor
    from execution.queue import CommandQueue
    from agent.planner import TaskPlanner
    from agent.policy import ActionPolicy
    from agent.orchestrator import AgentOrchestrator

    # TTS Engine
    tts = TextToSpeech(cfg)

    # Speech Recognition
    wake_listener = WakeWordListener(cfg)
    cmd_listener = CommandListener(cfg)

    # Intelligence
    llm_client = OllamaClient(cfg)
    context = ContextManager(cfg)
    memory = MemoryManager(cfg)

    # Security
    validator = SecurityValidator(cfg)
    audit = AuditLogger(
        audit_file=cfg.logging.audit_file,
        max_size_mb=cfg.logging.max_file_size_mb,
        backup_count=cfg.logging.backup_count,
    )

    # Execution
    handler_registry = HandlerRegistry(cfg, tts)
    retry_engine = RetryEngine(cfg.retry, tts, handler_registry, audit)
    executor = ActionExecutor(
        cfg=cfg,
        registry=handler_registry,
        retry_engine=retry_engine,
        validator=validator,
        audit=audit,
        context=context,
        tts=tts,
    )

    # Agent Core — goal-based controlled autonomy
    planner = TaskPlanner(cfg, llm_client)
    policy = ActionPolicy(cfg)
    agent = AgentOrchestrator(
        cfg=cfg,
        planner=planner,
        executor=executor,
        policy=policy,
        confirm=tts.confirm,
        speak=tts.speak,
    )

    # Command Queue (sequential consumer)
    cmd_queue = CommandQueue(
        process_fn=executor.execute_chain,
        max_size=cfg.automation.queue_size,
    )
    cmd_queue.start()

    # ── 4. Graceful Shutdown Handler ────────────────────────

    def shutdown(signum: Optional[int] = None, frame=None) -> None:
        log.info("Shutting down FRIDAY…")
        tts.speak("Shutting down. Goodbye!")
        cmd_queue.shutdown()
        tts.shutdown()
        log.info("FRIDAY shut down complete")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── 5. Main Loop ───────────────────────────────────────

    log.info("FRIDAY is ready. Say '%s' to activate.", cfg.speech.wake_word)
    tts.speak("Friday is ready.")

    while True:
        try:
            # --- Wake Word Detection ---
            wake_result = wake_listener.wait()

            if wake_result == "stop":
                shutdown()
                return

            # Acknowledge wake word
            tts.speak("Yes?")

            # --- Command Listening ---
            log.info("Listening for command…")
            command = cmd_listener.listen()

            if not command:
                tts.speak("I didn't hear anything.")
                continue

            log.info("Command received: '%s'", command)

            # --- Input Sanitization ---
            sanitized = validator.sanitize_input(command)
            if not sanitized.allowed:
                tts.speak(f"Sorry, I can't process that. {sanitized.reason}")
                log.warning("Input rejected: %s", sanitized.reason)
                continue

            clean_command = sanitized.sanitized_value or command

            parsed = memory.feedback_from_command(
                clean_command,
                last_app=context.last_app_opened,
            )
            if parsed:
                log.info("Command handled by memory feedback")

            parsed = parsed or memory.learn_from_command(clean_command)
            if parsed:
                log.info("Command handled by memory learning")

            if cfg.automation.direct_commands_enabled:
                parsed = parsed or parse_local_command(
                    clean_command,
                    aliases=memory.aliases,
                    policy=memory.alias_policy(),
                )
                if parsed:
                    log.info("Command handled by local intent parser")

            if parsed is None:
                # --- Goal-based Agent Processing ---
                if cfg.automation.agent_enabled:
                    log.info("Agent mode: planning goal '%s'", clean_command)
                    task = agent.run(clean_command)
                    log.info(
                        "Agent task finished: status=%s, steps=%d, replans=%d",
                        task.status.value,
                        len(task.completed_steps),
                        task.replans,
                    )
                    continue

                # --- Legacy LLM Processing ---
                try:
                    history = context.get_history_strings()
                    system_prompt = build_system_prompt(
                        cfg,
                        clean_command,
                        history,
                        memory=memory.prompt_lines(),
                    )
                    raw_response = llm_client.query(
                        system_prompt=system_prompt,
                        user_message=clean_command,
                        use_cache=(len(history) == 0),
                    )
                    log.debug("LLM raw response: %s", raw_response[:200])
                except ConnectionError as exc:
                    tts.speak(
                        "I can't reach my brain right now. "
                        "Please check that Ollama is running."
                    )
                    log.error("LLM connection error: %s", exc)
                    continue

                try:
                    parsed = parse_response(raw_response)
                except ParseError as exc:
                    tts.speak("I didn't understand the response from my brain. Please try again.")
                    log.error("Parse error: %s", exc)
                    audit.record(
                        action="parse_error",
                        params={"raw": raw_response[:500]},
                        result="failure",
                        error=str(exc),
                    )
                    continue

            # Legacy/direct parsed responses still use the existing command queue.
            if not cmd_queue.enqueue(parsed, clean_command):
                tts.speak(
                    "I'm overwhelmed right now. Please wait a moment."
                )


        except KeyboardInterrupt:
            shutdown()
            return
        except Exception as exc:
            log.error("Unhandled error in main loop: %s", exc, exc_info=True)
            tts.speak("Something went wrong. Please try again.")
            continue


if __name__ == "__main__":
    main()
