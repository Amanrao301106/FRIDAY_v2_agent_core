"""
FRIDAY Voice Assistant — Speech Recognition.

Provides two main classes:

* :class:`WakeWordListener` — Continuously listens for the wake
  phrase ("Hey Friday") and invokes a callback when detected.
* :class:`CommandListener` — One-shot listener that captures a single
  command phrase after the wake word is detected.

Both classes share a single ``sr.Recognizer`` instance and calibrate
ambient noise automatically. The microphone is re-opened per listen
cycle to avoid hardware lock issues on Windows.

Usage::

    from speech.recognizer import WakeWordListener, CommandListener
    from config import load_config

    cfg = load_config()
    wake = WakeWordListener(cfg)
    cmd  = CommandListener(cfg)

    while True:
        wake.wait()          # blocks until "hey friday"
        text = cmd.listen()  # blocks until command captured
        print(text)
"""

from __future__ import annotations

import speech_recognition as sr
from typing import Optional

from config import FridayConfig
from utils.logger import get_logger

log = get_logger("speech.recognizer")


# ── Custom Exceptions ───────────────────────────────────────

class RecognitionError(Exception):
    """Base class for speech recognition errors."""


class WakeWordTimeout(RecognitionError):
    """Raised when wake word detection times out (unused currently
    since we listen indefinitely, but available for future use)."""


class CommandTimeout(RecognitionError):
    """Raised when the user doesn't speak a command in time."""


class RecognitionFailed(RecognitionError):
    """Raised when Google Speech API can't understand the audio."""


# ── Shared Recognizer ──────────────────────────────────────

class _SharedRecognizer:
    """
    Internal singleton-ish wrapper around ``sr.Recognizer``.

    Configures energy threshold and dynamic energy adjustment
    once, then reuses the recognizer across all listen calls.
    """

    def __init__(self, cfg: FridayConfig) -> None:
        self._cfg = cfg
        self._recognizer = sr.Recognizer()

        # Configure recognizer settings
        self._recognizer.pause_threshold = cfg.speech.pause_threshold
        self._recognizer.dynamic_energy_threshold = cfg.speech.dynamic_energy

        if cfg.speech.energy_threshold is not None:
            self._recognizer.energy_threshold = cfg.speech.energy_threshold
            log.info(
                "Energy threshold set to %d (manual)",
                cfg.speech.energy_threshold,
            )
        else:
            log.info("Energy threshold: auto (dynamic adjustment enabled)")

    @property
    def recognizer(self) -> sr.Recognizer:
        return self._recognizer

    def calibrate(self, source: sr.Microphone) -> None:
        """Calibrate ambient noise levels from the given source."""
        duration = self._cfg.speech.ambient_noise_duration
        log.debug("Calibrating ambient noise (%.1fs)…", duration)
        self._recognizer.adjust_for_ambient_noise(source, duration=duration)
        log.debug(
            "Ambient calibration done — energy_threshold=%.0f",
            self._recognizer.energy_threshold,
        )


# ── Wake Word Listener ─────────────────────────────────────

class WakeWordListener:
    """
    Blocks until the configured wake phrase is detected.

    Also handles the "stop" keyword to signal shutdown.

    Args:
        cfg: FRIDAY configuration.
    """

    def __init__(self, cfg: FridayConfig) -> None:
        self._cfg = cfg
        self._shared = _SharedRecognizer(cfg)
        self._wake_word = cfg.speech.wake_word.lower()
        self._phrase_limit = cfg.speech.wake_phrase_time_limit

    def wait(self) -> str:
        """
        Block until the wake word is heard.

        Returns:
            ``'wake'`` if the wake word was detected, or
            ``'stop'`` if the shutdown keyword was heard.

        Raises:
            RecognitionError: On unrecoverable microphone errors.
        """
        log.info("Listening for wake word '%s'…", self._wake_word)

        with sr.Microphone() as source:
            self._shared.calibrate(source)

            while True:
                try:
                    audio = self._shared.recognizer.listen(
                        source,
                        timeout=None,
                        phrase_time_limit=self._phrase_limit,
                    )

                    text = self._shared.recognizer.recognize_google(audio)
                    text_lower = text.lower()

                    log.debug("Heard (wake phase): '%s'", text_lower)

                    if "stop" in text_lower:
                        log.info("Shutdown keyword detected")
                        return "stop"

                    if self._wake_word in text_lower:
                        log.info("Wake word detected!")
                        return "wake"

                except sr.UnknownValueError:
                    # Speech was unintelligible — keep listening
                    continue
                except sr.RequestError as exc:
                    log.error(
                        "Google Speech API error: %s — retrying", exc
                    )
                    continue
                except OSError as exc:
                    log.error("Microphone error: %s", exc)
                    raise RecognitionError(
                        f"Microphone error: {exc}"
                    ) from exc


# ── Command Listener ───────────────────────────────────────

class CommandListener:
    """
    One-shot listener that captures a single spoken command.

    Args:
        cfg: FRIDAY configuration.
    """

    def __init__(self, cfg: FridayConfig) -> None:
        self._cfg = cfg
        self._shared = _SharedRecognizer(cfg)
        self._timeout = cfg.speech.listen_timeout
        self._phrase_limit = cfg.speech.phrase_time_limit

    def listen(self) -> Optional[str]:
        """
        Listen for a single command phrase.

        Returns:
            The recognized text (lowercased), or ``None`` if nothing
            was captured within the timeout.
        """
        log.info("Listening for command (timeout=%ds)…", self._timeout)

        try:
            with sr.Microphone() as source:
                audio = self._shared.recognizer.listen(
                    source,
                    timeout=self._timeout,
                    phrase_time_limit=self._phrase_limit,
                )

                text = self._shared.recognizer.recognize_google(audio)
                result = text.lower().strip()
                log.info("Command recognized: '%s'", result)
                return result

        except sr.WaitTimeoutError:
            log.warning("Command listen timed out")
            return None
        except sr.UnknownValueError:
            log.warning("Could not understand audio")
            return None
        except sr.RequestError as exc:
            log.error("Google Speech API error: %s", exc)
            return None
        except OSError as exc:
            log.error("Microphone error during command listen: %s", exc)
            return None
