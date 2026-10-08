"""
FRIDAY Voice Assistant — Text-to-Speech Engine.

Wraps :mod:`pyttsx3` with:

* Configurable voice, rate, and volume (via ``config.yaml``)
* Thread-safe ``speak()`` and ``speak_async()`` methods
* Queued speech to prevent overlapping utterances
* ``confirm()`` method for user decision points (returns spoken
  choice for the retry engine)

Usage::

    from speech.tts import TextToSpeech
    from config import load_config

    cfg = load_config()
    tts = TextToSpeech(cfg)

    tts.speak("Hello!")              # blocking
    tts.speak_async("Processing…")   # non-blocking

    choice = tts.confirm(
        "Step failed. Continue, skip, or abort?",
        options=["continue", "skip", "abort"],
    )
"""

from __future__ import annotations

import queue
import threading
from typing import List, Optional

import pyttsx3

from config import FridayConfig
from utils.logger import get_logger

log = get_logger("speech.tts")


class TextToSpeech:
    """
    Thread-safe text-to-speech engine.

    Runs ``pyttsx3`` on a dedicated background thread to prevent
    blocking the main loop during speech synthesis.  All utterances
    are queued and spoken sequentially.

    Args:
        cfg: FRIDAY configuration.
    """

    def __init__(self, cfg: FridayConfig) -> None:
        self._cfg = cfg
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._lock = threading.Lock()

        # Initialize engine on the background thread (pyttsx3
        # requires the engine to be used on the thread that created it)
        self._ready = threading.Event()
        self._init_error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="tts-worker"
        )
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            self._init_error = RuntimeError("TTS engine initialization timed out")

        log.info(
            "TTS engine initialized (rate=%d, volume=%.1f)",
            cfg.tts.rate,
            cfg.tts.volume,
        )

    # ── Public API ──────────────────────────────────────────

    def speak(self, text: str) -> None:
        """
        Speak text and block until utterance completes.

        Args:
            text: The text to speak.
        """
        if not text:
            return
        if self._init_error is not None:
            log.error("TTS unavailable: %s", self._init_error)
            print(f"AI: {text}")
            return
        done = threading.Event()
        self._queue.put((text, done))  # type: ignore[arg-type]
        log.debug("TTS queued (blocking): '%s'", text[:60])
        done.wait()

    def speak_async(self, text: str) -> None:
        """
        Queue text for speaking without blocking the caller.

        Args:
            text: The text to speak.
        """
        if not text:
            return
        if self._init_error is not None:
            log.error("TTS unavailable: %s", self._init_error)
            print(f"AI: {text}")
            return
        self._queue.put((text, None))  # type: ignore[arg-type]
        log.debug("TTS queued (async): '%s'", text[:60])

    def confirm(
        self,
        prompt: str,
        options: Optional[List[str]] = None,
    ) -> str:
        """
        Speak a prompt and listen for a spoken response.

        This is used by the retry engine at user decision points.

        Args:
            prompt: The question to ask the user.
            options: Valid response options (for logging; actual
                     matching is done by the caller).

        Returns:
            The user's spoken response (lowercased), or ``"abort"``
            if recognition fails.
        """
        import speech_recognition as sr

        self.speak(prompt)
        log.info(
            "Waiting for user decision (options: %s)",
            options or ["any"],
        )

        try:
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
                text = recognizer.recognize_google(audio).lower().strip()
                log.info("User decision: '%s'", text)
                return text
        except Exception as exc:
            log.warning("Failed to capture user decision: %s — defaulting to 'abort'", exc)
            self.speak("I couldn't hear your response. Aborting for safety.")
            return "abort"

    def shutdown(self) -> None:
        """Stop the TTS worker thread."""
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join(timeout=5.0)
        log.info("TTS engine shut down")

    # ── Worker Thread ───────────────────────────────────────

    def _worker(self) -> None:
        """Background thread that processes the speech queue."""
        try:
            engine = pyttsx3.init()
        except Exception as exc:
            self._init_error = exc
            self._ready.set()
            log.error("TTS engine failed to initialize: %s", exc)
            return

        # Apply configuration
        engine.setProperty("rate", self._cfg.tts.rate)
        engine.setProperty("volume", self._cfg.tts.volume)

        # Select voice by index
        voices = engine.getProperty("voices")
        if voices and self._cfg.tts.voice_index < len(voices):
            engine.setProperty("voice", voices[self._cfg.tts.voice_index].id)

        self._ready.set()

        while True:
            item = self._queue.get()

            # Shutdown sentinel
            if item is None:
                break

            text, done_event = item  # type: ignore[misc]

            try:
                log.debug("TTS speaking: '%s'", text[:60])
                print(f"AI: {text}")
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                log.error("TTS error: %s", exc)
            finally:
                if done_event is not None:
                    done_event.set()
