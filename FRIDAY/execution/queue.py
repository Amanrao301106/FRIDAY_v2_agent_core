"""
FRIDAY Voice Assistant — Command Queue.

Thread-safe queue that accepts commands from the speech recognition
thread and processes them sequentially on a single executor thread.

This hybrid model ensures:
* Voice input is accepted concurrently (never blocked)
* Commands execute one at a time for predictability
* Queue depth is visible for monitoring

Usage::

    from execution.queue import CommandQueue

    queue = CommandQueue(executor, max_size=50)
    queue.start()                      # starts consumer thread
    queue.enqueue(parsed_response)     # from speech thread
    queue.shutdown()                   # graceful stop
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, Optional, TYPE_CHECKING

from intelligence.response_parser import ParsedResponse
from utils.logger import get_logger

if TYPE_CHECKING:
    pass

log = get_logger("execution.queue")


class CommandQueue:
    """
    Thread-safe command queue with a single-threaded consumer.

    Args:
        process_fn: Callable that processes a single
                    :class:`ParsedResponse`.  This is typically
                    ``ActionExecutor.execute_chain``.
        max_size: Maximum queue depth (0 = unlimited).
    """

    def __init__(
        self,
        process_fn: Callable[[ParsedResponse, str], None],
        max_size: int = 50,
    ) -> None:
        self._queue: queue.Queue[Optional[tuple]] = queue.Queue(
            maxsize=max_size
        )
        self._process_fn = process_fn
        self._thread: Optional[threading.Thread] = None
        self._running = False

        log.info("CommandQueue initialized (max_size=%d)", max_size)

    def start(self) -> None:
        """Start the consumer thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._consumer_loop,
            daemon=True,
            name="cmd-queue-consumer",
        )
        self._thread.start()
        log.info("Command queue consumer started")

    def enqueue(self, response: ParsedResponse, user_input: str) -> bool:
        """
        Add a parsed response to the queue for sequential execution.

        Args:
            response: The parsed LLM response.
            user_input: The original user command (for context tracking).

        Returns:
            ``True`` if enqueued successfully, ``False`` if the queue
            is full.
        """
        try:
            self._queue.put_nowait((response, user_input))
            log.info(
                "Enqueued command (%d actions, queue depth=%d)",
                len(response.actions),
                self._queue.qsize(),
            )
            return True
        except queue.Full:
            log.warning("Command queue is full — dropping command")
            return False

    @property
    def depth(self) -> int:
        """Return the current number of commands waiting in the queue."""
        return self._queue.qsize()

    def shutdown(self) -> None:
        """Signal the consumer thread to stop and wait for it."""
        if not self._running:
            return
        self._running = False
        try:
            self._queue.put_nowait(None)  # Sentinel to unblock .get()
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(None)
        if self._thread:
            self._thread.join(timeout=10.0)
        log.info("Command queue shut down")

    # ── Consumer Loop ───────────────────────────────────────

    def _consumer_loop(self) -> None:
        """Background thread that processes commands sequentially."""
        log.info("Consumer loop running")
        while self._running:
            item = self._queue.get()

            # Shutdown sentinel
            if item is None:
                break

            response, user_input = item

            try:
                self._process_fn(response, user_input)
            except Exception as exc:
                log.error(
                    "Unhandled exception in command processing: %s", exc,
                    exc_info=True,
                )

        log.info("Consumer loop exited")
