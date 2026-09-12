"""Best-effort statistics output that cannot wait for the log destination."""

import logging
from logging.handlers import QueueListener
from queue import Full, Queue


class _ForwardHandler(logging.Handler):
    def __init__(self, logger: logging.Logger):
        super().__init__()
        self.logger = logger

    def emit(self, record: logging.LogRecord):
        try:
            self.logger.handle(record)
        except Exception:
            # A custom handler must not permanently stop draining the queue.
            self.handleError(record)


class AsyncStatsLogger:
    """A single producer queues immutable INFO snapshots to a daemon writer.

    Create inside the scheduler process, after logging is configured. Preserve
    the producer's timestamp/caller and use the existing logger's handlers and
    filters. A stalled sink drops snapshots once the queue fills; the next
    accepted snapshot reports the omission count. Pending snapshots are not
    flushed on process exit, since flushing could block shutdown indefinitely.
    """

    def __init__(self, logger: logging.Logger, capacity: int = 64):
        if capacity <= 0:
            raise ValueError("Statistics log capacity must be positive")
        self.logger = logger
        self.queue = Queue(maxsize=capacity)
        self.dropped = 0
        self.listener = QueueListener(self.queue, _ForwardHandler(logger))
        self.listener.start()

    def info(self, message: str):
        if not self.logger.isEnabledFor(logging.INFO):
            return
        filename, lineno, funcname, _ = self.logger.findCaller(stacklevel=2)
        if self.dropped:
            message += f", omitted statistics logs: {self.dropped}"
        record = self.logger.makeRecord(
            self.logger.name,
            logging.INFO,
            filename,
            lineno,
            message,
            (),
            None,
            funcname,
        )
        try:
            self.queue.put_nowait(record)
        except Full:
            self.dropped += 1
        else:
            self.dropped = 0
