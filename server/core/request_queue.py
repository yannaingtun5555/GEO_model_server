"""Bounded async admission control for CPU-bound model inference."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from server.config import (
    MAX_CONCURRENT_REQUESTS,
    QUEUE_TIMEOUT_SECONDS,
    REQUEST_EXECUTION_TIMEOUT_SECONDS,
)


class QueueTimeout(RuntimeError):
    """The service could not admit a request within the configured deadline."""


class ExecutionTimeout(RuntimeError):
    """Inference exceeded its response deadline but remains capacity-accounted."""


class AsyncRequestQueue:
    def __init__(
        self,
        max_concurrent: int = MAX_CONCURRENT_REQUESTS,
        queue_timeout: float = QUEUE_TIMEOUT_SECONDS,
        execution_timeout: float = REQUEST_EXECUTION_TIMEOUT_SECONDS,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.queue_timeout = queue_timeout
        self.execution_timeout = execution_timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self.active_requests = 0
        self.queued_requests = 0
        self.total_processed = 0
        self.total_timeouts = 0
        self.total_execution_timeouts = 0

    def _release_background_worker(self, task: asyncio.Task[Any]) -> None:
        # Consume a late exception so the event loop does not emit an unhandled
        # task warning. The worker retained its semaphore slot until this point.
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            pass
        self.active_requests -= 1
        self._semaphore.release()

    async def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
        started = time.perf_counter()
        self.queued_requests += 1
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(), timeout=self.queue_timeout
            )
        except TimeoutError as exc:
            self.total_timeouts += 1
            raise QueueTimeout("model server is at its bounded inference capacity") from exc
        finally:
            self.queued_requests -= 1

        queue_wait_ms = round((time.perf_counter() - started) * 1000, 2)
        self.active_requests += 1
        worker = asyncio.create_task(asyncio.to_thread(func, *args, **kwargs))
        release_on_exit = True
        try:
            result = await asyncio.wait_for(
                asyncio.shield(worker), timeout=self.execution_timeout
            )
            self.total_processed += 1
            return result, queue_wait_ms
        except TimeoutError as exc:
            self.total_execution_timeouts += 1
            release_on_exit = False
            worker.add_done_callback(self._release_background_worker)
            raise ExecutionTimeout(
                "model inference exceeded the configured execution deadline"
            ) from exc
        except asyncio.CancelledError:
            # asyncio threads cannot be killed safely. Keep the capacity slot
            # until the computation actually exits instead of allowing hidden
            # work to bypass the concurrency/RAM bound.
            release_on_exit = False
            worker.add_done_callback(self._release_background_worker)
            raise
        finally:
            if release_on_exit:
                self.active_requests -= 1
                self._semaphore.release()

    def get_metrics(self) -> dict[str, Any]:
        return {
            "max_concurrent_requests": self.max_concurrent,
            "active_requests": self.active_requests,
            "queued_requests": self.queued_requests,
            "total_processed": self.total_processed,
            "total_timeouts": self.total_timeouts,
            "total_execution_timeouts": self.total_execution_timeouts,
        }


request_queue = AsyncRequestQueue()
