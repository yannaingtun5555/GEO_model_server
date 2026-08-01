#!/usr/bin/env python3
"""
server/core/request_queue.py — Asynchronous Request Queue & Concurrency Manager
================================================================================
Prevents server CPU overload and Out-Of-Memory (OOM) failures under heavy traffic spikes
by queuing incoming prediction tasks through a bounded concurrency worker pool.
"""

import asyncio
import time
from typing import Callable, Any, Dict
from server.config import MAX_CONCURRENT_REQUESTS, QUEUE_TIMEOUT_SECONDS

class AsyncRequestQueue:
    """
    Bounded Concurrency Worker Queue for Model Inference.
    - Limits active inference worker tasks to MAX_CONCURRENT_REQUESTS (default: 10).
    - Queues excess incoming requests safely with automated queue wait metrics.
    """

    def __init__(self, max_concurrent: int = MAX_CONCURRENT_REQUESTS, timeout: float = QUEUE_TIMEOUT_SECONDS):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        # Diagnostics
        self.active_requests: int = 0
        self.queued_requests: int = 0
        self.total_processed: int = 0
        self.total_queued: int = 0

    async def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Executes an async task through the concurrency semaphore queue.
        Returns result dict containing output and queue execution metrics.
        """
        start_queue_t = time.time()
        self.queued_requests += 1
        self.total_queued += 1

        try:
            # Acquire concurrency slot (wait if max workers busy)
            async with asyncio.timeout(self.timeout):
                await self._semaphore.acquire()
        except asyncio.TimeoutError:
            self.queued_requests -= 1
            raise RuntimeError(f"Request Queue Timeout: Server busy (waited > {self.timeout}s in queue).")

        self.queued_requests -= 1
        self.active_requests += 1
        queue_wait_ms = round((time.time() - start_queue_t) * 1000.0, 2)

        try:
            # Execute actual model prediction function
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self.total_processed += 1
            
            # Attach queue metrics if result is a dictionary
            if isinstance(result, dict):
                result["_queue_metadata"] = {
                    "queue_wait_ms": queue_wait_ms,
                    "active_workers": self.active_requests,
                    "queued_depth": self.queued_requests
                }
            return result
        finally:
            self.active_requests -= 1
            self._semaphore.release()

    def get_metrics(self) -> Dict[str, Any]:
        """Returns request queue metrics."""
        return {
            "max_concurrent_capacity": self.max_concurrent,
            "currently_active_workers": self.active_requests,
            "currently_queued_requests": self.queued_requests,
            "total_requests_processed": self.total_processed,
            "total_requests_queued": self.total_queued
        }


# Global singleton instance
request_queue = AsyncRequestQueue()
