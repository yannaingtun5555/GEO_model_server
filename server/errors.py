"""Stable service errors that are safe to expose through the internal API."""

from __future__ import annotations

from typing import Any


class ServiceError(RuntimeError):
    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        retryable: bool = False,
        details: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.safe_message = message
        self.retryable = retryable
        self.details = details
