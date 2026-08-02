"""Version-aware Redis cache with a bounded in-process fallback."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any

from server.config import (
    CACHE_TTL_SECONDS,
    MAX_CACHE_ENTRIES,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
)


class CacheManager:
    def __init__(self) -> None:
        self.redis_client = None
        self.redis_error: str | None = None
        self._memory_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()
        self._init_redis()

    def _init_redis(self) -> None:
        try:
            import redis

            client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                socket_connect_timeout=0.5,
                socket_timeout=0.5,
                decode_responses=True,
            )
            client.ping()
            self.redis_client = client
            self.redis_error = None
        except Exception as exc:
            self.redis_client = None
            self.redis_error = str(exc)

    @staticmethod
    def generate_cache_key(namespace: str, payload: dict[str, Any]) -> str:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return f"{namespace}:{digest}"

    def get(self, key: str) -> dict[str, Any] | None:
        if self.redis_client is not None:
            try:
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
            except Exception as exc:
                self.redis_error = str(exc)

        with self._lock:
            item = self._memory_cache.get(key)
            if item is None:
                return None
            if item["expires_at"] <= time.time():
                self._memory_cache.pop(key, None)
                return None
            self._memory_cache.move_to_end(key)
            return copy.deepcopy(item["data"])

    def set(
        self,
        key: str,
        data: dict[str, Any],
        ttl_seconds: int = CACHE_TTL_SECONDS,
    ) -> None:
        if self.redis_client is not None:
            try:
                self.redis_client.setex(
                    key,
                    ttl_seconds,
                    json.dumps(data, separators=(",", ":")),
                )
                return
            except Exception as exc:
                self.redis_error = str(exc)

        with self._lock:
            self._memory_cache[key] = {
                "data": copy.deepcopy(data),
                "expires_at": time.time() + ttl_seconds,
            }
            self._memory_cache.move_to_end(key)
            while len(self._memory_cache) > MAX_CACHE_ENTRIES:
                self._memory_cache.popitem(last=False)

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "redis_connected": self.redis_client is not None,
                "memory_cache_entries": len(self._memory_cache),
                "memory_cache_limit": MAX_CACHE_ENTRIES,
                "redis_error": self.redis_error,
            }


cache_manager = CacheManager()
