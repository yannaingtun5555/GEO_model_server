#!/usr/bin/env python3
"""
server/core/cache.py — Redis & In-Memory Cache Manager
"""

import json
import hashlib
import time
from typing import Optional, Dict, Any
from server.config import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD, CACHE_TTL_SECONDS

class CacheManager:
    """
    Manages Redis caching for model predictions and pre-computed regional layers.
    Falls back gracefully to an in-memory dictionary if Redis is offline or not installed.
    """

    def __init__(self):
        self.redis_client = None
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._init_redis()

    def _init_redis(self):
        try:
            import redis
            client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                socket_timeout=2.0,
                decode_responses=True
            )
            client.ping()
            self.redis_client = client
            print(f"[CACHE] Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            print(f"[CACHE WARN] Redis connection failed ({e}). Operating in-memory fallback cache.")
            self.redis_client = None

    def generate_cache_key(self, key_prefix: str, payload: Dict[str, Any]) -> str:
        """Generates a deterministic SHA256 cache key from request parameters."""
        serialized = json.dumps(payload, sort_keys=True)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        return f"{key_prefix}:{digest}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached JSON object by key."""
        if self.redis_client:
            try:
                val = self.redis_client.get(key)
                if val:
                    return json.loads(val)
            except Exception as e:
                print(f"[CACHE WARN] Redis GET error: {e}")

        # In-memory fallback lookup
        if key in self._memory_cache:
            item = self._memory_cache[key]
            if item["expires_at"] > time.time():
                return item["data"]
            else:
                del self._memory_cache[key]
        return None

    def set(self, key: str, data: Dict[str, Any], ttl_seconds: int = CACHE_TTL_SECONDS):
        """Save JSON object to cache with TTL."""
        if self.redis_client:
            try:
                val = json.dumps(data)
                self.redis_client.setex(key, ttl_seconds, val)
                return
            except Exception as e:
                print(f"[CACHE WARN] Redis SET error: {e}")

        # In-memory fallback store
        self._memory_cache[key] = {
            "data": data,
            "expires_at": time.time() + ttl_seconds
        }

    def delete(self, key: str):
        """Remove a key from cache."""
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception:
                pass
        self._memory_cache.pop(key, None)


# Global singleton instance
cache_manager = CacheManager()
