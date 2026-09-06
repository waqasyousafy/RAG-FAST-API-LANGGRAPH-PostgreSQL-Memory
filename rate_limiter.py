"""
Rate Limiter
Per-user request throttling backed by Redis (fixed-window counter).
"""

import time
import redis
from typing import Optional


class RateLimiter:
    """
    Fixed-window rate limiter. Each user gets `max_requests` per
    `window_seconds`. Fails open (allows the request) if Redis is
    unreachable — infrastructure issues shouldn't take down the app.
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        max_requests: int = 20,
        window_seconds: int = 60,
        namespace: str = "ratelimit",
    ):
        self._client = redis_client
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.namespace = namespace

    def is_allowed(self, user_id: str) -> tuple[bool, dict]:
        """
        Returns (allowed, info). info contains 'remaining' and 'reset_in'
        (seconds until the current window resets), useful for user-facing
        messages.
        """
        bucket = int(time.time() // self.window_seconds)
        key = f"{self.namespace}:{user_id}:{bucket}"

        try:
            current = self._client.incr(key)
            if current == 1:
                self._client.expire(key, self.window_seconds)
        except redis.exceptions.RedisError as e:
            print(f"[ratelimit] Redis unavailable, failing open: {e}")
            return True, {"remaining": None, "reset_in": None}

        remaining = max(0, self.max_requests - current)
        reset_in = self.window_seconds - int(time.time() % self.window_seconds)

        return current <= self.max_requests, {"remaining": remaining, "reset_in": reset_in}