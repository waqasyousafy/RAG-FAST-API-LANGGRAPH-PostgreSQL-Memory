"""
Response Caching Layer (Redis-backed)
Persistent, shared cache with native TTL for LLM response deduplication.
"""

import hashlib
import json
from typing import Optional
import redis


class ResponseCache:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl_seconds: int = 300,
        namespace: str = "llm_cache",
    ):
        self.ttl = ttl_seconds
        self.namespace = namespace
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def _make_key(self, query: str) -> str:
        normalized = query.lower().strip()
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        return f"{self.namespace}:response:{digest}"

    def get(self, query: str) -> Optional[str]:
        key = self._make_key(query)
        try:
            raw = self._client.get(key)
        except redis.exceptions.RedisError as e:
            print(f"[cache] Redis unavailable, skipping cache: {e}")
            return None

        if raw is not None:
            self._client.incr(f"{self.namespace}:stats:hits")
            return json.loads(raw)["response"]

        self._client.incr(f"{self.namespace}:stats:misses")
        return None

    def set(self, query: str, response: str) -> None:
        key = self._make_key(query)
        payload = json.dumps({"response": response, "query": query})
        try:
            self._client.setex(key, self.ttl, payload)
        except redis.exceptions.RedisError as e:
            print(f"[cache] Redis unavailable, skipping cache write: {e}")

    def flush(self) -> None:
        """Call this after re-ingesting documents so stale answers don't survive on TTL alone."""
        for key in self._client.scan_iter(f"{self.namespace}:response:*"):
            self._client.delete(key)

    @property
    def stats(self) -> dict:
        hits = int(self._client.get(f"{self.namespace}:stats:hits") or 0)
        misses = int(self._client.get(f"{self.namespace}:stats:misses") or 0)
        total = hits + misses
        hit_rate = hits / total if total > 0 else 0.0
        cached_entries = sum(1 for _ in self._client.scan_iter(f"{self.namespace}:response:*"))
        return {
            "hits": hits,
            "misses": misses,
            "hit_rate": f"{hit_rate:.1%}",
            "cached_entries": cached_entries,
        }