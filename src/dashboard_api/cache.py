"""Singleton TTL-cache used by every dashboard route.

Heavy operations (loading the analyzer, fetching benchmark
prices, running the Monte Carlo) are cached so a navigation
between tabs is cheap. The cache invalidates after
``default_ttl_seconds`` or on an explicit POST /api/refresh.
"""

import threading
import time
from typing import Any, Callable, Dict, Optional


default_ttl_seconds = 600.0


class SnapshotCache:
    """Thread-safe TTL cache for per-key compute callables.

    Attributes:
        ttl_seconds: Expiry window for each entry.
    """

    def __init__(self, ttl_seconds: float = default_ttl_seconds):
        self.ttl_seconds = ttl_seconds
        self._store: Dict[str, Any] = {}
        self._stamps: Dict[str, float] = {}
        self._lock = threading.Lock()

    def get_or_compute(
            self, key: str, compute: Callable[[], Any]) -> Any:
        """Return cached value for `key`, otherwise compute it.

        Args:
            key: Cache key.
            compute: Zero-arg callable that produces the value
                when the cache misses or has expired.

        Returns:
            The cached or freshly-computed value.
        """
        now = time.monotonic()
        with self._lock:
            stamp = self._stamps.get(key)
            if (stamp is not None
                    and (now - stamp) < self.ttl_seconds
                    and key in self._store):
                return self._store[key]
        # Compute outside the lock so concurrent keys don't block.
        value = compute()
        with self._lock:
            self._store[key] = value
            self._stamps[key] = time.monotonic()
        return value

    def invalidate(self, key: Optional[str] = None) -> None:
        """Drop a key (or the whole cache when `key` is None)."""
        with self._lock:
            if key is None:
                self._store.clear()
                self._stamps.clear()
            else:
                self._store.pop(key, None)
                self._stamps.pop(key, None)
