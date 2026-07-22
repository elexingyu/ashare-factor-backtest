"""Memory-bounded cache for evaluated expression subtrees."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    evictions: int
    current_bytes: int
    peak_bytes: int

    def as_dict(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "current_bytes": self.current_bytes,
            "peak_bytes": self.peak_bytes,
        }


class PanelLRU:
    """An LRU cache whose limit is measured in DataFrame bytes, not item count."""

    def __init__(self, max_bytes: int) -> None:
        if isinstance(max_bytes, bool) or max_bytes <= 0:
            raise ValueError("cache max_bytes must be a positive integer")
        self.max_bytes = int(max_bytes)
        self._items: OrderedDict[str, tuple[pd.DataFrame, int]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._current_bytes = 0
        self._peak_bytes = 0

    def get(self, key: str) -> pd.DataFrame | None:
        item = self._items.pop(key, None)
        if item is None:
            self._misses += 1
            return None
        self._hits += 1
        self._items[key] = item
        return item[0]

    def put(self, key: str, value: pd.DataFrame) -> None:
        size = int(value.memory_usage(index=True, deep=True).sum())
        previous = self._items.pop(key, None)
        if previous is not None:
            self._current_bytes -= previous[1]
        if size > self.max_bytes:
            return
        while self._items and self._current_bytes + size > self.max_bytes:
            _, (_, removed_size) = self._items.popitem(last=False)
            self._current_bytes -= removed_size
            self._evictions += 1
        self._items[key] = (value, size)
        self._current_bytes += size
        self._peak_bytes = max(self._peak_bytes, self._current_bytes)

    def clear(self, *, reset_stats: bool = True) -> None:
        self._items.clear()
        self._current_bytes = 0
        if reset_stats:
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._peak_bytes = 0

    def stats(self) -> CacheStats:
        return CacheStats(
            self._hits,
            self._misses,
            self._evictions,
            self._current_bytes,
            self._peak_bytes,
        )
