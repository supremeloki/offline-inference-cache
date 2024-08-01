from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

ValueT = TypeVar("ValueT")

DEFAULT_TTL_SECONDS = 86_400
DEFAULT_MAX_ENTRIES = 10_000


class CacheError(Exception):
    pass


class UnhashablePayloadError(CacheError):
    pass


@dataclass(frozen=True)
class CacheHit(Generic[ValueT]):
    value: ValueT
    age_seconds: float
    key: str

    @property
    def is_fresh(self) -> bool:
        return self.age_seconds >= 0


def stable_hash(payload: Any) -> str:
    try:
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                               default=_reject_unsupported)
    except TypeError as exc:
        raise UnhashablePayloadError(f"payload not JSON-serializable: {exc}") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _reject_unsupported(value: Any) -> str:
    raise TypeError(f"{type(value).__name__} is not supported")


class InferenceCache(Generic[ValueT]):
    def __init__(
        self,
        store_path: Path | None = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if ttl_seconds <= 0 or max_entries < 1:
            raise CacheError("ttl must be positive and max_entries >= 1")
        self.clock = clock
        self._clock = clock
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._conn = sqlite3.connect(str(store_path or ":memory:"))
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                cache_key TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        self._hits = 0
        self._misses = 0

    def lookup(self, prompt_payload: dict[str, Any]) -> CacheHit[ValueT] | None:
        key = stable_hash(prompt_payload)
        now = self._clock()
        row = self._conn.execute(
            "SELECT payload, created_at FROM entries WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None:
            self._misses += 1
            return None
        payload_text, created_at = row
        if now - created_at >= self._ttl:
            self._conn.execute("DELETE FROM entries WHERE cache_key = ?", (key,))
            self._conn.commit()
            self._misses += 1
            return None
        self._hits += 1
        return CacheHit(value=json.loads(payload_text), age_seconds=now - created_at, key=key)

    def store(self, prompt_payload: dict[str, Any], result: ValueT) -> str:
        key = stable_hash(prompt_payload)
        self._conn.execute(
            "INSERT OR REPLACE INTO entries (cache_key, payload, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(result, ensure_ascii=False), self._clock()),
        )
        self._conn.commit()
        self._evict_if_needed()
        return key

    def cached_call(
        self,
        prompt_payload: dict[str, Any],
        inference_fn: Callable[[dict[str, Any]], ValueT],
    ) -> tuple[ValueT, bool]:
        hit = self.lookup(prompt_payload)
        if hit is not None:
            return hit.value, True
        fresh = inference_fn(prompt_payload)
        self.store(prompt_payload, fresh)
        return fresh, False

    def invalidate(self, prompt_payload: dict[str, Any]) -> bool:
        key = stable_hash(prompt_payload)
        cursor = self._conn.execute("DELETE FROM entries WHERE cache_key = ?", (key,))
        self._conn.commit()
        return cursor.rowcount > 0

    def clear(self) -> int:
        removed = self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        self._conn.execute("DELETE FROM entries")
        self._conn.commit()
        return removed

    def stats(self) -> dict[str, int]:
        total_rows = self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        return {
            "entries": total_rows,
            "hits": self._hits,
            "misses": self._misses,
            "hit_ratio": round(self._hits / (self._hits + self._misses), 4)
            if (self._hits + self._misses) else 0.0,
        }

    def _evict_if_needed(self) -> None:
        overflow = self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0] - self._max_entries
        if overflow <= 0:
            return
        oldest = self._conn.execute(
            "SELECT cache_key FROM entries ORDER BY created_at ASC LIMIT ?",
            (overflow,),
        ).fetchall()
        self._conn.executemany(
            "DELETE FROM entries WHERE cache_key = ?",
            [(row[0],) for row in oldest],
        )

    def close(self) -> None:
        self._conn.close()
