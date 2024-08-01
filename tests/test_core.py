import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from inference_cache import (
    CacheError,
    InferenceCache,
    UnhashablePayloadError,
    stable_hash,
)


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def cache(clock):
    return InferenceCache(ttl_seconds=3600, max_entries=100, clock=clock)


def test_stable_hash_is_deterministic_and_order_insensitive():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


def test_unserializable_payload_rejected():
    with pytest.raises(UnhashablePayloadError):
        stable_hash({"fn": lambda x: x})


def test_store_then_lookup_hits(cache):
    payload = {"prompt": "hello", "model": "m3"}
    key = cache.store(payload, {"answer": 42})
    hit = cache.lookup(payload)
    assert hit is not None
    assert hit.value == {"answer": 42}
    assert hit.key == key


def test_miss_returns_none(cache):
    assert cache.lookup({"prompt": "unknown"}) is None


def test_ttl_expiry_evicts_entry(cache):
    payload = {"q": "x"}
    cache.store(payload, "result")
    cache.clock.advance(3599)
    assert cache.lookup(payload) is not None
    cache.clock.advance(2)
    assert cache.lookup(payload) is None


def test_cached_call_uses_inference_only_once(cache):
    calls = {"n": 0}

    def fake_model(payload):
        calls["n"] += 1
        return f"reply-{payload['id']}"

    request = {"id": 7}
    first, was_cached = cache.cached_call(request, fake_model)
    second, cached_again = cache.cached_call(request, fake_model)
    assert first == "reply-7"
    assert second == "reply-7"
    assert was_cached is False
    assert cached_again is True
    assert calls["n"] == 1


def test_invalidate_removes_specific_entry(cache):
    payload = {"k": "v"}
    cache.store(payload, 1)
    assert cache.invalidate(payload) is True
    assert cache.lookup(payload) is None
    assert cache.invalidate(payload) is False


def test_lru_eviction_respects_max_entries(clock):
    tiny = InferenceCache(ttl_seconds=9999, max_entries=3, clock=clock)
    for i in range(5):
        tiny.store({"i": i}, i)
    stats = tiny.stats()
    assert stats["entries"] == 3


def test_clear_wipes_all(cache):
    for i in range(4):
        cache.store({"i": i}, i)
    assert cache.clear() == 4
    assert cache.stats()["entries"] == 0


def test_hit_ratio_tracked(cache):
    cache.store({"hit": True}, "yes")
    cache.lookup({"hit": True})
    cache.lookup({"miss": True})
    ratio = cache.stats()["hit_ratio"]
    assert ratio == 0.5


def test_file_backed_persistence(tmp_path):
    db_path = tmp_path / "cache.db"
    first = InferenceCache(store_path=db_path, ttl_seconds=999)
    first.store({"q": "persist"}, "stored-value")
    first.close()

    reopened = InferenceCache(store_path=db_path, ttl_seconds=999)
    hit = reopened.lookup({"q": "persist"})
    assert hit is not None
    assert hit.value == "stored-value"


def test_invalid_config_rejected():
    with pytest.raises(CacheError):
        InferenceCache(ttl_seconds=0)
    with pytest.raises(CacheError):
        InferenceCache(max_entries=0)
