# offline-inference-cache

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A local, dependency-free response cache for LLM inference: SHA-256 prompt keys, TTL expiry, LRU eviction, file-backed persistence — so identical prompts never pay for inference twice.

## 🚀 Overview

Running models locally means every token costs *your* hardware. When users repeat questions (they always do), `offline-inference-cache` short-circuits the model call: the full request payload (prompt + params + model id) is canonicalized into a stable SHA-256 key, and the response is served from a SQLite store until it expires or gets evicted.

## ✨ Features

- **Stable keys:** `json.dumps(sort_keys=True)` + SHA-256 — key order and unicode don't change the hash
- **Unhashable payloads rejected:** lambdas and open files fail fast with `UnhashablePayloadError`, never silently
- **TTL expiry:** stale entries deleted lazily on lookup; injectable clock makes time travel trivial in tests
- **LRU eviction:** `max_entries` cap evicts oldest first
- **`cached_call` helper:** one line wraps any inference function with cache semantics
- **File-backed or in-memory:** pass a path for persistence across restarts; omit it for RAM-only
- **Hit-ratio stats:** hits / misses / ratio for observability
- **Zero dependencies**

## 🚧 Structure

```
offline-inference-cache/
├── src/inference_cache/
│   ├── __init__.py
│   └── core.py
├── tests/
│   └── test_core.py
├── README.md
└── pyproject.toml
```

## 📦 Installation

```bash
git clone https://github.com/supremeloki/offline-inference-cache.git
cd offline-inference-cache
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## 📋 Requirements

- Python 3.11+
- No runtime dependencies

## 🏃 Quick Start

```python
from pathlib import Path
from inference_cache import InferenceCache

cache = InferenceCache(store_path=Path("llm_cache.db"), ttl_seconds=86_400)

def ask_model(payload: dict) -> str:
    return expensive_local_inference(payload)

answer, was_cached = cache.cached_call(
    {"model": "llama3-8b", "prompt": "سلام!", "temp": 0.7},
    ask_model,
)
```

### Manual control

```python
key = cache.store({"prompt": "x"}, "result")
hit = cache.lookup({"prompt": "x"})
cache.invalidate({"prompt": "x"})
print(cache.stats())
```

## 🔧 Error Handling

```text
CacheError
└── UnhashablePayloadError   # payload contains non-JSON values
```

Expired lookups behave as misses — they never raise.

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📝 Code Quality

- Full type hints (`X | None` style), generic `InferenceCache[ValueT]`
- Zero comments — names carry the meaning
- Injectable clock → fully deterministic tests

## 📄 License

MIT — see [LICENSE](LICENSE).

## 👤 Author

**Kooroush Masoumi**

---

⭐ Star this repo if you find it useful!
