"""Two-tier query result cache: L1 in-memory LRU, L2 optional Redis."""
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from cachetools import TTLCache

logger = logging.getLogger(__name__)

class QueryCache:
    def __init__(self, l1_maxsize: int = 10000, l1_ttl: int = 60):
        self._l1: TTLCache = TTLCache(maxsize=l1_maxsize, ttl=l1_ttl)
        self._hits = 0
        self._misses = 0
        self._l2_enabled = False
        self._redis_client = None

    def _make_key(self, collection_id: str, query_embedding: List[float],
                  k: int, filters: Optional[Dict] = None) -> str:
        raw = f"{collection_id}:{hashlib.md5(json.dumps(query_embedding, sort_keys=True).encode()).hexdigest()}:{k}:{json.dumps(filters or {}, sort_keys=True)}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, collection_id: str, query_embedding: List[float],
            k: int, filters: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        key = self._make_key(collection_id, query_embedding, k, filters)
        result = self._l1.get(key)
        if result is not None:
            self._hits += 1
            return result
        if self._l2_enabled and self._redis_client:
            try:
                val = self._redis_client.get(f"cache:{key}")
                if val:
                    self._hits += 1
                    result = json.loads(val)
                    self._l1[key] = result
                    return result
            except Exception:
                pass
        self._misses += 1
        return None

    def set(self, collection_id: str, query_embedding: List[float],
            k: int, value: Dict[str, Any], filters: Optional[Dict] = None):
        key = self._make_key(collection_id, query_embedding, k, filters)
        self._l1[key] = value
        if self._l2_enabled and self._redis_client:
            try:
                self._redis_client.setex(f"cache:{key}", 300, json.dumps(value, default=str))
            except Exception:
                pass

    def flush(self):
        self._l1.clear()
        logger.info("Query cache flushed")

    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "size": len(self._l1),
            "maxsize": self._l1.maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_ratio": round(self._hits / total, 4) if total else 0.0,
            "l1_ttl": self._l1.ttl,
            "l2_enabled": self._l2_enabled,
        }

    def enable_l2(self, redis_client):
        self._l2_enabled = True
        self._redis_client = redis_client

# Singleton
query_cache = QueryCache()
