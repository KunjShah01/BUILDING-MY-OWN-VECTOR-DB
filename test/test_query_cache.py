"""Tests for the query cache service."""
from __future__ import annotations
from services.query_cache import QueryCache


class TestQueryCache:
    def setup_method(self):
        self.cache = QueryCache(l1_maxsize=100, l1_ttl=60)

    def test_get_miss(self):
        result = self.cache.get("coll1", [0.1, 0.2], 10)
        assert result is None

    def test_set_and_get(self):
        self.cache.set("coll1", [0.1, 0.2], 10, {"results": [1, 2, 3]})
        result = self.cache.get("coll1", [0.1, 0.2], 10)
        assert result == {"results": [1, 2, 3]}

    def test_different_key_produces_miss(self):
        self.cache.set("coll1", [0.1, 0.2], 10, {"results": [1]})
        result = self.cache.get("coll1", [0.3, 0.4], 10)
        assert result is None

    def test_flush_clears(self):
        self.cache.set("coll1", [0.1], 5, {"x": 1})
        self.cache.flush()
        assert self.cache.stats()["size"] == 0

    def test_stats(self):
        self.cache.get("c1", [0.1], 5)
        self.cache.set("c1", [0.1], 5, {"x": 1})
        self.cache.get("c1", [0.1], 5)
        stats = self.cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_ratio"] == 0.5
