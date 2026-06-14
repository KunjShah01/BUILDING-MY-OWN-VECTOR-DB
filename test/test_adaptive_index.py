"""Tests for adaptive index selector."""
from __future__ import annotations
from services.adaptive_index import AdaptiveIndexSelector, QueryProfile


class TestAdaptiveIndexSelector:
    def setup_method(self):
        self.selector = AdaptiveIndexSelector()

    def test_default_method(self):
        method = self.selector.select_method("new_coll")
        assert method == "hnsw"

    def test_records_and_selects(self):
        self.selector.record_query("coll1", "hnsw", 5.0, 0.95)
        self.selector.record_query("coll1", "hnsw", 6.0, 0.94)
        self.selector.record_query("coll1", "ivf", 20.0, 0.85)
        method = self.selector.select_method("coll1")
        assert method == "hnsw"

    def test_report(self):
        self.selector.record_query("coll1", "hnsw", 5.0, 0.95)
        report = self.selector.get_performance_report("coll1")
        assert "hnsw" in report["methods"]
        assert report["methods"]["hnsw"]["sample_count"] == 1
