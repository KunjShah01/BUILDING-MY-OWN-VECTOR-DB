"""Tests for the streaming search service."""
from __future__ import annotations
from services.streaming_search import StreamingSearchService


class TestStreamingSearch:
    def setup_method(self):
        self.svc = StreamingSearchService()

    def test_register_and_unregister(self):
        sub_id = self.svc.register("coll1", [0.1, 0.2], threshold=0.9)
        assert sub_id is not None
        sub = self.svc.get_subscription(sub_id)
        assert sub is not None
        assert sub.collection_id == "coll1"
        self.svc.unregister(sub_id)
        assert self.svc.get_subscription(sub_id) is None

    def test_evaluate_notifies_on_match(self):
        sub_id = self.svc.register("coll1", [1.0, 0.0], threshold=0.8)
        self.svc.evaluate_and_notify("coll1", "v1", [0.9, 0.1], {"text": "test"})
        sub = self.svc.get_subscription(sub_id)
        assert len(sub.queue) == 1
        assert sub.queue[0]["vector_id"] == "v1"

    def test_evaluate_ignores_low_similarity(self):
        sub_id = self.svc.register("coll1", [1.0, 0.0], threshold=0.99)
        self.svc.evaluate_and_notify("coll1", "v1", [-1.0, 0.0])
        sub = self.svc.get_subscription(sub_id)
        assert len(sub.queue) == 0

    def test_different_collection_ignored(self):
        sub_id = self.svc.register("coll1", [1.0, 0.0], threshold=0.0)
        self.svc.evaluate_and_notify("coll2", "v1", [1.0, 0.0])
        sub = self.svc.get_subscription(sub_id)
        assert len(sub.queue) == 0
