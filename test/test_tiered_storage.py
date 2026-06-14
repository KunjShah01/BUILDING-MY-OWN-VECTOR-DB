"""Tests for the tiered storage service."""
from __future__ import annotations
from services.tiered_storage import TierManager


class TestTierManager:
    def setup_method(self):
        self.mgr = TierManager()

    def test_add_and_track(self):
        self.mgr.add_vector("v1", "coll1", [0.1, 0.2], {"text": "hi"})
        info = self.mgr.get_tier_info("coll1")
        assert info["hot"] == 1

    def test_remove(self):
        self.mgr.add_vector("v1", "coll1", [0.1, 0.2])
        self.mgr.remove_vector("v1")
        info = self.mgr.get_tier_info("coll1")
        assert info["hot"] == 0

    def test_promote(self):
        self.mgr.add_vector("v1", "coll1", [0.1, 0.2])
        self.mgr.promote("v1", "warm")
        info = self.mgr.get_tier_info("coll1")
        assert info["warm"] == 1
