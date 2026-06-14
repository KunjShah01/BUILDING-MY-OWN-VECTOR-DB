"""Tests for materialized views service."""
from __future__ import annotations
from services.materialized_views import MaterializedViewService


class TestMaterializedViewService:
    def setup_method(self):
        self.svc = MaterializedViewService()

    def test_create_and_get(self):
        view_id = self.svc.create_view("top docs", "coll1", [0.1, 0.2], k=50, refresh_interval=600)
        view = self.svc.get_view(view_id)
        assert view is not None
        assert view.name == "top docs"
        assert view.k == 50

    def test_list_views(self):
        self.svc.create_view("v1", "coll1", [0.1], k=10)
        self.svc.create_view("v2", "coll2", [0.2], k=20)
        views = self.svc.list_views()
        assert len(views) == 2

    def test_delete_view(self):
        view_id = self.svc.create_view("v1", "coll1", [0.1], k=10)
        self.svc.delete_view(view_id)
        assert self.svc.get_view(view_id) is None
