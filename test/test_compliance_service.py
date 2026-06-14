"""Tests for the compliance service."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from services.compliance_service import ComplianceService


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None
    db.query.return_value.filter.return_value.count.return_value = 0
    db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
    return db


@pytest.fixture
def svc(mock_db):
    return ComplianceService(mock_db)


class TestComplianceService:
    def test_set_retention_policy(self, svc, mock_db):
        result = svc.set_retention_policy("coll1", ttl_days=90, archive_after_days=180)
        assert result["success"] is True
        assert result["policy"]["ttl_days"] == 90
        mock_db.add.assert_called_once()

    def test_get_retention_policy_not_found(self, svc):
        result = svc.get_retention_policy("nonexistent")
        assert result["policy"] is None

    def test_set_query_budget(self, svc, mock_db):
        result = svc.set_query_budget("tenant1", max_vectors_scanned=50000, max_ef_search=400)
        assert result["success"] is True
        assert result["budget"]["max_vectors_scanned"] == 50000

    def test_generate_report(self, svc, mock_db):
        with patch("utils.audit_log.query_logs", return_value={"logs": []}):
            with patch("config.settings.get_settings") as mock_settings:
                mock_settings.return_value.ENCRYPTION_KEY = "test-key"
                result = svc.generate_report("SOC2", "tenant1")
        assert result["success"] is True
        assert result["report"]["report_type"] == "SOC2"
