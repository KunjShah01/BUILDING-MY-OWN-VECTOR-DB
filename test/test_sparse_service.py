"""Tests for the sparse vector (SPLADE) service."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from services.sparse_service import SparseService


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None
    db.query.return_value.filter.return_value.all.return_value = []
    return db


@pytest.fixture
def service(mock_db):
    return SparseService(mock_db)


class TestSparseService:
    def test_create_sparse_vector(self, service, mock_db):
        with patch("services.sparse_service._splade_encode", return_value=[{"0": 0.5, "1": 0.3}]):
            result = service.create_sparse_vector("coll1", "doc1", "hello world")
        assert result == "doc1"
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

    def test_search_sparse(self, service, mock_db):
        mock_row = MagicMock()
        mock_row.doc_id = "doc1"
        mock_row.text = "hello"
        mock_row.sparse_embedding = {"0": 0.5}
        mock_row.created_at = None
        mock_db.query.return_value.filter.return_value.all.return_value = [mock_row]
        with patch("services.sparse_service._splade_encode", return_value=[{"0": 0.5}]):
            result = service.search_sparse("coll1", "hello", k=5)
        assert len(result) == 1
        assert result[0]["doc_id"] == "doc1"

    def test_delete_sparse(self, service, mock_db):
        mock_row = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_row
        result = service.delete_sparse_vector("coll1", "doc1")
        assert result is True
        mock_db.delete.assert_called_once_with(mock_row)
        mock_db.commit.assert_called_once()
