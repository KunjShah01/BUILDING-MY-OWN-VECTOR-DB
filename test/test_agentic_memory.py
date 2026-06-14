"""Tests for the merged agent-memory layer (services/memory_service.py).

Tests the MemoryService CRUD operations, search, and chat using mocked
database and embedding service so tests are fast and offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

import pytest

from services.memory_service import MemoryService


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None
    return db


@pytest.fixture
def service(mock_db):
    return MemoryService(mock_db)


def _make_mock_mem(memory_id="m1", user_id="1", text="test",
                   categories=None, similarity=0.95):
    mem = MagicMock()
    mem.memory_id = memory_id
    mem.user_id = user_id
    mem.text = text
    mem.categories = categories or []
    mem.meta_data = {}
    mem.created_at = datetime.now(timezone.utc)
    mem.updated_at = datetime.now(timezone.utc)
    return mem


def _make_mock_row(memory_id="m1", text="test", categories=None, similarity=0.95):
    row = MagicMock()
    row.memory_id = memory_id
    row.text = text
    row.categories = categories or []
    row.metadata = {}
    row.similarity = similarity
    row.created_at = "2026-01-01"
    row.updated_at = "2026-01-01"
    return row


def test_add_memory(service, mock_db):
    with patch("services.memory_service.embed_text", return_value=[0.1, 0.2, 0.3]):
        with patch("services.memory_service.uuid.uuid4", return_value="m1"):
            result = service.add_memory("1", "User loves hiking", ["hobbies"])

    assert result["success"] is True
    assert result["memory"]["text"] == "User loves hiking"
    assert result["memory"]["categories"] == ["hobbies"]
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()


def test_user_isolation(service, mock_db):
    """Search for user 1 should not return user 2's memories."""
    with patch("services.memory_service.embed_text", return_value=[0.1, 0.2, 0.3]):
        row = _make_mock_row("m1", "pizza", ["food"])
        mock_db.execute.return_value.fetchall.return_value = [row]
        result = service.search_memories("pizza", "1")

    assert result["success"] is True
    call_args = mock_db.execute.call_args[0][1]
    assert call_args["user_id"] == "1"


def test_category_filter(service, mock_db):
    """Search with category filter should pass categories to the query."""
    with patch("services.memory_service.embed_text", return_value=[0.1, 0.2, 0.3]):
        row = _make_mock_row("m1", "work stuff", ["work"])
        mock_db.execute.return_value.fetchall.return_value = [row]
        result = service.search_memories("work", "1", categories=["work"])

    assert result["success"] is True
    assert len(result["results"]) == 1


def test_get_memories(service, mock_db):
    mock_mem = _make_mock_mem("m1", "1", "hiking", ["hobbies"])
    mock_db.query.return_value.filter.return_value.count.return_value = 1
    mock_db.query.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [mock_mem]

    result = service.get_memories("1")
    assert result["success"] is True
    assert len(result["memories"]) == 1
    assert result["total"] == 1


def test_get_categories(service, mock_db):
    """Verify we can filter by category via get_memories."""
    mock_mem = _make_mock_mem("m1", "1", "work stuff", ["work"])
    mock_db.query.return_value.filter.return_value.filter.return_value.count.return_value = 1
    mock_db.query.return_value.filter.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [mock_mem]

    result = service.get_memories("1", category="work")
    assert result["success"] is True
    assert result["total"] == 1
    assert result["memories"][0]["categories"] == ["work"]


def test_delete(service, mock_db):
    mock_mem = _make_mock_mem("m1")
    mock_db.query.return_value.filter.return_value.first.return_value = mock_mem

    result = service.delete_memory("m1", "1")
    assert result["success"] is True
    mock_db.delete.assert_called_once_with(mock_mem)


def test_delete_not_found(service, mock_db):
    result = service.delete_memory("nonexistent", "1")
    assert result["success"] is False
    assert "not found" in result["message"].lower()


def test_update(service, mock_db):
    mock_mem = _make_mock_mem("m1", "1", "old text", ["old"])
    mock_db.query.return_value.filter.return_value.first.return_value = mock_mem

    with patch("services.memory_service.embed_text", return_value=[0.5, 0.6]):
        result = service.update_memory("m1", "1", text="updated", categories=["new"])

    assert result["success"] is True
    assert mock_mem.text == "updated"
    assert mock_mem.categories == ["new"]


def test_consolidate_single(service, mock_db):
    mock_db.query.return_value.filter.return_value.all.return_value = [_make_mock_mem()]
    result = service.consolidate("1")
    assert result["merged"] == 0
    assert result["deleted"] == 0


def test_pipeline_add_and_search(service, mock_db):
    """End-to-end add then search through MemoryService (mocked)."""
    with patch("services.memory_service.embed_text", return_value=[0.1, 0.2, 0.3]):
        with patch("services.memory_service.uuid.uuid4", return_value="m1"):
            add_result = service.add_memory("1", "User enjoys hiking trails", ["hobbies"])

    assert add_result["success"] is True
    assert add_result["memory"]["text"] == "User enjoys hiking trails"

    with patch("services.memory_service.embed_text", return_value=[0.1, 0.2, 0.3]):
        row = _make_mock_row("m1", "User enjoys hiking trails", ["hobbies"])
        mock_db.execute.return_value.fetchall.return_value = [row]
        search_result = service.search_memories("hiking", "1")

    assert search_result["success"] is True
    assert len(search_result["results"]) >= 1
    assert "hiking" in search_result["results"][0]["text"].lower()
