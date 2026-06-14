"""Tests for the memories module — MemoryService unit tests."""

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


class TestMemoryService:
    def test_add_memory(self, service, mock_db):
        with patch("services.memory_service.embed_text", return_value=[0.1, 0.2, 0.3]):
            with patch("services.memory_service.uuid.uuid4", return_value="test-id"):
                result = service.add_memory("user1", "hello world", ["greeting"])

        assert result["success"] is True
        assert result["memory"]["text"] == "hello world"
        assert result["memory"]["memory_id"] == "test-id"
        assert result["memory"]["categories"] == ["greeting"]
        assert result["memory"]["user_id"] == "user1"
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

    def test_get_memory_not_found(self, service):
        result = service.get_memory("nonexistent")
        assert result is None

    def test_get_memory_found(self, service, mock_db):
        mock_mem = MagicMock()
        mock_mem.memory_id = "m1"
        mock_mem.user_id = "u1"
        mock_mem.text = "test"
        mock_mem.categories = ["cat"]
        mock_mem.metadata = {}
        mock_mem.created_at = datetime.now(timezone.utc)
        mock_mem.updated_at = datetime.now(timezone.utc)
        mock_db.query.return_value.filter.return_value.first.return_value = mock_mem

        result = service.get_memory("m1")
        assert result is not None
        assert result.memory_id == "m1"

    def test_get_memories(self, service, mock_db):
        mock_mem = MagicMock()
        mock_mem.memory_id = "m1"
        mock_mem.user_id = "u1"
        mock_mem.text = "test"
        mock_mem.categories = ["cat"]
        mock_mem.metadata = {}
        mock_mem.created_at = datetime.now(timezone.utc)
        mock_mem.updated_at = datetime.now(timezone.utc)
        mock_db.query.return_value.filter.return_value.count.return_value = 1
        mock_db.query.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [mock_mem]

        result = service.get_memories("u1")
        assert result["success"] is True
        assert len(result["memories"]) == 1
        assert result["total"] == 1

    def test_delete_memory_not_found(self, service, mock_db):
        result = service.delete_memory("nonexistent", "user1")
        assert result["success"] is False
        assert "not found" in result["message"].lower()

    def test_delete_memory_success(self, service, mock_db):
        mock_mem = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_mem

        result = service.delete_memory("m1", "user1")
        assert result["success"] is True
        mock_db.delete.assert_called_once_with(mock_mem)
        mock_db.commit.assert_called_once()

    def test_update_memory_not_found(self, service, mock_db):
        mock_db.query.return_value.filter.return_value.first.return_value = None
        result = service.update_memory("nonexistent", "user1", text="new text")
        assert result["success"] is False

    def test_update_memory_success(self, service, mock_db):
        mock_mem = MagicMock()
        mock_mem.memory_id = "m1"
        mock_mem.user_id = "u1"
        mock_mem.text = "old"
        mock_mem.categories = ["old"]
        mock_mem.metadata = {}
        mock_mem.created_at = datetime.now(timezone.utc)
        mock_mem.updated_at = datetime.now(timezone.utc)
        mock_db.query.return_value.filter.return_value.first.return_value = mock_mem

        with patch("services.memory_service.embed_text", return_value=[0.5, 0.6]):
            result = service.update_memory("m1", "u1", text="updated", categories=["new"])

        assert result["success"] is True
        assert mock_mem.text == "updated"
        assert mock_mem.categories == ["new"]
        assert mock_mem.metadata == {}

    def test_search_memories(self, service, mock_db):
        with patch("services.memory_service.embed_text", return_value=[0.1, 0.2, 0.3]):
            mock_row = MagicMock()
            mock_row.memory_id = "m1"
            mock_row.text = "test memory"
            mock_row.categories = ["test"]
            mock_row.metadata = {}
            mock_row.similarity = 0.95
            mock_row.created_at = "2026-01-01"
            mock_row.updated_at = "2026-01-01"
            mock_db.execute.return_value.fetchall.return_value = [mock_row]

            result = service.search_memories("test query", "user1")

        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["results"][0]["memory_id"] == "m1"
        assert result["results"][0]["distance"] == 0.05

    def test_consolidate_single_memory(self, service, mock_db):
        mock_db.query.return_value.filter.return_value.all.return_value = [MagicMock()]
        result = service.consolidate("user1")
        assert result["merged"] == 0
        assert result["deleted"] == 0
        assert "Less than 2" in result["message"]

    def test_chat(self, service, mock_db):
        with patch("services.memory_service.embed_text", return_value=[0.1, 0.2, 0.3]):
            with patch("services.rag_service.openai_chat_completion", return_value="AI response"):
                mock_row = MagicMock()
                mock_row.memory_id = "m1"
                mock_row.text = "user likes pizza"
                mock_row.categories = ["food"]
                mock_row.metadata = {}
                mock_row.similarity = 0.95
                mock_row.created_at = "2026-01-01"
                mock_row.updated_at = "2026-01-01"
                mock_db.execute.return_value.fetchall.return_value = [mock_row]

                result = service.chat("what do i like", "user1")

        assert result["success"] is True
        assert result["response"] == "AI response"
        assert result["memories_retrieved"] == 1
