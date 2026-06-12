"""Tests for memory-mapped vector storage."""

import numpy as np
import pytest

from utils.mmap_store import MmapVectorStore


def test_add_and_get(tmp_path):
    s = MmapVectorStore(str(tmp_path), dim=8, capacity=10)
    vec = np.arange(8, dtype=np.float32)
    s.add("v1", vec)
    got = s.get("v1")
    assert got is not None
    assert np.allclose(got, vec)
    s.close()


def test_get_missing_returns_none(tmp_path):
    s = MmapVectorStore(str(tmp_path), dim=4, capacity=4)
    assert s.get("nope") is None
    s.close()


def test_dimension_mismatch_raises(tmp_path):
    s = MmapVectorStore(str(tmp_path), dim=4, capacity=4)
    with pytest.raises(ValueError):
        s.add("v1", [1, 2, 3])  # wrong dim
    s.close()


def test_grows_past_capacity(tmp_path):
    s = MmapVectorStore(str(tmp_path), dim=4, capacity=2)
    for i in range(10):
        s.add(f"v{i}", np.random.rand(4))
    assert len(s) == 10
    assert s.capacity >= 10
    s.close()


def test_delete_reclaims_row(tmp_path):
    s = MmapVectorStore(str(tmp_path), dim=4, capacity=8)
    for i in range(5):
        s.add(f"v{i}", np.random.rand(4))
    assert s.delete("v2") is True
    assert s.get("v2") is None
    assert 2 in s.free_rows
    # next add reuses the freed row
    s.add("v_new", np.random.rand(4))
    assert s.id_to_row["v_new"] == 2
    s.close()


def test_persistence_across_reopen(tmp_path):
    s = MmapVectorStore(str(tmp_path), dim=4, capacity=4)
    for i in range(6):
        s.add(f"v{i}", np.full(4, i, dtype=np.float32))
    s.close()

    s2 = MmapVectorStore(str(tmp_path))
    assert len(s2) == 6
    assert s2.dim == 4
    assert np.allclose(s2.get("v3"), np.full(4, 3))
    s2.close()


def test_matrix_view(tmp_path):
    s = MmapVectorStore(str(tmp_path), dim=4, capacity=8)
    for i in range(5):
        s.add(f"v{i}", np.random.rand(4))
    s.delete("v0")
    matrix, ids = s.matrix_view()
    assert matrix.shape == (4, 4)
    assert "v0" not in ids
    s.close()


def test_add_batch(tmp_path):
    s = MmapVectorStore(str(tmp_path), dim=4, capacity=2)
    items = [(f"v{i}", np.random.rand(4)) for i in range(20)]
    written = s.add_batch(items)
    assert written == 20
    assert len(s) == 20
    s.close()
