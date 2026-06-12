"""Tests for HNSW tombstone soft-delete and background compaction."""

import numpy as np

from utils.hnsw_index import HNSWIndex
from utils.compaction import BackgroundCompactor


def _rand(dim=8):
    return list(np.random.rand(dim).astype(float))


def _build(n=30):
    idx = HNSWIndex()
    for i in range(n):
        idx.insert(_rand(), f"v{i}", {"n": i})
    return idx


def test_soft_delete_tombstones_node():
    idx = _build()
    assert idx.delete("v0") is True
    assert "v0" in idx.deleted
    # node still in graph (preserves connectivity) until compaction
    assert "v0" in idx.graph
    assert idx.tombstone_count() == 1


def test_search_excludes_tombstones():
    idx = _build()
    for i in range(10):
        idx.delete(f"v{i}")
    results = idx.search(_rand(), k=30)
    ids = {r["vector_id"] for r in results}
    assert ids.isdisjoint({f"v{i}" for i in range(10)})


def test_reinsert_revives_tombstone():
    idx = _build()
    idx.delete("v0")
    idx.insert(_rand(), "v0", {"n": 0})
    assert "v0" not in idx.deleted


def test_hard_delete_removes_node():
    idx = _build()
    idx.delete("v0", hard=True)
    assert "v0" not in idx.graph
    assert "v0" not in idx.vectors


def test_compact_reclaims_tombstones():
    idx = _build()
    for i in range(10):
        idx.delete(f"v{i}")
    summary = idx.compact()
    assert summary["reclaimed"] == 10
    assert idx.tombstone_count() == 0
    assert len(idx.graph) == 20
    for i in range(10):
        assert f"v{i}" not in idx.graph


def test_compact_repairs_entry_point():
    idx = _build()
    ep = idx.entry_point
    idx.delete(ep)
    idx.compact()
    assert idx.entry_point is not None
    assert idx.entry_point not in idx.deleted
    assert idx.entry_point in idx.graph


def test_background_compactor_threshold():
    idx = _build()
    for i in range(5):
        idx.delete(f"v{i}")
    # min_tombstones high, ratio high -> should skip
    comp = BackgroundCompactor(lambda: [("x", idx)],
                               min_tombstones=100, tombstone_ratio=0.9)
    comp.run_once()
    assert idx.tombstone_count() == 5  # not compacted

    # low threshold -> compacts
    comp2 = BackgroundCompactor(lambda: [("x", idx)], min_tombstones=1)
    comp2.run_once()
    assert idx.tombstone_count() == 0


def test_tombstones_persist_across_save_load(tmp_path):
    idx = _build()
    idx.delete("v0")
    idx.delete("v1")
    path = str(tmp_path / "idx.json")
    idx.save(path)

    loaded = HNSWIndex()
    loaded.load(path)
    assert loaded.tombstone_count() == 2
    assert "v0" in loaded.deleted
