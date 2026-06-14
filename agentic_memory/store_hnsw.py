"""HNSW-backed memory store.

Wraps the from-scratch HNSW index from the "BUILDING MY OWN VECTOR DB" project
behind the same interface as `mem.store.MemoryStore`, so the pipeline can swap
linear-scan JSON storage for a real ANN graph index without other changes.

The HNSW index file is self-contained (numpy + stdlib only), so we load it by
path with importlib rather than vendoring a copy or polluting sys.path with the
other project's generic `utils` package. Point at it with:

    MEM_VECTORDB_PATH = <repo root of the vector-db project>
                        (default: C:\\BUILDING MY OWN VECTOR DB)

The index stores per-user isolation via its native `namespace` metadata field
and category filtering via a metadata predicate.
"""

import importlib.util
import os
from typing import List, Optional

from mem.store import MemoryRecord, SearchResult, _generate_id

_DEFAULT_VECTORDB_PATH = r"C:\BUILDING MY OWN VECTOR DB"


def _load_hnsw_index_class():
    """Import HNSWIndex from the external vector-db project by file path."""
    root = os.environ.get("MEM_VECTORDB_PATH", _DEFAULT_VECTORDB_PATH)
    index_file = os.path.join(root, "utils", "hnsw_index.py")
    if not os.path.exists(index_file):
        raise FileNotFoundError(
            f"HNSW index not found at {index_file}. "
            "Set MEM_VECTORDB_PATH to the vector-db project root."
        )
    spec = importlib.util.spec_from_file_location("vendor_hnsw_index", index_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HNSWIndex


class HNSWMemoryStore:
    """MemoryStore-compatible store backed by the external HNSW ANN index.

    Same method surface as mem.store.MemoryStore: initialize / insert / search /
    delete / delete_user / get_categories / fetch_all.
    """

    def __init__(self, persist_path: Optional[str] = None, m: int = 16,
                 ef_construction: int = 200):
        HNSWIndex = _load_hnsw_index_class()
        self.index = HNSWIndex(
            m=m, ef_construction=ef_construction, distance_metric="cosine"
        )
        self.persist_path = persist_path
        # The HNSW index keeps vectors + metadata in memory; we mirror enough
        # metadata to rebuild SearchResult/MemoryRecord objects.
        if persist_path and os.path.exists(persist_path):
            self._load()

    def initialize(self):
        if self.persist_path and not os.path.exists(self.persist_path):
            os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
            self._save()

    def insert(self, memories: List[MemoryRecord]):
        for m in memories:
            if not m.point_id:
                object.__setattr__(m, "point_id", _generate_id())
            self.index.insert(
                vector=m.embedding,
                vector_id=m.point_id,
                metadata={
                    "_namespace": str(m.user_id),
                    "user_id": m.user_id,
                    "memory_text": m.memory_text,
                    "categories": m.categories,
                    "date": m.date,
                },
            )
        self._save()

    def search(self, vector: List[float], user_id: int,
               categories: Optional[List[str]] = None,
               limit: int = 5) -> List[SearchResult]:
        cat_filter = None
        if categories:
            wanted = set(categories)
            def cat_filter(meta):
                return meta is not None and bool(
                    wanted.intersection(meta.get("categories", []))
                )

        raw = self.index.search(
            query_vector=vector,
            k=limit,
            metadata_filter=cat_filter,
            namespace=str(user_id),
        )
        results = []
        for hit in raw:
            meta = hit.get("metadata") or {}
            # cosine distance -> similarity
            score = 1.0 - float(hit.get("distance", 1.0))
            results.append(SearchResult(
                point_id=hit["vector_id"],
                user_id=meta.get("user_id", user_id),
                memory_text=meta.get("memory_text", ""),
                categories=meta.get("categories", []),
                date=meta.get("date", ""),
                score=score,
            ))
        return results

    def delete(self, point_ids: List[str]):
        for pid in point_ids:
            self.index.delete(pid, hard=True)
        self._save()

    def delete_user(self, user_id: int):
        ns = str(user_id)
        ids = [
            vid for vid, meta in self.index.metadata.items()
            if meta is not None and meta.get("_namespace") == ns
        ]
        for vid in ids:
            self.index.delete(vid, hard=True)
        self._save()

    def get_categories(self, user_id: int) -> List[str]:
        cats: set = set()
        ns = str(user_id)
        for meta in self.index.metadata.values():
            if meta is not None and meta.get("_namespace") == ns:
                cats.update(meta.get("categories", []))
        return sorted(cats)

    def fetch_all(self, user_id: int) -> List[SearchResult]:
        ns = str(user_id)
        out = []
        for vid, meta in self.index.metadata.items():
            if meta is None or meta.get("_namespace") != ns:
                continue
            if vid in getattr(self.index, "deleted", set()):
                continue
            out.append(SearchResult(
                point_id=vid, user_id=meta.get("user_id", user_id),
                memory_text=meta.get("memory_text", ""),
                categories=meta.get("categories", []),
                date=meta.get("date", ""), score=1.0,
            ))
        return out

    def _save(self):
        if self.persist_path:
            self.index.save(self.persist_path, format="json")

    def _load(self):
        self.index = self.index.load(self.persist_path)
