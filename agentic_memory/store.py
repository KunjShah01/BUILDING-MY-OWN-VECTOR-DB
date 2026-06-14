import json
import os
import uuid
from typing import Optional, List
from dataclasses import dataclass, asdict

try:
    import numpy as np
except ImportError:  # pure-stdlib fallback path still works
    np = None


@dataclass
class MemoryRecord:
    point_id: str
    user_id: int
    memory_text: str
    categories: List[str]
    date: str
    embedding: List[float]


@dataclass
class SearchResult:
    point_id: str
    user_id: int
    memory_text: str
    categories: List[str]
    date: str
    score: float


def _cosine_sim(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _generate_id() -> str:
    return uuid.uuid4().hex


class MemoryStore:
    """In-memory vector store with JSON persistence. Zero external deps."""

    def __init__(self, persist_path: Optional[str] = None):
        self.records: List[MemoryRecord] = []
        self.persist_path = persist_path
        if persist_path and os.path.exists(persist_path):
            self._load()

    def initialize(self):
        if self.persist_path and not os.path.exists(self.persist_path):
            os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
            self._save()

    def insert(self, memories: List[MemoryRecord]):
        for m in memories:
            if not m.point_id:
                object.__setattr__(m, 'point_id', _generate_id())
            self.records.append(m)
        self._save()

    def search(
        self,
        vector: List[float],
        user_id: int,
        categories: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[SearchResult]:
        dim = len(vector)
        candidates = [
            r for r in self.records
            if r.user_id == user_id and len(r.embedding) == dim
        ]
        if categories:
            candidates = [
                r for r in candidates
                if any(c in r.categories for c in categories)
            ]

        if not candidates:
            return []

        if np is not None:
            mat = np.array([r.embedding for r in candidates], dtype=np.float32)
            q = np.array(vector, dtype=np.float32)
            sims = mat @ q  # both sides unit-normalized -> cosine
            scored = list(zip(sims.tolist(), candidates))
        else:
            scored = [(_cosine_sim(vector, r.embedding), r) for r in candidates]

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            SearchResult(
                point_id=r.point_id,
                user_id=r.user_id,
                memory_text=r.memory_text,
                categories=r.categories,
                date=r.date,
                score=s,
            )
            for s, r in scored[:limit]
        ]

    def delete(self, point_ids: List[str]):
        id_set = set(point_ids)
        self.records = [r for r in self.records if r.point_id not in id_set]
        self._save()

    def delete_user(self, user_id: int):
        self.records = [r for r in self.records if r.user_id != user_id]
        self._save()

    def get_categories(self, user_id: int) -> List[str]:
        cats: set = set()
        for r in self.records:
            if r.user_id == user_id:
                cats.update(r.categories)
        return sorted(cats)

    def fetch_all(self, user_id: int) -> List[SearchResult]:
        return [
            SearchResult(
                point_id=r.point_id, user_id=r.user_id,
                memory_text=r.memory_text, categories=r.categories,
                date=r.date, score=1.0,
            )
            for r in self.records if r.user_id == user_id
        ]

    def _save(self):
        if not self.persist_path:
            return
        data = []
        for r in self.records:
            d = asdict(r)
            data.append(d)
        os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
        with open(self.persist_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        with open(self.persist_path) as f:
            data = json.load(f)
        self.records = [MemoryRecord(**d) for d in data]
