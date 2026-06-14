"""ColBERT-style multi-vector service using MaxSim scoring."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy.orm import Session

from database.schema import MultiVectorGroup
from services.embedding_service import embed_text

logger = logging.getLogger(__name__)


def _maxsim(query_vecs: List[List[float]], doc_vecs: List[List[float]]) -> float:
    q_arr = np.array(query_vecs)
    d_arr = np.array(doc_vecs)
    scores = []
    for qv in q_arr:
        sims = np.dot(d_arr, qv)
        scores.append(float(np.max(sims)))
    return float(np.sum(scores))


class MultiVectorService:
    def __init__(self, db: Session):
        self.db = db

    def create_group(
        self,
        collection_id: str,
        group_id: str,
        text: str,
        vectors: Optional[List[List[float]]] = None,
    ) -> str:
        if vectors is None:
            tokens = text.split()
            vectors = []
            for token in tokens:
                if token.strip():
                    vec = embed_text(token.strip())
                    vectors.append(vec)
        existing = self.db.query(MultiVectorGroup).filter(
            MultiVectorGroup.group_id == group_id,
        ).first()
        if existing:
            existing.text = text
            existing.vectors = vectors
        else:
            g = MultiVectorGroup(
                group_id=group_id,
                collection_id=collection_id,
                text=text,
                vectors=vectors,
            )
            self.db.add(g)
        self.db.commit()
        return group_id

    def search_multi_vector(
        self,
        collection_id: str,
        query: str,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        query_tokens = query.split()
        query_vecs = [embed_text(t.strip()) for t in query_tokens if t.strip()]
        if not query_vecs:
            return []

        groups = self.db.query(MultiVectorGroup).filter(
            MultiVectorGroup.collection_id == collection_id,
        ).all()
        if not groups:
            return []

        scored = []
        for g in groups:
            doc_vecs = g.vectors
            if not doc_vecs:
                continue
            score = _maxsim(query_vecs, doc_vecs)
            scored.append((score, g))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]
        return [
            {
                "group_id": g.group_id,
                "score": float(score),
                "text": g.text,
                "vector_count": len(g.vectors) if g.vectors else 0,
                "created_at": str(g.created_at) if g.created_at else None,
            }
            for score, g in top
        ]
