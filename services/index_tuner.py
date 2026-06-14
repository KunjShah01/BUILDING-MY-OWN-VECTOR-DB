"""Index tuning recommendation service."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from database.schema import Collection

logger = logging.getLogger(__name__)


def _suggest_hnsw_params(vector_count: int) -> Dict[str, Any]:
    if vector_count < 1000:
        return {"M": 16, "ef_construction": 200}
    elif vector_count < 10000:
        return {"M": 32, "ef_construction": 300}
    elif vector_count < 100000:
        return {"M": 48, "ef_construction": 400}
    else:
        return {"M": 64, "ef_construction": 500}


def _suggest_ivf_params(vector_count: int) -> Dict[str, Any]:
    nlist = max(1, int(math.sqrt(vector_count)))
    return {"nlist": nlist, "nprobes": min(10, max(1, nlist // 10))}


class IndexTuner:
    def __init__(self, db: Session):
        self.db = db

    def get_recommendations(self) -> List[Dict[str, Any]]:
        collections = self.db.query(Collection).all()
        recommendations = []

        for col in collections:
            from database.schema import Vector
            count = self.db.query(Vector).filter(
                Vector.collection_id == col.collection_id,
            ).count()

            hnsw_rec = _suggest_hnsw_params(count)
            ivf_rec = _suggest_ivf_params(count)

            recommendations.append({
                "collection_id": col.collection_id,
                "vector_count": count,
                "current_params": {},
                "recommended_params": {
                    "hnsw": hnsw_rec,
                    "ivf": ivf_rec,
                },
                "reasoning": (
                    f"Collection '{col.collection_id}' has {count} vectors. "
                    f"HNSW M={hnsw_rec['M']}, ef_construction={hnsw_rec['ef_construction']}. "
                    f"IVF nlist={ivf_rec['nlist']}, nprobes={ivf_rec['nprobes']}."
                ),
            })

        return recommendations

    def apply_recommendations(self, collection_id: str) -> Dict[str, Any]:
        from database.schema import Vector
        count = self.db.query(Vector).filter(
            Vector.collection_id == collection_id,
        ).count()

        hnsw = _suggest_hnsw_params(count)
        ivf = _suggest_ivf_params(count)

        logger.info(
            "Index tuning applied for collection %s: HNSW=%s, IVF=%s",
            collection_id, hnsw, ivf,
        )

        return {
            "collection_id": collection_id,
            "applied": True,
            "hnsw_params": hnsw,
            "ivf_params": ivf,
            "note": "Parameters logged. Manual reindex required to apply changes.",
        }
