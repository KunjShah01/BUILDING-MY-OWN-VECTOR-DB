"""SPLADE sparse vector service for bag-of-weights retrieval."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy.orm import Session

from database.schema import SparseVector

logger = logging.getLogger(__name__)

_splade_model = None
_splade_tokenizer = None


def _load_splade():
    global _splade_model, _splade_tokenizer
    if _splade_model is not None:
        return _splade_model, _splade_tokenizer
    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        model_name = "naver/splade-cocondenser-ensembledistil"
        _splade_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _splade_model = AutoModelForMaskedLM.from_pretrained(model_name)
        _splade_model.eval()
        logger.info("SPLADE model %s loaded", model_name)
    except Exception as exc:
        logger.error("Failed to load SPLADE model: %s", exc)
        raise
    return _splade_model, _splade_tokenizer


def _splade_encode(texts: List[str]) -> List[Dict[str, float]]:
    model, tokenizer = _load_splade()
    try:
        import torch
    except ImportError:
        raise RuntimeError("torch is required for SPLADE encoding")
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        logits = model(**tokens).logits
    logits, idx = torch.topk(logits, 50, dim=-1)
    results = []
    for batch_idx in range(logits.size(0)):
        weights = torch.relu(logits[batch_idx]).cpu().numpy()
        ids = idx[batch_idx].cpu().numpy()
        token_weights = {str(int(i)): float(w) for i, w in zip(ids, weights) if w > 0}
        results.append(token_weights)
    return results


class SparseService:
    def __init__(self, db: Session):
        self.db = db

    def create_sparse_vector(
        self,
        collection_id: str,
        doc_id: str,
        text: str,
        tokens: Optional[Dict[str, float]] = None,
    ) -> str:
        if tokens is None:
            encoded = _splade_encode([text])
            tokens = encoded[0]
        existing = self.db.query(SparseVector).filter(
            SparseVector.collection_id == collection_id,
            SparseVector.doc_id == doc_id,
        ).first()
        if existing:
            existing.sparse_embedding = tokens
            existing.text = text
        else:
            sv = SparseVector(
                collection_id=collection_id,
                doc_id=doc_id,
                sparse_embedding=tokens,
                text=text,
            )
            self.db.add(sv)
        self.db.commit()
        return doc_id

    def search_sparse(
        self,
        collection_id: str,
        query: str,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        query_vec = _splade_encode([query])[0]
        stored = self.db.query(SparseVector).filter(
            SparseVector.collection_id == collection_id,
        ).all()
        if not stored:
            return []

        scores = []
        for sv in stored:
            doc_vec = sv.sparse_embedding
            dot = 0.0
            q_norm = 0.0
            d_norm = 0.0
            for qid, qw in query_vec.items():
                q_norm += qw * qw
                dw = doc_vec.get(qid, 0.0)
                dot += qw * dw
            for _, dw in doc_vec.items():
                d_norm += dw * dw
            if q_norm > 0 and d_norm > 0:
                sim = dot / (np.sqrt(q_norm) * np.sqrt(d_norm))
            else:
                sim = 0.0
            scores.append((sim, sv))

        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[:k]
        return [
            {
                "doc_id": sv.doc_id,
                "score": float(score),
                "text": sv.text,
                "sparse_embedding": sv.sparse_embedding,
                "created_at": str(sv.created_at) if sv.created_at else None,
            }
            for score, sv in top
        ]

    def delete_sparse_vector(self, collection_id: str, doc_id: str) -> bool:
        sv = self.db.query(SparseVector).filter(
            SparseVector.collection_id == collection_id,
            SparseVector.doc_id == doc_id,
        ).first()
        if not sv:
            return False
        self.db.delete(sv)
        self.db.commit()
        return True
