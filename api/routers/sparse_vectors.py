"""Sparse vector API router: SPLADE-based sparse retrieval."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from config.database import get_db
from models.pydantic_models import (
    SparseSearchRequest,
    SparseSearchResponse,
    SparseVectorCreate,
    SparseVectorResponse,
)
from services.sparse_service import SparseService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Sparse Vectors"])


def _tenant_id(request: Request) -> str:
    return getattr(request.state, "tenant_id", None) or "default"


def get_sparse_service(db: Session = Depends(get_db)) -> SparseService:
    return SparseService(db)


@router.post("/collections/{collection_id}/sparse", response_model=SparseVectorResponse)
def create_sparse_vector(
    request: Request,
    collection_id: str,
    body: SparseVectorCreate,
    service: SparseService = Depends(get_sparse_service),
):
    try:
        doc_id = service.create_sparse_vector(
            collection_id=collection_id,
            doc_id=body.doc_id,
            text=body.text,
            tokens=body.tokens,
        )
        return SparseVectorResponse(
            success=True,
            sparse_vector={"doc_id": doc_id, "collection_id": collection_id},
        )
    except Exception as exc:
        logger.exception("Failed to create sparse vector")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/collections/{collection_id}/search/sparse", response_model=SparseSearchResponse)
def search_sparse(
    request: Request,
    collection_id: str,
    body: SparseSearchRequest,
    service: SparseService = Depends(get_sparse_service),
):
    start = time.perf_counter()
    try:
        results = service.search_sparse(
            collection_id=collection_id,
            query=body.query,
            k=body.k,
        )
        elapsed = (time.perf_counter() - start) * 1000
        return SparseSearchResponse(
            success=True,
            results=results,
            total=len(results),
            search_time_ms=round(elapsed, 2),
        )
    except Exception as exc:
        logger.exception("Sparse search failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/collections/{collection_id}/sparse/{doc_id}", response_model=SparseVectorResponse)
def delete_sparse_vector(
    request: Request,
    collection_id: str,
    doc_id: str,
    service: SparseService = Depends(get_sparse_service),
):
    ok = service.delete_sparse_vector(
        collection_id=collection_id,
        doc_id=doc_id,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Sparse vector not found")
    return SparseVectorResponse(success=True, message="Deleted")
