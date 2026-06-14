"""Multi-vector API router: ColBERT-style multi-vector per document."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from config.database import get_db
from models.pydantic_models import (
    MultiVectorCreate,
    MultiVectorResponse,
    MultiVectorSearchRequest,
    MultiVectorSearchResponse,
)
from services.multi_vector_service import MultiVectorService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Multi-Vector"])


def _tenant_id(request: Request) -> str:
    return getattr(request.state, "tenant_id", None) or "default"


def get_multi_vector_service(db: Session = Depends(get_db)) -> MultiVectorService:
    return MultiVectorService(db)


@router.post("/collections/{collection_id}/multi-vector", response_model=MultiVectorResponse)
def create_multi_vector_group(
    request: Request,
    collection_id: str,
    body: MultiVectorCreate,
    service: MultiVectorService = Depends(get_multi_vector_service),
):
    try:
        group_id = service.create_group(
            collection_id=collection_id,
            group_id=body.group_id,
            text=body.text,
            vectors=body.vectors,
        )
        return MultiVectorResponse(
            success=True,
            group={"group_id": group_id, "collection_id": collection_id},
        )
    except Exception as exc:
        logger.exception("Failed to create multi-vector group")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/collections/{collection_id}/search/multi", response_model=MultiVectorSearchResponse)
def search_multi_vector(
    request: Request,
    collection_id: str,
    body: MultiVectorSearchRequest,
    service: MultiVectorService = Depends(get_multi_vector_service),
):
    start = time.perf_counter()
    try:
        results = service.search_multi_vector(
            collection_id=collection_id,
            query=body.query,
            k=body.k,
        )
        elapsed = (time.perf_counter() - start) * 1000
        return MultiVectorSearchResponse(
            success=True,
            results=results,
            total=len(results),
            search_time_ms=round(elapsed, 2),
        )
    except Exception as exc:
        logger.exception("Multi-vector search failed")
        raise HTTPException(status_code=500, detail=str(exc))
