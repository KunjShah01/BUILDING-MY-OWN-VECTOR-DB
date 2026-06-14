"""Memory API router: CRUD, search, chat, consolidation, streaming."""

from __future__ import annotations

import json as _json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from config.database import get_db
from models.pydantic_models import (
    MemoryBatchDeleteRequest,
    MemoryBatchDeleteResponse,
    MemoryChatRequest,
    MemoryChatResponse,
    MemoryConsolidateRequest,
    MemoryConsolidateResponse,
    MemoryCreate,
    MemoryListResponse,
    MemoryResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryStatsResponse,
    MemoryUpdate,
)
from services.memory_service import MemoryService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Memories"])


def _tenant_id(request: Request) -> str:
    return getattr(request.state, "tenant_id", None) or "default"


def get_memory_service(db: Session = Depends(get_db)) -> MemoryService:
    return MemoryService(db)


@router.post("/memories", response_model=MemoryResponse)
def create_memory(
    request: Request,
    body: MemoryCreate,
    service: MemoryService = Depends(get_memory_service),
):
    result = service.add_memory(
        user_id=_tenant_id(request),
        text=body.text,
        categories=body.categories,
        metadata=body.metadata,
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("message", "Failed to add memory"))
    return result


@router.get("/memories/stats", response_model=MemoryStatsResponse)
def memory_stats(
    request: Request,
    service: MemoryService = Depends(get_memory_service),
):
    return service.get_stats(user_id=_tenant_id(request))


@router.get("/memories", response_model=MemoryListResponse)
def list_memories(
    request: Request,
    category: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    service: MemoryService = Depends(get_memory_service),
):
    return service.get_memories(
        user_id=_tenant_id(request), category=category, limit=limit, offset=offset,
    )


@router.get("/memories/{memory_id}", response_model=MemoryResponse)
def get_memory(
    request: Request,
    memory_id: str,
    service: MemoryService = Depends(get_memory_service),
):
    mem = service.get_memory(memory_id)
    if not mem or mem.user_id != _tenant_id(request):
        raise HTTPException(status_code=404, detail="Memory not found")
    return {
        "success": True,
        "memory": {
            "memory_id": mem.memory_id,
            "user_id": mem.user_id,
            "text": mem.text,
            "categories": list(mem.categories) if mem.categories else [],
            "metadata": mem.meta_data,
            "created_at": str(mem.created_at) if mem.created_at else None,
            "updated_at": str(mem.updated_at) if mem.updated_at else None,
        },
    }


@router.patch("/memories/{memory_id}", response_model=MemoryResponse)
def update_memory(
    request: Request,
    memory_id: str,
    body: MemoryUpdate,
    service: MemoryService = Depends(get_memory_service),
):
    result = service.update_memory(
        memory_id=memory_id,
        user_id=_tenant_id(request),
        text=body.text,
        categories=body.categories,
        metadata=body.metadata,
    )
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result.get("message", "Memory not found"))
    return result


@router.delete("/memories/{memory_id}", response_model=MemoryResponse)
def delete_memory(
    request: Request,
    memory_id: str,
    service: MemoryService = Depends(get_memory_service),
):
    result = service.delete_memory(memory_id=memory_id, user_id=_tenant_id(request))
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result.get("message", "Memory not found"))
    return result


@router.post("/memories/batch-delete", response_model=MemoryBatchDeleteResponse)
def batch_delete_memories(
    request: Request,
    body: MemoryBatchDeleteRequest,
    service: MemoryService = Depends(get_memory_service),
):
    return service.batch_delete(user_id=_tenant_id(request), memory_ids=body.memory_ids)


@router.post("/memories/search", response_model=MemorySearchResponse)
def search_memories(
    request: Request,
    body: MemorySearchRequest,
    service: MemoryService = Depends(get_memory_service),
):
    return service.search_memories(
        query=body.query,
        user_id=_tenant_id(request),
        categories=body.categories,
        limit=body.limit,
    )


@router.post("/memories/chat", response_model=MemoryChatResponse)
def chat_with_memories(
    request: Request,
    body: MemoryChatRequest,
    service: MemoryService = Depends(get_memory_service),
):
    result = service.chat(
        message=body.message,
        user_id=_tenant_id(request),
        history=body.history,
        llm_model=body.llm_model or "gpt-4o-mini",
        max_tokens=body.max_tokens or 500,
        temperature=body.temperature or 0.3,
    )
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("message", "Chat failed"))
    return result


@router.post("/memories/chat/stream")
async def chat_with_memories_stream(
    request: Request,
    body: MemoryChatRequest,
    service: MemoryService = Depends(get_memory_service),
):
    return StreamingResponse(
        service.chat_stream(
            message=body.message,
            user_id=_tenant_id(request),
            history=body.history,
            llm_model=body.llm_model or "gpt-4o-mini",
            max_tokens=body.max_tokens or 500,
            temperature=body.temperature or 0.3,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/memories/consolidate", response_model=MemoryConsolidateResponse)
def consolidate_memories(
    request: Request,
    body: MemoryConsolidateRequest,
    service: MemoryService = Depends(get_memory_service),
):
    return service.consolidate(user_id=_tenant_id(request))
