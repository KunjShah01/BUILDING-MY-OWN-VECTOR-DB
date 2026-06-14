"""Memory API router: CRUD, search, chat, consolidation."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from config.database import get_db
from models.pydantic_models import (
    MemoryCreate,
    MemoryUpdate,
    MemoryResponse,
    MemoryListResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryChatRequest,
    MemoryChatResponse,
    MemoryConsolidateRequest,
    MemoryConsolidateResponse,
)
from services.memory_service import MemoryService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Memories"])


def get_memory_service(db: Session = Depends(get_db)) -> MemoryService:
    return MemoryService(db)


@router.post("/memories", response_model=MemoryResponse)
def create_memory(
    body: MemoryCreate,
    user_id: str = Query("default"),
    service: MemoryService = Depends(get_memory_service),
):
    result = service.add_memory(
        user_id=user_id,
        text=body.text,
        categories=body.categories,
        metadata=body.metadata,
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("message", "Failed to add memory"))
    return result


@router.get("/memories", response_model=MemoryListResponse)
def list_memories(
    user_id: str = Query("default"),
    category: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    service: MemoryService = Depends(get_memory_service),
):
    return service.get_memories(
        user_id=user_id, category=category, limit=limit, offset=offset,
    )


@router.get("/memories/{memory_id}", response_model=MemoryResponse)
def get_memory(
    memory_id: str,
    service: MemoryService = Depends(get_memory_service),
):
    mem = service.get_memory(memory_id)
    if not mem:
        raise HTTPException(status_code=404, detail="Memory not found")
    from database.schema import Memory as MemoryModel

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
    memory_id: str,
    body: MemoryUpdate,
    user_id: str = Query("default"),
    service: MemoryService = Depends(get_memory_service),
):
    result = service.update_memory(
        memory_id=memory_id,
        user_id=user_id,
        text=body.text,
        categories=body.categories,
        metadata=body.metadata,
    )
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result.get("message", "Memory not found"))
    return result


@router.delete("/memories/{memory_id}", response_model=MemoryResponse)
def delete_memory(
    memory_id: str,
    user_id: str = Query("default"),
    service: MemoryService = Depends(get_memory_service),
):
    result = service.delete_memory(memory_id=memory_id, user_id=user_id)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result.get("message", "Memory not found"))
    return result


@router.post("/memories/search", response_model=MemorySearchResponse)
def search_memories(
    body: MemorySearchRequest,
    service: MemoryService = Depends(get_memory_service),
):
    return service.search_memories(
        query=body.query,
        user_id=body.user_id,
        categories=body.categories,
        limit=body.limit,
    )


@router.post("/memories/chat", response_model=MemoryChatResponse)
def chat_with_memories(
    body: MemoryChatRequest,
    service: MemoryService = Depends(get_memory_service),
):
    result = service.chat(
        message=body.message,
        user_id=body.user_id,
        history=body.history,
        llm_model=body.llm_model or "gpt-4o-mini",
        max_tokens=body.max_tokens or 500,
        temperature=body.temperature or 0.3,
    )
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("message", "Chat failed"))
    return result


@router.post("/memories/consolidate", response_model=MemoryConsolidateResponse)
def consolidate_memories(
    body: MemoryConsolidateRequest,
    service: MemoryService = Depends(get_memory_service),
):
    return service.consolidate(user_id=body.user_id)
