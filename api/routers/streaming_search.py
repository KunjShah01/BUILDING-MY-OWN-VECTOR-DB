"""Streaming search and webhook endpoints."""
import asyncio
import json
import logging
from typing import Optional
from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.streaming_search import streaming_service
from services.embedding_service import embed_text

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Streaming Search"])

class WebhookRegisterRequest(BaseModel):
    collection_id: str
    query: str
    threshold: float = 0.85
    callback_url: str

class WebhookRegisterResponse(BaseModel):
    success: bool
    subscription_id: str = ""
    message: str = ""

@router.post("/webhooks")
def register_webhook(body: WebhookRegisterRequest):
    embedding = embed_text(body.query)
    sub_id = streaming_service.register(
        collection_id=body.collection_id,
        query_embedding=embedding,
        threshold=body.threshold,
        callback_url=body.callback_url,
    )
    return {"success": True, "subscription_id": sub_id}

@router.delete("/webhooks/{sub_id}")
def unregister_webhook(sub_id: str):
    streaming_service.unregister(sub_id)
    return {"success": True, "message": "Webhook unregistered"}

@router.get("/collections/{collection_id}/search/stream")
async def stream_search(collection_id: str, query: str = Query(...),
                        threshold: float = Query(0.85)):
    embedding = embed_text(query)
    sub_id = streaming_service.register(
        collection_id=collection_id,
        query_embedding=embedding,
        threshold=threshold,
    )
    sub = streaming_service.get_subscription(sub_id)

    async def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'connected', 'subscription_id': sub_id})}\n\n"
            while sub and sub.active:
                if sub.queue:
                    item = sub.queue.pop(0)
                    yield f"data: {json.dumps(item)}\n\n"
                else:
                    await asyncio.sleep(0.5)
        finally:
            streaming_service.unregister(sub_id)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
