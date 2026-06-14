"""Tiered storage API endpoints."""
import logging
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

from services.tiered_storage import tier_manager

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Tiered Storage"])

@router.get("/collections/{collection_id}/tiers")
def get_tier_info(collection_id: str):
    info = tier_manager.get_tier_info(collection_id)
    return {"success": True, "collection_id": collection_id, "tiers": info}

@router.post("/collections/{collection_id}/tiers/promote")
def promote_vector(vector_id: str, target_tier: str = "hot"):
    tier_manager.promote(vector_id, target_tier)
    return {"success": True, "message": f"Promoted {vector_id} to {target_tier}"}
