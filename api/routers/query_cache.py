"""Query cache admin endpoints."""
import logging
from fastapi import APIRouter

from services.query_cache import query_cache

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Cache"])

@router.get("/admin/cache/stats")
def cache_stats():
    return {"success": True, "stats": query_cache.stats()}

@router.delete("/admin/cache")
def cache_flush():
    query_cache.flush()
    return {"success": True, "message": "Cache flushed"}
