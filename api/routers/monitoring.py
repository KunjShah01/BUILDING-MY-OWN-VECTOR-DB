"""Monitoring and observability API endpoints."""
import logging
from typing import Optional
from fastapi import APIRouter

from services.slow_query_analyzer import slow_query_analyzer

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Monitoring"])

@router.get("/monitoring/slow-queries")
def get_slow_queries(limit: int = 50):
    return {"success": True, "slow_queries": slow_query_analyzer.get_recent(limit=limit)}

@router.get("/monitoring/slow-queries/stats")
def get_slow_query_stats():
    return {"success": True, "stats": slow_query_analyzer.get_stats()}

@router.get("/monitoring/health/details")
def health_details():
    """Detailed health check including system stats."""
    import os
    import psutil
    info = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage_percent": psutil.disk_usage("/").percent,
        "open_files": len(psutil.Process().open_files()),
        "threads": psutil.Process().num_threads(),
    }
    return {"success": True, "health": info}
