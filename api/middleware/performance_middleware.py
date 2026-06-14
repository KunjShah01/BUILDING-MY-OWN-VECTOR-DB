"""Middleware for tracking request performance and recording slow queries."""
import logging
import time
from fastapi import Request, Response
from services.slow_query_analyzer import slow_query_analyzer

logger = logging.getLogger(__name__)

async def performance_middleware(request: Request, call_next):
    """Track request duration and record slow query events."""
    start = time.perf_counter()
    response: Response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if "/search" in request.url.path and elapsed_ms > slow_query_analyzer.threshold_ms:
        tenant_id = getattr(request.state, "tenant_id", "unknown")
        slow_query_analyzer.record(
            collection_id=request.path_params.get("collection_id", "unknown"),
            method=request.method,
            latency_ms=elapsed_ms,
            query_preview=str(request.query_params)[:200],
            tenant_id=tenant_id,
        )

    response.headers.append("Server-Timing", f"total;dur={elapsed_ms:.1f}")
    return response
