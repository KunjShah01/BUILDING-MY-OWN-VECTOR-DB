"""Slow query analyzer — captures, stores, and analyzes slow queries."""
import json
import logging
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

@dataclass
class SlowQueryRecord:
    query_id: str
    collection_id: str
    method: str
    latency_ms: float
    k: int
    filters: Optional[Dict] = None
    query_preview: str = ""
    timestamp: str = ""
    tenant_id: str = ""

class SlowQueryAnalyzer:
    """Captures and analyzes slow queries for performance tuning."""

    def __init__(self, threshold_ms: float = 100.0, max_records: int = 1000):
        self.threshold_ms = threshold_ms
        self._records: deque = deque(maxlen=max_records)
        self._lock = threading.Lock()

    def record(self, collection_id: str, method: str, latency_ms: float,
               k: int = 10, filters: Optional[Dict] = None,
               query_preview: str = "", tenant_id: str = ""):
        if latency_ms < self.threshold_ms:
            return
        record = SlowQueryRecord(
            query_id=f"slow_{int(time.time() * 1e6)}_{len(self._records)}",
            collection_id=collection_id,
            method=method,
            latency_ms=round(latency_ms, 2),
            k=k,
            filters=filters,
            query_preview=query_preview[:200],
            timestamp=datetime.now(timezone.utc).isoformat(),
            tenant_id=tenant_id,
        )
        with self._lock:
            self._records.append(record)
        logger.warning("Slow query [%.1fms] %s/%s: %s", latency_ms, collection_id, method, query_preview[:60])

    def get_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            records = list(self._records)[-limit:]
        return [
            {"query_id": r.query_id, "collection_id": r.collection_id,
             "method": r.method, "latency_ms": r.latency_ms,
             "k": r.k, "filters": r.filters,
             "query_preview": r.query_preview,
             "timestamp": r.timestamp, "tenant_id": r.tenant_id}
            for r in records
        ]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._records)
            if total == 0:
                return {"total": 0, "avg_latency_ms": 0, "max_latency_ms": 0, "p95_latency_ms": 0}
            latencies = sorted(r.latency_ms for r in self._records)
            avg = sum(latencies) / total
            p95_idx = int(total * 0.95)
            return {
                "total": total,
                "threshold_ms": self.threshold_ms,
                "avg_latency_ms": round(avg, 2),
                "max_latency_ms": round(latencies[-1], 2),
                "p95_latency_ms": round(latencies[p95_idx], 2),
                "by_method": self._group_by_method(),
                "by_collection": self._group_by_collection(),
            }

    def _group_by_method(self) -> Dict[str, int]:
        counts = {}
        with self._lock:
            for r in self._records:
                counts[r.method] = counts.get(r.method, 0) + 1
        return counts

    def _group_by_collection(self) -> Dict[str, int]:
        counts = {}
        with self._lock:
            for r in self._records:
                counts[r.collection_id] = counts.get(r.collection_id, 0) + 1
        return counts

slow_query_analyzer = SlowQueryAnalyzer()
