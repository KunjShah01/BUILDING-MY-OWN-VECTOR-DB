"""Tiered storage: hot (memmap), warm (NVMe/SSD), cold (S3/Blob)."""
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

HOT_DIR = "indexes/hot"
WARM_DIR = "indexes/warm"
COLD_DIR = "indexes/cold"

@dataclass
class VectorRecord:
    vector_id: str
    collection_id: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_access: float = 0.0
    tier: str = "hot"  # hot, warm, cold

class TierManager:
    def __init__(self):
        self._records: Dict[str, VectorRecord] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        os.makedirs(HOT_DIR, exist_ok=True)
        os.makedirs(WARM_DIR, exist_ok=True)
        os.makedirs(COLD_DIR, exist_ok=True)

    def track_access(self, vector_id: str):
        with self._lock:
            rec = self._records.get(vector_id)
            if rec:
                rec.access_count += 1
                rec.last_access = time.time()

    def add_vector(self, vector_id: str, collection_id: str, embedding: List[float],
                   metadata: Optional[Dict] = None):
        with self._lock:
            self._records[vector_id] = VectorRecord(
                vector_id=vector_id,
                collection_id=collection_id,
                embedding=embedding,
                metadata=metadata or {},
                access_count=0,
                last_access=time.time(),
                tier="hot",
            )

    def remove_vector(self, vector_id: str):
        with self._lock:
            self._records.pop(vector_id, None)

    def get_tier_info(self, collection_id: str) -> Dict[str, int]:
        counts = {"hot": 0, "warm": 0, "cold": 0}
        with self._lock:
            for rec in self._records.values():
                if rec.collection_id == collection_id:
                    counts[rec.tier] = counts.get(rec.tier, 0) + 1
        return counts

    def promote(self, vector_id: str, target_tier: str = "hot"):
        with self._lock:
            rec = self._records.get(vector_id)
            if rec:
                rec.tier = target_tier
                logger.info("Promoted %s to %s", vector_id, target_tier)

    def start_auto_tiering(self, interval: int = 300):
        self._running = True
        self._thread = threading.Thread(target=self._tier_loop, args=(interval,), daemon=True)
        self._thread.start()
        logger.info("Auto-tiering started (interval=%ds)", interval)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

    def _tier_loop(self, interval: int):
        while self._running:
            time.sleep(interval)
            self._evaluate_tiers()

    def _evaluate_tiers(self):
        now = time.time()
        with self._lock:
            for rec in list(self._records.values()):
                if rec.tier == "hot" and rec.last_access and (now - rec.last_access) > 86400:
                    if rec.access_count < 5:
                        rec.tier = "warm"
                        logger.info("Demoted %s to warm", rec.vector_id)
                elif rec.tier == "warm" and rec.last_access and (now - rec.last_access) > 604800:
                    rec.tier = "cold"
                    logger.info("Demoted %s to cold", rec.vector_id)

# Singleton
tier_manager = TierManager()
