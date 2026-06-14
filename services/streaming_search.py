"""Streaming search service — SSE subscriptions and webhook notifications."""
import json
import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)

class StreamSubscription:
    def __init__(self, sub_id: str, collection_id: str, query_embedding: List[float],
                 threshold: float, callback_url: Optional[str] = None):
        self.sub_id = sub_id
        self.collection_id = collection_id
        self.query_embedding = query_embedding
        self.threshold = threshold
        self.callback_url = callback_url
        self.queue: List[Dict[str, Any]] = []
        self.event = threading.Event()
        self.active = True

    def notify(self, vector_data: Dict[str, Any]):
        self.queue.append(vector_data)
        self.event.set()

    def close(self):
        self.active = False
        self.event.set()

class StreamingSearchService:
    def __init__(self):
        self._subscriptions: Dict[str, StreamSubscription] = {}
        self._lock = threading.Lock()

    def register(self, collection_id: str, query_embedding: List[float],
                 threshold: float = 0.85, callback_url: Optional[str] = None) -> str:
        sub_id = str(uuid.uuid4())
        sub = StreamSubscription(sub_id, collection_id, query_embedding, threshold, callback_url)
        with self._lock:
            self._subscriptions[sub_id] = sub
        logger.info("Registered subscription %s for collection %s", sub_id, collection_id)
        return sub_id

    def unregister(self, sub_id: str):
        with self._lock:
            sub = self._subscriptions.pop(sub_id, None)
        if sub:
            sub.close()

    def get_subscription(self, sub_id: str) -> Optional[StreamSubscription]:
        with self._lock:
            return self._subscriptions.get(sub_id)

    def evaluate_and_notify(self, collection_id: str, vector_id: str,
                            embedding: List[float], metadata: Optional[Dict] = None):
        import numpy as np
        emb = np.array(embedding)
        with self._lock:
            subs = list(self._subscriptions.values())
        for sub in subs:
            if sub.collection_id != collection_id or not sub.active:
                continue
            q = np.array(sub.query_embedding)
            sim = float(np.dot(emb, q) / (np.linalg.norm(emb) * np.linalg.norm(q) + 1e-10))
            if sim >= sub.threshold:
                payload = {
                    "vector_id": vector_id,
                    "similarity": round(sim, 6),
                    "metadata": metadata or {},
                    "timestamp": time.time(),
                }
                sub.notify(payload)
                if sub.callback_url:
                    self._fire_webhook(sub.callback_url, payload)

    def _fire_webhook(self, url: str, payload: Dict):
        try:
            data = json.dumps(payload).encode()
            req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
            urlopen(req, timeout=5)
        except URLError as e:
            logger.warning("Webhook %s failed: %s", url, e)

# Singleton
streaming_service = StreamingSearchService()
