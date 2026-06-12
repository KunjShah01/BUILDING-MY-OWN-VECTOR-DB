"""
Background compaction for HNSW indexes.

Soft-deletes accumulate as tombstones in the graph (see ``HNSWIndex.delete``).
Tombstones preserve graph connectivity for traversal but waste memory and
slightly degrade search quality once they pile up. The BackgroundCompactor
runs a daemon thread that periodically hard-removes tombstones and repairs the
graph, triggered either on an interval or when a tombstone-ratio threshold is
crossed.
"""

import threading
import time
import logging
from typing import Callable, Dict, Iterable, Any, Optional

logger = logging.getLogger(__name__)


class BackgroundCompactor:
    """
    Periodically compacts HNSW indexes to reclaim tombstoned nodes.

    Args:
        index_provider: Callable returning the current iterable of
            (name, index) pairs to consider. Re-invoked each cycle so newly
            created collections are picked up automatically.
        interval_seconds: How often to wake and check (default 60s).
        min_tombstones: Skip an index unless it has at least this many
            tombstones (avoids churn on tiny deletes).
        tombstone_ratio: Compact when tombstones / live nodes exceeds this
            ratio, regardless of min_tombstones being met.
        lock: Optional shared lock acquired around each index compaction to
            avoid racing concurrent writers.
    """

    def __init__(
        self,
        index_provider: Callable[[], Iterable[Any]],
        interval_seconds: float = 60.0,
        min_tombstones: int = 100,
        tombstone_ratio: float = 0.2,
        lock: Optional[threading.Lock] = None,
    ):
        self.index_provider = index_provider
        self.interval_seconds = interval_seconds
        self.min_tombstones = min_tombstones
        self.tombstone_ratio = tombstone_ratio
        self.lock = lock
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.last_run: Dict[str, Any] = {}

    def _should_compact(self, index) -> bool:
        tombstones = index.tombstone_count()
        if tombstones == 0:
            return False
        live = max(len(index.graph) - tombstones, 1)
        if tombstones >= self.min_tombstones:
            return True
        return (tombstones / live) >= self.tombstone_ratio

    def run_once(self) -> Dict[str, Any]:
        """Compact every eligible index once. Returns per-index summaries."""
        results: Dict[str, Any] = {}
        for name, index in self.index_provider():
            if index is None or not hasattr(index, "compact"):
                continue
            try:
                if not self._should_compact(index):
                    continue
                if self.lock is not None:
                    with self.lock:
                        summary = index.compact()
                else:
                    summary = index.compact()
                results[name] = summary
                logger.info("Compacted index %s: %s", name, summary)
            except Exception as exc:  # noqa: BLE001 - thread must not die
                logger.exception("Compaction failed for index %s: %s", name, exc)
                results[name] = {"error": str(exc)}
        if results:
            self.last_run = {"ts": time.time(), "results": results}
        return results

    def _loop(self):
        while not self._stop.is_set():
            self.run_once()
            self._stop.wait(self.interval_seconds)

    def start(self):
        """Start the daemon compaction thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="hnsw-compactor", daemon=True
        )
        self._thread.start()
        logger.info("BackgroundCompactor started (interval=%ss)", self.interval_seconds)

    def stop(self, timeout: float = 5.0):
        """Signal the thread to stop and wait for it to exit."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("BackgroundCompactor stopped")
