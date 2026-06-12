"""
Memory-mapped vector storage for datasets larger than RAM.

Vectors are stored in a single contiguous ``.dat`` file as a fixed-shape
float32 matrix and accessed through ``numpy.memmap`` so the OS page cache
serves hot rows while cold rows stay on SSD. An append-only sidecar JSON maps
vector IDs to row offsets, plus a free list of rows reclaimed from deletes.

This is the storage substrate the roadmap calls for (DiskANN / mmap graphs):
a billion-row matrix can be searched with only the working set resident.

File layout (``<path>/``):
    vectors.dat   — raw float32 matrix, (capacity, dim)
    meta.json     — {dim, capacity, count, id_to_row, free_rows}
"""

import gc
import json
import os
import threading
from typing import Dict, List, Optional, Iterable, Tuple

import numpy as np

_META = "meta.json"
_DATA = "vectors.dat"


class MmapVectorStore:
    """Disk-backed, memory-mapped store of fixed-dimension float32 vectors."""

    def __init__(self, directory: str, dim: Optional[int] = None,
                 capacity: int = 100_000, growth_factor: float = 2.0):
        self.directory = directory
        self.growth_factor = growth_factor
        os.makedirs(directory, exist_ok=True)
        self._lock = threading.RLock()

        self.meta_path = os.path.join(directory, _META)
        self.data_path = os.path.join(directory, _DATA)

        if os.path.exists(self.meta_path):
            self._load_meta()
            self._open_mmap()
        else:
            if dim is None:
                raise ValueError("dim is required when creating a new store")
            self.dim = dim
            self.capacity = capacity
            self.count = 0
            self.id_to_row: Dict[str, int] = {}
            self.free_rows: List[int] = []
            self._create_data_file(capacity)
            self._open_mmap()
            self._save_meta()

    # ---- persistence ----

    def _load_meta(self):
        with open(self.meta_path, "r") as f:
            meta = json.load(f)
        self.dim = meta["dim"]
        self.capacity = meta["capacity"]
        self.count = meta["count"]
        self.id_to_row = meta["id_to_row"]
        self.free_rows = meta["free_rows"]

    def _save_meta(self):
        tmp = self.meta_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({
                "dim": self.dim,
                "capacity": self.capacity,
                "count": self.count,
                "id_to_row": self.id_to_row,
                "free_rows": self.free_rows,
            }, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.meta_path)

    def _create_data_file(self, capacity: int):
        mm = np.memmap(self.data_path, dtype=np.float32, mode="w+",
                       shape=(capacity, self.dim))
        mm.flush()
        del mm

    def _open_mmap(self):
        self._mm = np.memmap(self.data_path, dtype=np.float32, mode="r+",
                             shape=(self.capacity, self.dim))

    def _close_mmap(self):
        """Release the underlying mmap handle (required before file replace on Windows)."""
        mm = getattr(self, "_mm", None)
        if mm is not None:
            mm.flush()
            base = getattr(mm, "_mmap", None)
            if base is not None:
                base.close()
            self._mm = None
            del mm
            gc.collect()

    # ---- capacity ----

    def _grow(self, min_capacity: int):
        new_cap = max(int(self.capacity * self.growth_factor), min_capacity)
        old_cap = self.capacity
        # Re-create larger file, copy existing rows
        new_path = self.data_path + ".new"
        new_mm = np.memmap(new_path, dtype=np.float32, mode="w+",
                           shape=(new_cap, self.dim))
        new_mm[:old_cap] = self._mm[:]
        new_mm.flush()
        base = getattr(new_mm, "_mmap", None)
        if base is not None:
            base.close()
        del new_mm
        self._close_mmap()  # release old handle before replace (Windows)
        os.replace(new_path, self.data_path)
        self.capacity = new_cap
        self._open_mmap()

    # ---- operations ----

    def add(self, vector_id: str, vector: Iterable[float]) -> int:
        """Insert or overwrite a vector. Returns the row index used."""
        arr = np.asarray(vector, dtype=np.float32)
        if arr.shape != (self.dim,):
            raise ValueError(f"expected dim {self.dim}, got {arr.shape}")
        with self._lock:
            if vector_id in self.id_to_row:
                row = self.id_to_row[vector_id]
            elif self.free_rows:
                row = self.free_rows.pop()
            else:
                if self.count >= self.capacity:
                    self._grow(self.count + 1)
                row = self.count
            self._mm[row] = arr
            if vector_id not in self.id_to_row:
                self.id_to_row[vector_id] = row
                self.count += 1
            self._save_meta()
            return row

    def add_batch(self, items: Iterable[Tuple[str, Iterable[float]]]) -> int:
        """Bulk insert; one metadata flush at the end. Returns rows written."""
        written = 0
        with self._lock:
            for vector_id, vector in items:
                arr = np.asarray(vector, dtype=np.float32)
                if arr.shape != (self.dim,):
                    raise ValueError(f"expected dim {self.dim}, got {arr.shape}")
                if vector_id in self.id_to_row:
                    row = self.id_to_row[vector_id]
                elif self.free_rows:
                    row = self.free_rows.pop()
                else:
                    if self.count >= self.capacity:
                        self._grow(self.count + 1)
                    row = self.count
                    self.id_to_row[vector_id] = row
                    self.count += 1
                self._mm[row] = arr
                written += 1
            self._save_meta()
        return written

    def get(self, vector_id: str) -> Optional[np.ndarray]:
        """Return a copy of the stored vector, or None if absent."""
        row = self.id_to_row.get(vector_id)
        if row is None:
            return None
        return np.array(self._mm[row], dtype=np.float32)

    def delete(self, vector_id: str) -> bool:
        """Remove a vector and reclaim its row for reuse."""
        with self._lock:
            row = self.id_to_row.pop(vector_id, None)
            if row is None:
                return False
            self._mm[row] = 0.0
            self.free_rows.append(row)
            self.count -= 1
            self._save_meta()
            return True

    def matrix_view(self) -> Tuple[np.ndarray, List[str]]:
        """
        Return a (live_rows, ids) view for batch distance computation.
        Copies only live rows; cold rows stay paged out.
        """
        ids = list(self.id_to_row.keys())
        rows = [self.id_to_row[i] for i in ids]
        if not rows:
            return np.empty((0, self.dim), dtype=np.float32), []
        return np.array(self._mm[rows], dtype=np.float32), ids

    def flush(self):
        """Flush the mmap and metadata to disk."""
        with self._lock:
            self._mm.flush()
            self._save_meta()

    def close(self):
        with self._lock:
            self._save_meta()
            self._close_mmap()

    def __len__(self) -> int:
        return self.count

    def __contains__(self, vector_id: str) -> bool:
        return vector_id in self.id_to_row
