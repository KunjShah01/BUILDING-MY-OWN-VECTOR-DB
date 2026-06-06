import json
import os
import time
from typing import Dict, Any, List, Optional
from filelock import FileLock

class WriteAheadLog:
    """
    Append-only Write-Ahead Log (WAL) for durability.
    Records all index mutations (inserts, deletes) synchronously
    before acknowledging the request, ensuring crash recovery.
    """
    
    def __init__(self, collection_id: str, log_dir: str = "wal_logs"):
        self.collection_id = collection_id
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.wal_path = os.path.join(self.log_dir, f"{collection_id}.wal")
        
    def _write_entry(self, operation: str, data: Dict[str, Any]):
        entry = {
            "ts": time.time(),
            "op": operation,
            "data": data
        }
        entry_str = json.dumps(entry) + "\n"
        
        # Open file in append mode and acquire an exclusive lock
        # to ensure concurrent inserts don't corrupt the WAL
        lock = FileLock(self.wal_path + ".lock")
        with lock:
            with open(self.wal_path, "a") as f:
                f.write(entry_str)
                f.flush()
                os.fsync(f.fileno()) # Force write to physical disk
                
    def log_insert(self, vector_id: str, vector: List[float], metadata: Optional[Dict] = None):
        """Log a vector insertion."""
        self._write_entry("INSERT", {
            "id": vector_id,
            "vec": vector,
            "meta": metadata
        })
        
    def log_delete(self, vector_id: str):
        """Log a vector deletion."""
        self._write_entry("DELETE", {
            "id": vector_id
        })
        
    def log_update_metadata(self, vector_id: str, metadata: Dict):
        """Log a metadata update."""
        self._write_entry("UPDATE_META", {
            "id": vector_id,
            "meta": metadata
        })
        
    def truncate(self):
        """
        Clears the WAL. Typically called after a successful background 
        snapshot/compaction of the main index to disk.
        """
        lock = FileLock(self.wal_path + ".lock")
        with lock:
            with open(self.wal_path, "w") as f:
                f.truncate(0)
            
    def read_all(self) -> List[Dict[str, Any]]:
        """Reads all operations for crash recovery playback."""
        if not os.path.exists(self.wal_path):
            return []
            
        entries = []
        with open(self.wal_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Log corruption detected at the tail end
                        continue
        return entries
