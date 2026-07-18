import json
from pathlib import Path
from typing import Any, Dict


class IndexSerializer:
    """Safe index serializer using JSON only (no pickle/RCE risk)."""

    @staticmethod
    def save(data: Dict[str, Any], filepath: str) -> None:
        """Save index data as JSON."""
        filepath = Path(filepath).with_suffix('.json')
        with open(filepath, 'w') as f:
            json.dump(data, f, default=str)

    @staticmethod
    def load(filepath: str) -> Dict[str, Any]:
        """Load index data from JSON."""
        filepath = Path(filepath).with_suffix('.json')
        with open(filepath, 'r') as f:
            return json.load(f)

    @staticmethod
    def estimate_size(data: Dict[str, Any]) -> int:
        """Estimate serialized size without writing to disk."""
        return len(json.dumps(data, default=str).encode('utf-8'))
