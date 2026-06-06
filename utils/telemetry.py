"""
Telemetry and observability for vector search operations.

Provides structured logging and tracing for search requests.
Ready for OpenTelemetry integration.
"""
import time
from functools import wraps
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, List
import logging

logger = logging.getLogger(__name__)

# In-memory ring buffer for recent traces (useful for /metrics endpoint)
_MAX_TRACES = 100
_recent_traces: List['SearchTelemetry'] = []


@dataclass
class SearchTelemetry:
    """Structured telemetry data for a single search request."""
    query: str
    collection_id: str = ""
    embed_time_ms: float = 0.0
    db_time_ms: float = 0.0
    index_search_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    total_time_ms: float = 0.0
    results_count: int = 0
    success: bool = True
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "collection_id": self.collection_id,
            "latencies_ms": {
                "embed": round(self.embed_time_ms, 2),
                "db": round(self.db_time_ms, 2),
                "index_search": round(self.index_search_time_ms, 2),
                "rerank": round(self.rerank_time_ms, 2),
                "total": round(self.total_time_ms, 2)
            },
            "results_count": self.results_count,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata
        }


def trace_search(func: Callable):
    """
    Decorator to wrap a search function and record its total execution time.
    For more granular tracing (embed vs index vs db), the function itself
    should populate a SearchTelemetry object.
    
    If OpenTelemetry is added later, spans would be started/ended here.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        success = True
        error_msg = ""
        results = None
        
        try:
            results = func(*args, **kwargs)
            return results
        except Exception as e:
            success = False
            error_msg = str(e)
            raise e
        finally:
            end_time = time.perf_counter()
            total_ms = (end_time - start_time) * 1000
            
            # Attempt to extract some info from kwargs for the trace
            query = kwargs.get("query", "")
            collection_id = kwargs.get("collection_id", "")
            if not query and len(args) > 1 and isinstance(args[1], str):
                query = args[1]
                
            res_count = 0
            if isinstance(results, list):
                res_count = len(results)
            elif isinstance(results, dict) and "results" in results:
                res_count = len(results["results"])
                
            telemetry = SearchTelemetry(
                query=query,
                collection_id=collection_id,
                total_time_ms=total_ms,
                results_count=res_count,
                success=success,
                error_message=error_msg,
                metadata={"function": func.__name__}
            )
            
            _recent_traces.append(telemetry)
            if len(_recent_traces) > _MAX_TRACES:
                _recent_traces.pop(0)
                
            status = "SUCCESS" if success else "FAILED"
            logger.info(f"Search trace [{func.__name__}] {status} in {total_ms:.2f}ms")

    return wrapper


def get_recent_traces() -> List[Dict[str, Any]]:
    """Retrieve recent traces as dictionaries."""
    return [t.to_dict() for t in _recent_traces]


def get_average_latencies() -> Dict[str, float]:
    """Calculate average latencies from recent traces."""
    if not _recent_traces:
        return {}
        
    successful_traces = [t for t in _recent_traces if t.success]
    if not successful_traces:
        return {}
        
    count = len(successful_traces)
    return {
        "avg_total_ms": round(sum(t.total_time_ms for t in successful_traces) / count, 2),
        "avg_embed_ms": round(sum(t.embed_time_ms for t in successful_traces) / count, 2),
        "avg_index_ms": round(sum(t.index_search_time_ms for t in successful_traces) / count, 2),
        "sample_size": count
    }
