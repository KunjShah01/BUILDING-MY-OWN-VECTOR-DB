"""
Index health checking utilities.

Analyzes the structural integrity and optimality of HNSW indexes.
Useful for determining when an index needs to be rebuilt or if parameters
(M, ef_construction) need adjustment.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class IndexHealthChecker:
    @staticmethod
    def check_health(index) -> Dict[str, Any]:
        """
        Analyze HNSW index health: connectivity, neighbor distribution, etc.
        Expects an instance of HNSWIndex.
        """
        if not index.graph:
            return {"status": "empty", "vector_count": 0}
            
        vector_count = len(index.graph)
        entry_point = index.entry_point
        max_level = index.max_level
        
        # 1. Check graph connectivity (can we reach all nodes from the entry point?)
        visited = set()
        queue = [entry_point] if entry_point else []
        
        # BFS through the graph (following all layers)
        while queue:
            node_id = queue.pop(0)
            if node_id not in visited:
                visited.add(node_id)
                node = index.graph.get(node_id)
                if node:
                    # Add all neighbors from all levels to queue
                    for level, neighbors in node.neighbors.items():
                        for neighbor_id in neighbors:
                            if neighbor_id not in visited:
                                queue.append(neighbor_id)
                                
        unreachable_count = vector_count - len(visited)
        
        # 2. Analyze neighbor distribution at layer 0 (the dense base layer)
        layer_0_neighbors = []
        for node in index.graph.values():
            layer_0_neighbors.append(len(node.neighbors.get(0, [])))
            
        avg_neighbors = sum(layer_0_neighbors) / vector_count if vector_count > 0 else 0
        
        # Detect under-connected nodes (less than m0/2 neighbors)
        m0 = index.m0
        under_connected = sum(1 for n in layer_0_neighbors if n < (m0 / 2))
        
        # 3. Check level distribution
        levels = {}
        for node in index.graph.values():
            node_max_level = max(node.neighbors.keys()) if node.neighbors else 0
            levels[node_max_level] = levels.get(node_max_level, 0) + 1
            
        is_healthy = unreachable_count == 0 and (under_connected / vector_count) < 0.1
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "vector_count": vector_count,
            "max_level": max_level,
            "connectivity": {
                "reachable_from_ep": len(visited),
                "unreachable_nodes": unreachable_count,
                "is_fully_connected": unreachable_count == 0
            },
            "distribution": {
                "avg_layer_0_neighbors": round(avg_neighbors, 2),
                "under_connected_nodes": under_connected,
                "under_connected_percent": round((under_connected / vector_count) * 100, 2),
                "target_m0": m0
            },
            "level_counts": levels
        }
