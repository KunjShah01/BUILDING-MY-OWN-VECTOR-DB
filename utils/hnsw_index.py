import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set, Callable
from dataclasses import dataclass, field
import random
import heapq
from collections import defaultdict
import json
import os
import logging

logger = logging.getLogger(__name__)

@dataclass
class Node:
    """
    Node in the HNSW graph
    """
    node_id: str
    vector: np.ndarray
    neighbors: Dict[int, List[str]] = field(default_factory=dict)
    # neighbors[layer] = list of neighbor node IDs
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.node_id == other.node_id
        return False

class HNSWIndex:
    """
    Hierarchical Navigable Small World Index Implementation
    
    This implementation provides:
    - Hierarchical graph structure
    - Greedy best-first search
    - Dynamic index construction
    - Adjustable parameters for recall/speed tradeoff
    """
    
    def __init__(self, m: int = 16, m0: int = None, ef_construction: int = 200,
                 level_mult: float = 1.0 / np.log(2.0),
                 distance_metric: str = "cosine"):
        """
        Initialize HNSW index
        
        Args:
            m: Number of neighbors per node in each layer
            m0: Number of neighbors in layer 0 (default: 2*m)
            ef_construction: Search breadth during construction (higher = better quality, slower)
            level_mult: Multiplier for level generation (controls height distribution)
            distance_metric: Distance metric to use (cosine or euclidean)
        """
        self.m = m
        self.m0 = m0 if m0 is not None else 2 * m
        self.ef_construction = ef_construction
        self.level_mult = level_mult
        self.distance_metric = distance_metric
        
        # Graph structure
        self.graph: Dict[str, Node] = {}
        self.entry_point: Optional[str] = None
        self.max_level: int = 0
        
        # Vector storage
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

        # Tombstones: soft-deleted ids kept in the graph for traversal
        # until a background compaction hard-removes them
        self.deleted: set = set()

        # Statistics
        self.total_inserted = 0
        self.total_connections_made = 0
    
    def _calculate_level(self) -> int:
        """
        Randomly determine the level for a new node using exponential distribution
        
        Returns:
            Level for the new node
        """
        # Generate random number between 0 and 1
        r = random.random()
        
        # Calculate level based on exponential distribution
        level = int(-np.log(r) * self.level_mult)
        
        # Limit level to prevent very high levels
        level = min(level, 255)
        
        return level
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        if self.distance_metric != "cosine":
            return vector

        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate distance between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Distance between vectors
        """
        if self.distance_metric == "cosine":
            return 1.0 - float(np.dot(vec1, vec2))

        return float(np.linalg.norm(vec1 - vec2))
    
    def _search_layer(self, query_vector: np.ndarray, ef: int, 
                     node_id: str, level: int) -> List[Tuple[str, float]]:
        """
        Vectorized best-first search in a single layer.

        Instead of computing distances one-at-a-time per neighbor in a Python
        loop, we gather all unvisited neighbor vectors into a matrix and compute
        distances with a single NumPy matmul (for cosine) or vectorized L2.

        Args:
            query_vector: Query vector (already normalized for cosine)
            ef: Number of nearest neighbors to maintain
            node_id: Starting node ID
            level: Layer to search in

        Returns:
            List of (node_id, distance) tuples, sorted by distance
        """
        visited: Set[str] = set()

        # Min-heap for candidates (exploration frontier, closest first)
        candidates: List[Tuple[float, str]] = []
        # Max-heap for results (negative distance for max-heap behavior)
        results: List[Tuple[float, str]] = []

        if node_id not in self.graph:
            return []

        start_node = self.graph[node_id]
        start_distance = self._distance(query_vector, start_node.vector)
        heapq.heappush(candidates, (start_distance, node_id))
        visited.add(node_id)

        while candidates:
            distance, current_id = heapq.heappop(candidates)

            # Termination: closest remaining candidate is >= farthest result
            if results:
                farthest_distance = -results[0][0]
                if distance >= farthest_distance:
                    break

            # Add to results
            heapq.heappush(results, (-distance, current_id))

            # If we have more than ef results, pop farthest
            if len(results) > ef:
                heapq.heappop(results)

            # ---- Vectorized neighbor exploration ----
            current_node = self.graph[current_id]
            if level not in current_node.neighbors:
                continue

            # Collect unvisited neighbors
            unvisited_ids = [
                nid for nid in current_node.neighbors[level]
                if nid not in visited
            ]
            if not unvisited_ids:
                continue

            # Mark visited immediately to avoid duplicates from parallel paths
            for nid in unvisited_ids:
                visited.add(nid)

            # Batch distance computation: stack vectors → single matmul
            neighbor_vecs = np.stack(
                [self.graph[nid].vector for nid in unvisited_ids]
            )

            if self.distance_metric == "cosine":
                # Vectors are pre-normalized at insert time, so
                # cosine_distance = 1 - dot(v, q)  — no norm needed
                dists = 1.0 - neighbor_vecs @ query_vector
            else:
                # Euclidean: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
                vec_sq = np.sum(neighbor_vecs ** 2, axis=1)
                q_sq = np.dot(query_vector, query_vector)
                cross = neighbor_vecs @ query_vector
                dists = np.sqrt(np.maximum(vec_sq + q_sq - 2.0 * cross, 0.0))

            for nid, d in zip(unvisited_ids, dists):
                heapq.heappush(candidates, (float(d), nid))

        # Convert results to sorted list
        sorted_results = [(nid, -neg_dist) for neg_dist, nid in results]
        sorted_results.sort(key=lambda x: x[1])

        return sorted_results

    def _search_layer_filtered(
        self, query_vector: np.ndarray, ef: int,
        node_id: str, level: int,
        metadata_filter: Callable[[Optional[Dict[str, Any]]], bool] = None,
        max_ef_expansion: int = 4,
    ) -> List[Tuple[str, float]]:
        """
        Filtered search: only candidates passing the metadata predicate
        enter the result set. The search beam (ef) is dynamically expanded
        when too many candidates are filtered out, up to ef * max_ef_expansion.

        Args:
            query_vector: Query vector
            ef: Base search breadth
            node_id: Starting node ID
            level: Layer to search in
            metadata_filter: Predicate that receives a node's metadata dict
                             and returns True to include, False to skip.
            max_ef_expansion: Maximum multiplier for ef when filter is aggressive

        Returns:
            List of (node_id, distance) tuples that pass the filter
        """
        if metadata_filter is None:
            return self._search_layer(query_vector, ef, node_id, level)

        visited: Set[str] = set()
        candidates: List[Tuple[float, str]] = []
        results: List[Tuple[float, str]] = []  # max-heap (neg dist)
        expanded_ef = ef
        max_expanded = ef * max_ef_expansion

        if node_id not in self.graph:
            return []

        start_node = self.graph[node_id]
        start_distance = self._distance(query_vector, start_node.vector)
        heapq.heappush(candidates, (start_distance, node_id))
        visited.add(node_id)

        explored = 0
        filtered_out = 0

        while candidates:
            distance, current_id = heapq.heappop(candidates)

            if results:
                farthest_distance = -results[0][0]
                if distance >= farthest_distance and len(results) >= expanded_ef:
                    break

            # Check filter — only add to results if it passes
            node_meta = self.metadata.get(current_id)
            if metadata_filter(node_meta):
                heapq.heappush(results, (-distance, current_id))
                if len(results) > expanded_ef:
                    heapq.heappop(results)
            else:
                filtered_out += 1
                # Dynamically expand ef if filter is aggressive
                if filtered_out > explored * 0.5 and expanded_ef < max_expanded:
                    expanded_ef = min(expanded_ef * 2, max_expanded)

            explored += 1

            # Vectorized neighbor exploration (same as _search_layer)
            current_node = self.graph[current_id]
            if level not in current_node.neighbors:
                continue

            unvisited_ids = [
                nid for nid in current_node.neighbors[level]
                if nid not in visited
            ]
            if not unvisited_ids:
                continue

            for nid in unvisited_ids:
                visited.add(nid)

            neighbor_vecs = np.stack(
                [self.graph[nid].vector for nid in unvisited_ids]
            )
            if self.distance_metric == "cosine":
                dists = 1.0 - neighbor_vecs @ query_vector
            else:
                vec_sq = np.sum(neighbor_vecs ** 2, axis=1)
                q_sq = np.dot(query_vector, query_vector)
                cross = neighbor_vecs @ query_vector
                dists = np.sqrt(np.maximum(vec_sq + q_sq - 2.0 * cross, 0.0))

            for nid, d in zip(unvisited_ids, dists):
                heapq.heappush(candidates, (float(d), nid))

        sorted_results = [(nid, -neg_dist) for neg_dist, nid in results]
        sorted_results.sort(key=lambda x: x[1])
        return sorted_results
    
    def _select_neighbors(self, query_vector: np.ndarray, candidates: List[str], 
                         m: int, level: int) -> List[str]:
        """
        Select neighbors using the pruning heuristic from the HNSW paper
        (Algorithm 4 — "SELECT-NEIGHBORS-HEURISTIC").

        Instead of simply picking the m closest candidates, this heuristic
        prefers candidates that are closer to the query than to any already-
        selected neighbor.  This keeps the graph diverse (neighbors point in
        different directions) and dramatically improves navigability, which
        translates to higher recall at the same M and ef_construction.

        Args:
            query_vector: Query vector
            candidates: List of candidate node IDs
            m: Maximum number of neighbors to select
            level: Layer level (unused but kept for API compat)

        Returns:
            List of selected neighbor IDs (len <= m)
        """
        if len(candidates) <= m:
            return candidates

        # Vectorized distance computation for all candidates at once
        cand_vecs = np.stack([self.graph[cid].vector for cid in candidates])
        if self.distance_metric == "cosine":
            dists = 1.0 - cand_vecs @ query_vector
        else:
            diff = cand_vecs - query_vector
            dists = np.sqrt(np.sum(diff ** 2, axis=1))

        # Sort candidates by distance to query (closest first)
        sorted_indices = np.argsort(dists)
        sorted_candidates = [(candidates[i], float(dists[i])) for i in sorted_indices]

        selected: List[str] = []

        for cid, dist_to_query in sorted_candidates:
            if len(selected) >= m:
                break

            # Pruning heuristic: accept candidate only if it is closer to
            # the query than to every already-selected neighbor.
            # This ensures selected neighbors are spread across different
            # regions of the space rather than clustered together.
            is_good = True
            if selected:
                cand_vec = self.graph[cid].vector
                for sel_id in selected:
                    sel_vec = self.graph[sel_id].vector
                    dist_to_selected = self._distance(cand_vec, sel_vec)
                    if dist_to_selected < dist_to_query:
                        is_good = False
                        break

            if is_good:
                selected.append(cid)

        # If the heuristic was too aggressive and we have fewer than m,
        # fill remaining slots with closest unused candidates
        if len(selected) < m:
            selected_set = set(selected)
            for cid, _ in sorted_candidates:
                if cid not in selected_set:
                    selected.append(cid)
                    if len(selected) >= m:
                        break

        return selected
    
    def _connect_node(self, node_id: str, level: int, 
                     ep_node_id: str = None) -> str:
        """
        Connect a node at a specific level
        
        Args:
            node_id: Node to connect
            level: Level to connect at
            ep_node_id: Entry point node (default: current entry point)
            
        Returns:
            The entry point used
        """
        if ep_node_id is None:
            ep_node_id = self.entry_point
        
        if ep_node_id is None:
            # This is the first node
            self.entry_point = node_id
            return node_id
        
        # Search for nearest neighbors in this layer
        search_results = self._search_layer(
            self.graph[node_id].vector,
            self.ef_construction,
            ep_node_id,
            level
        )
        
        # Get candidate neighbors (excluding the node itself)
        candidates = [nid for nid, _ in search_results 
                     if nid != node_id and level in self.graph[nid].neighbors]
        
        # Select neighbors
        neighbors = self._select_neighbors(
            self.graph[node_id].vector,
            candidates,
            self.m,
            level
        )
        
        # Add bidirectional connections
        node = self.graph[node_id]
        if level not in node.neighbors:
            node.neighbors[level] = []
        
        for neighbor_id in neighbors:
            neighbor = self.graph[neighbor_id]
            if level not in neighbor.neighbors:
                neighbor.neighbors[level] = []
            
            # Add to current node
            if neighbor_id not in node.neighbors[level]:
                node.neighbors[level].append(neighbor_id)
                self.total_connections_made += 1

            # Add bidirectional connection
            if node_id not in neighbor.neighbors[level]:
                neighbor.neighbors[level].append(node_id)
                self.total_connections_made += 1
        
        return search_results[0][0] if search_results else ep_node_id
    
    def insert(self, vector: List[float], vector_id: str, 
              metadata: Dict[str, Any] = None, level: int = None):
        """
        Insert a vector into the HNSW index
        
        Args:
            vector: Vector to insert
            vector_id: Unique vector ID
            metadata: Optional metadata
            level: Optional level (randomly determined if not provided)
        """
        # Create node
        vector_array = np.array(vector, dtype=np.float32)
        vector_array = self._normalize_vector(vector_array)
        
        # Determine level if not provided
        if level is None:
            level = self._calculate_level()
        
        node = Node(node_id=vector_id, vector=vector_array)
        self.graph[vector_id] = node
        self.vectors[vector_id] = vector_array
        self.metadata[vector_id] = metadata
        self.deleted.discard(vector_id)  # re-insert revives a tombstoned id
        self.total_inserted += 1
        
        # Update max level
        if level > self.max_level:
            self.max_level = level
        
        # Start from entry point
        ep = self.entry_point
        
        # Navigate down from max_level to level+1 (ef=1, no connections)
        for l in range(self.max_level, level, -1):
            if ep is not None:
                search_results = self._search_layer(vector_array, 1, ep, l)
                if search_results:
                    ep = search_results[0][0]

        # Connect at each level from level down to 0
        if ep is not None:
            for l in range(level, -1, -1):
                ep = self._connect_node(vector_id, l, ep)
        else:
            self.entry_point = vector_id
            # Initialize neighbor structure for the first node
            self._connect_node(vector_id, 0, self.entry_point)
            if level > 0:
                self._connect_node(vector_id, level, self.entry_point)
    
    def insert_batch(self, vectors: List[Dict[str, Any]]):
        """
        Insert multiple vectors into the index
        
        Args:
            vectors: List of vector dictionaries with 'vector', 'vector_id', 'metadata'
        """
        for vector_data in vectors:
            self.insert(
                vector=vector_data["vector"],
                vector_id=vector_data["vector_id"],
                metadata=vector_data.get("metadata")
            )
    
    def search(self, query_vector: List[float], k: int = 5, 
              ef: int = None, level: int = None,
              metadata_filter: Callable[[Optional[Dict[str, Any]]], bool] = None,
              namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors with optional metadata filtering and
        namespace isolation.

        Args:
            query_vector: Query vector
            k: Number of results to return
            ef: Search breadth (higher = better recall, slower)
            level: Level to start search from (default: max_level)
            metadata_filter: Optional predicate applied during graph traversal.
                Returns True to include a node in results.
            namespace: If set, only return vectors with matching namespace
                in their metadata["_namespace"] field.

        Returns:
            List of search results with distances and metadata
        """
        query_array = np.array(query_vector, dtype=np.float32)
        query_array = self._normalize_vector(query_array)

        if len(self.graph) == 0:
            return []

        if ef is None:
            ef = max(k, 10)  # Default ef

        if level is None:
            level = self.max_level

        # Build combined filter
        active_filter = metadata_filter
        if namespace is not None:
            ns_filter = lambda meta: (
                meta is not None and meta.get("_namespace") == namespace
            )
            if active_filter is not None:
                orig_filter = active_filter
                active_filter = lambda meta: ns_filter(meta) and orig_filter(meta)
            else:
                active_filter = ns_filter

        # Start from entry point
        ep = self.entry_point

        if ep is None:
            return []

        # Navigate down from top level (greedy, ef=1)
        for l in range(self.max_level, 0, -1):
            search_results = self._search_layer(query_array, 1, ep, l)
            if search_results:
                ep = search_results[0][0]

        # Search at level 0 with ef parameter
        if active_filter is not None:
            results = self._search_layer_filtered(
                query_array, ef, ep, 0, active_filter
            )
        else:
            results = self._search_layer(query_array, ef, ep, 0)

        # Get top k results
        top_k = results[:k]

        # Build response (over-fetch so tombstones don't shrink result set)
        response = []
        for node_id, distance in results:
            if node_id in self.graph and node_id not in self.deleted:
                response.append({
                    "vector_id": node_id,
                    "distance": distance,
                    "metadata": self.metadata.get(node_id)
                })
            if len(response) >= k:
                break

        return response

    def search_optimized(self, query_vector: List[float], k: int = 5, 
                        ef_search: int = None) -> List[Dict[str, Any]]:
        """
        Optimized search with early termination
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            ef_search: Search breadth (default: max(k, 10))
            
        Returns:
            List of search results
        """
        return self.search(query_vector, k, ef=ef_search)
    
    def delete(self, vector_id: str, hard: bool = False) -> bool:
        """
        Delete a vector from the index.

        By default this is a *soft* delete: the node is tombstoned and kept in
        the graph so traversal connectivity is preserved, but it is excluded
        from search results. Background compaction (``compact()``) later removes
        tombstones in bulk and repairs the graph. Pass ``hard=True`` to remove
        the node immediately (used internally by compaction).

        Args:
            vector_id: Vector ID to delete
            hard: If True, physically remove the node now

        Returns:
            True if deleted, False if not found
        """
        if vector_id not in self.graph:
            return False

        if not hard:
            if vector_id in self.deleted:
                return False  # already tombstoned
            self.deleted.add(vector_id)
            return True

        node = self.graph[vector_id]

        # Remove from all neighbor lists
        for level, neighbors in node.neighbors.items():
            for neighbor_id in neighbors:
                if neighbor_id in self.graph:
                    neighbor = self.graph[neighbor_id]
                    if level in neighbor.neighbors:
                        neighbor.neighbors[level] = [n for n in neighbor.neighbors[level]
                                                    if n != vector_id]

        # Remove from graph
        del self.graph[vector_id]
        del self.vectors[vector_id]

        if vector_id in self.metadata:
            del self.metadata[vector_id]
        self.deleted.discard(vector_id)

        # Update entry point if needed
        if self.entry_point == vector_id:
            self.entry_point = next(
                (nid for nid in self.graph if nid not in self.deleted), None
            )

        return True

    def compact(self) -> Dict[str, Any]:
        """
        Background compaction: hard-remove all tombstoned nodes and repair the
        graph. Returns a summary of how many tombstones were reclaimed.

        Safe to call concurrently with reads only if the caller holds the
        appropriate lock; the BackgroundCompactor handles this.
        """
        reclaimed = 0
        for vector_id in list(self.deleted):
            if self.delete(vector_id, hard=True):
                reclaimed += 1

        # Entry point may have been a tombstone with no live replacement
        if self.entry_point is None and self.graph:
            self.entry_point = next(iter(self.graph.keys()))

        return {
            "reclaimed": reclaimed,
            "remaining_nodes": len(self.graph),
            "tombstones": len(self.deleted),
        }

    def tombstone_count(self) -> int:
        """Number of soft-deleted nodes awaiting compaction."""
        return len(self.deleted)
    
    def get_neighbors(self, vector_id: str, level: int = 0) -> List[str]:
        """
        Get neighbors of a node at a specific level
        
        Args:
            vector_id: Node ID
            level: Level
            
        Returns:
            List of neighbor IDs
        """
        if vector_id in self.graph:
            node = self.graph[vector_id]
            return node.neighbors.get(level, [])
        return []
    
    def get_node_info(self, vector_id: str) -> Dict[str, Any]:
        """
        Get information about a node
        
        Args:
            vector_id: Node ID
            
        Returns:
            Node information dictionary
        """
        if vector_id not in self.graph:
            return {}
        
        node = self.graph[vector_id]
        
        return {
            "node_id": node.node_id,
            "vector_shape": node.vector.shape,
            "neighbors": dict(node.neighbors),
            "total_neighbors": sum(len(neighbors) for neighbors in node.neighbors.values()),
            "metadata": self.metadata.get(vector_id)
        }
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the graph
        
        Returns:
            Dictionary with graph statistics
        """
        if len(self.graph) == 0:
            return {
                "total_nodes": 0,
                "total_edges": 0,
                "avg_connections": 0,
                "max_level": 0
            }
        
        # Calculate statistics
        total_edges = sum(
            sum(len(neighbors) for neighbors in node.neighbors.values())
            for node in self.graph.values()
        )
        
        # Count nodes per level
        level_counts = defaultdict(int)
        for node in self.graph.values():
            max_level = max(node.neighbors.keys()) if node.neighbors else 0
            level_counts[max_level] += 1
        
        # Calculate average connections per node
        avg_connections = total_edges / len(self.graph) if self.graph else 0
        
        return {
            "total_nodes": len(self.graph),
            "total_edges": total_edges,
            "avg_connections": avg_connections,
            "max_level": self.max_level,
            "level_distribution": dict(level_counts),
            "total_inserted": self.total_inserted,
            "entry_point": self.entry_point
        }
    
    def save(self, filepath: str, format: str = "json"):
        """
        Save the index to disk
        
        Args:
            filepath: Path to save the index
            format: Serialization format ("json" or "binary")
        """
        if format == "binary":
            import pickle
            graph_data = {}
            for node_id, node in self.graph.items():
                graph_data[node_id] = {
                    "node_id": node.node_id,
                    "vector": node.vector,
                    "neighbors": node.neighbors,
                    "metadata": self.metadata.get(node_id)
                }
            data = {
                "m": self.m,
                "m0": self.m0,
                "ef_construction": self.ef_construction,
                "level_mult": self.level_mult,
                "distance_metric": self.distance_metric,
                "entry_point": self.entry_point,
                "max_level": self.max_level,
                "total_inserted": self.total_inserted,
                "graph": graph_data
            }
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"HNSW Index saved to {filepath} (binary)")
            return
        
        # Default JSON serialization
        def make_serializable(obj):
            if obj is None:
                return None
            if isinstance(obj, (str, int, float, bool)):
                return obj
            if isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            if isinstance(obj, dict):
                return {str(k): make_serializable(v) for k, v in obj.items()}
            return str(obj)
        
        graph_data = {}
        for node_id, node in self.graph.items():
            metadata = self.metadata.get(node_id)
            serializable_metadata = make_serializable(metadata) if metadata else None
            graph_data[node_id] = {
                "node_id": node.node_id,
                "vector": node.vector.tolist(),
                "neighbors": {str(k): v for k, v in node.neighbors.items()},
                "metadata": serializable_metadata
            }
        
        index_data = {
            "m": self.m,
            "m0": self.m0,
            "ef_construction": self.ef_construction,
            "level_mult": self.level_mult,
            "distance_metric": self.distance_metric,
            "entry_point": self.entry_point,
            "max_level": self.max_level,
            "total_inserted": self.total_inserted,
            "deleted": list(self.deleted),
            "graph": graph_data
        }

        with open(filepath, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        logger.info(f"HNSW Index saved to {filepath}")

    def save_binary(self, directory: str):
        """
        Fast binary persistence using numpy .npy files for vector data and
        pickle for graph topology.  Achieves sub-second load times even for
        large indexes (vs multi-second JSON parsing).

        Directory layout:
            <directory>/
                config.json       — index parameters
                vectors.npy       — (N, D) float32 matrix
                id_map.json       — ordered list of vector IDs (index → ID)
                topology.pkl      — {node_id: {level: [neighbor_ids]}}
                metadata.pkl      — {node_id: metadata_dict}

        Args:
            directory: Directory to save index files into
        """
        os.makedirs(directory, exist_ok=True)

        # 1. Config
        config = {
            "m": self.m,
            "m0": self.m0,
            "ef_construction": self.ef_construction,
            "level_mult": self.level_mult,
            "distance_metric": self.distance_metric,
            "entry_point": self.entry_point,
            "max_level": self.max_level,
            "total_inserted": self.total_inserted,
            "deleted": list(self.deleted),
        }
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump(config, f)

        # 2. Vectors — contiguous float32 matrix
        if self.graph:
            id_list = list(self.graph.keys())
            vec_matrix = np.stack([self.graph[vid].vector for vid in id_list])
            np.save(os.path.join(directory, "vectors.npy"), vec_matrix)
            with open(os.path.join(directory, "id_map.json"), "w") as f:
                json.dump(id_list, f)
        else:
            np.save(os.path.join(directory, "vectors.npy"), np.empty((0, 0)))
            with open(os.path.join(directory, "id_map.json"), "w") as f:
                json.dump([], f)

        # 3. Topology — just neighbor dicts, no vector data
        import pickle
        topology = {
            node_id: dict(node.neighbors)
            for node_id, node in self.graph.items()
        }
        with open(os.path.join(directory, "topology.pkl"), "wb") as f:
            pickle.dump(topology, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 4. Metadata
        with open(os.path.join(directory, "metadata.pkl"), "wb") as f:
            pickle.dump(dict(self.metadata), f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            f"HNSW Index saved to {directory} (binary, "
            f"{len(self.graph)} vectors)"
        )

    def load_binary(self, directory: str) -> 'HNSWIndex':
        """
        Fast binary load from save_binary() directory.

        Args:
            directory: Directory containing index files

        Returns:
            self
        """
        import pickle

        # 1. Config
        with open(os.path.join(directory, "config.json"), "r") as f:
            config = json.load(f)

        self.m = config["m"]
        self.m0 = config["m0"]
        self.ef_construction = config["ef_construction"]
        self.level_mult = config["level_mult"]
        self.distance_metric = config.get("distance_metric", self.distance_metric)
        self.entry_point = config["entry_point"]
        self.max_level = config["max_level"]
        self.total_inserted = config["total_inserted"]
        self.deleted = set(config.get("deleted", []))

        # 2. Vectors + ID map
        vec_matrix = np.load(
            os.path.join(directory, "vectors.npy"), allow_pickle=False
        )
        with open(os.path.join(directory, "id_map.json"), "r") as f:
            id_list = json.load(f)

        # 3. Topology
        with open(os.path.join(directory, "topology.pkl"), "rb") as f:
            topology = pickle.load(f)

        # 4. Metadata
        with open(os.path.join(directory, "metadata.pkl"), "rb") as f:
            self.metadata = pickle.load(f)

        # Reconstruct graph
        self.graph.clear()
        self.vectors.clear()

        for idx, node_id in enumerate(id_list):
            vector = vec_matrix[idx].astype(np.float32)
            node = Node(node_id=node_id, vector=vector)
            node.neighbors = topology.get(node_id, {})
            self.graph[node_id] = node
            self.vectors[node_id] = vector

        logger.info(
            f"HNSW Index loaded from {directory} "
            f"({len(self.graph)} vectors)"
        )
        return self

    def load(self, filepath: str) -> 'HNSWIndex':
        """
        Load the index from disk
        
        Args:
            filepath: Path to load the index from
            
        Returns:
            self
        """
        # Support loading from binary directory
        if os.path.isdir(filepath):
            return self.load_binary(filepath)

        # Detect format from file extension or content
        is_binary = filepath.endswith('.pkl') or filepath.endswith('.pickle')
        
        if is_binary:
            import pickle
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(filepath, 'r') as f:
                data = json.load(f)
        
        self.m = data["m"]
        self.m0 = data["m0"]
        self.ef_construction = data["ef_construction"]
        self.level_mult = data["level_mult"]
        self.distance_metric = data.get("distance_metric", self.distance_metric)
        self.entry_point = data["entry_point"]
        self.max_level = data["max_level"]
        self.total_inserted = data["total_inserted"]
        self.deleted = set(data.get("deleted", []))

        if is_binary:
            for node_id, node_data in data["graph"].items():
                vector_array = self._normalize_vector(node_data["vector"])
                node = Node(
                    node_id=node_data["node_id"],
                    vector=vector_array
                )
                node.neighbors = node_data["neighbors"]
                self.graph[node_id] = node
                self.vectors[node_id] = node.vector
                self.metadata[node_id] = node_data.get("metadata")
        else:
            for node_id, node_data in data["graph"].items():
                vector_array = np.array(node_data["vector"], dtype=np.float32)
                vector_array = self._normalize_vector(vector_array)
                node = Node(
                    node_id=node_data["node_id"],
                    vector=vector_array
                )
                node.neighbors = {int(k): v for k, v in node_data["neighbors"].items()}
                self.graph[node_id] = node
                self.vectors[node_id] = node.vector
                self.metadata[node_id] = node_data.get("metadata")
        
        logger.info(f"HNSW Index loaded from {filepath}")
        
        return self
    
    def clear(self):
        """
        Clear all data from the index
        """
        self.graph.clear()
        self.vectors.clear()
        self.metadata.clear()
        self.deleted.clear()
        self.entry_point = None
        self.max_level = 0
        self.total_inserted = 0
        self.total_connections_made = 0
