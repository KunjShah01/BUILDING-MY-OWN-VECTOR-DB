"""
Search Engine Service: Orchestration layer over the Vector DB.

Implements a multi-stage retrieval pipeline:
1. Rewrite: Spellcheck, synonym expansion (mocked/stubbed for now)
2. Recall: BM25 Sparse Search + HNSW Dense Vector Search
3. Fusion: Reciprocal Rank Fusion (RRF)
4. Rerank: Cross-encoder reranking
5. Faceted Aggregation: Metadata aggregation for UI sidebars
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
import math
import os

from services.cache_service import CacheService
from utils.bm25_index import BM25Index
from utils.query_planner import plan_query, QueryPlan

class SearchEngineService:
    def __init__(self, vector_service, gnn_service=None, reranker_service=None):
        self.vector_service = vector_service
        self.gnn_service = gnn_service
        self.reranker_service = reranker_service
        self.cache_service = CacheService()
        self._bm25_cache = {}
        
    def _get_bm25_index(self, collection_id: str) -> Optional[BM25Index]:
        """Lazy load BM25 index for the collection."""
        if collection_id in self._bm25_cache:
            return self._bm25_cache[collection_id]
        path = f"indexes/{collection_id}/sparse.json"
        if os.path.exists(path):
            try:
                idx = BM25Index.load(path)
                self._bm25_cache[collection_id] = idx
                return idx
            except Exception:
                pass
        return None
        
    def _detect_intent(self, query: str) -> str:
        """Lightweight intent classification."""
        query_lower = query.lower()
        if any(w in query_lower for w in ["what", "how", "why", "who", "when", "?"]):
            return "QA"
        elif len(query.split()) > 7:
            return "EXPLORATORY"
        return "EXACT"

    def _reciprocal_rank_fusion(self, dense_results: List[Dict], sparse_results: List[Dict], k: int = 60) -> List[Dict]:
        """Fuses two lists of ranked results using Reciprocal Rank Fusion."""
        scores = defaultdict(float)
        items = {}
        
        for rank, res in enumerate(dense_results):
            vid = res["vector_id"]
            items[vid] = res
            scores[vid] += 1.0 / (k + rank + 1)
            
        for rank, res in enumerate(sparse_results):
            vid = res["vector_id"]
            if vid not in items:
                items[vid] = res
            scores[vid] += 1.0 / (k + rank + 1)
            
        fused = []
        for vid, score in scores.items():
            item = items[vid].copy()
            item["rrf_score"] = score
            fused.append(item)
            
        fused.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused
        
    def _aggregate_facets(self, results: List[Dict]) -> Dict[str, Dict[str, int]]:
        """Groups and counts metadata fields across the result set."""
        facets = defaultdict(lambda: defaultdict(int))
        
        for res in results:
            meta = res.get("metadata", {})
            for key, value in meta.items():
                if isinstance(value, str) and len(value) < 50:
                    facets[key][value] += 1
                elif isinstance(value, list):
                    for v in value:
                        if isinstance(v, str) and len(v) < 50:
                            facets[key][v] += 1
                            
        return {k: dict(v) for k, v in facets.items()}

    def search(
        self,
        query: str,
        query_vector: List[float],
        collection_id: str,
        tenant_id: Optional[str] = None,
        top_k: int = 10,
        enable_gnn: bool = False
    ) -> Dict[str, Any]:
        """Full orchestrated search experience."""
        # Check Redis Cache
        cached = self.cache_service.get_cached_search(query_vector, top_k, collection_id)
        if cached:
            return cached
            
        intent = self._detect_intent(query)
        
        # 1. Recall: Dense Search
        dense_res = self.vector_service.search_vectors(
            query_vector, k=top_k * 5, collection_id=collection_id, tenant_id=tenant_id
        )
        dense_candidates = dense_res.get("results", [])
        
        # 1b. Recall: Sparse Search (BM25)
        sparse_idx = self._get_bm25_index(collection_id)
        if sparse_idx:
            sparse_raw = sparse_idx.search(query, k=top_k * 5)
            sparse_candidates = [
                {"vector_id": doc_id, "score": float(score), "metadata": {}}
                for doc_id, score in sparse_raw
            ]
        else:
            sparse_candidates = []
        
        # 2. Fusion
        if sparse_candidates:
            candidates = self._reciprocal_rank_fusion(dense_candidates, sparse_candidates)
        else:
            candidates = dense_candidates
            
        # 3. Rerank
        if self.reranker_service and intent == "QA":
            # For QA, cross-encoder reranking is highly beneficial
            try:
                # Stubbing actual call if signature differs, 
                # assuming rerank(query, docs, top_k)
                candidates = self.reranker_service.rerank(query, candidates, top_k=top_k*2)
            except Exception as e:
                print(f"Reranking failed: {e}")
                
        # 4. GNN Rerank (Link Prediction discovery)
        if enable_gnn and self.gnn_service and intent == "EXPLORATORY":
            # GNN is great for exploratory search to find related content
            candidates = self.gnn_service.graph_rerank(
                query_vector, candidates[:top_k*3], collection_id, tenant_id
            )
            
        # Final Truncation
        final_results = candidates[:top_k]
        
        # 5. Faceted Aggregation
        facets = self._aggregate_facets(final_results)
        
        result = {
            "success": True,
            "intent": intent,
            "results": final_results,
            "facets": facets,
            "total_results": len(final_results)
        }
        
        # Cache the result
        self.cache_service.cache_search(query_vector, top_k, collection_id, result)

        return result

    def planned_search(
        self,
        hybrid_query: str,
        query_vector: List[float],
        collection_id: str,
        tenant_id: Optional[str] = None,
        top_k: int = 10,
        field_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a hybrid query (metadata predicates + semantic_match) using the
        AST cost-based query planner to choose filter-first vs vector-first.

        Args:
            hybrid_query: e.g. "(category = 'tech' AND price < 100) OR semantic_match(\"laptops\")"
            query_vector: embedding for the semantic_match leaf
            field_stats: optional {field: {"distinct": N}} for selectivity refinement
        """
        plan: QueryPlan = plan_query(hybrid_query, stats=field_stats, default_k=top_k)
        pred = plan.predicate_fn

        if plan.strategy in ("vector_only", "vector_first"):
            # Vector search then (optionally) post-filter by metadata predicate
            over_fetch = top_k if plan.strategy == "vector_only" else top_k * 5
            dense = self.vector_service.search_vectors(
                query_vector, k=over_fetch, collection_id=collection_id, tenant_id=tenant_id
            ).get("results", [])
            if pred is not None:
                dense = [r for r in dense if pred(r.get("metadata"))]
            results = dense[:top_k]

        elif plan.strategy == "filter_first":
            # Filter candidates first (cheap, selective), then rank by vector
            over_fetch = top_k * 10
            dense = self.vector_service.search_vectors(
                query_vector, k=over_fetch, collection_id=collection_id, tenant_id=tenant_id
            ).get("results", [])
            filtered = [r for r in dense if pred is None or pred(r.get("metadata"))]
            results = filtered[:top_k]

        else:  # filter_only
            dense = self.vector_service.search_vectors(
                query_vector, k=top_k * 10, collection_id=collection_id, tenant_id=tenant_id
            ).get("results", [])
            results = [r for r in dense if pred is None or pred(r.get("metadata"))][:top_k]

        return {
            "success": True,
            "strategy": plan.strategy,
            "estimated_selectivity": plan.estimated_selectivity,
            "plan_reason": plan.reason,
            "results": results,
            "total_results": len(results),
        }
