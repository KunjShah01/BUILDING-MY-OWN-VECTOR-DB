"""
Async SDK client for interacting with the Vector Database API.
Built on httpx.
"""

import httpx
from typing import List, Dict, Any, Optional
import json

_DEFAULT_TIMEOUT = 30.0  # seconds

class VectorDBClient:
    """Async Python SDK wrapper for the Vector DB API.
    
    Includes request timeout to prevent cascading failures.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None, tenant_id: Optional[str] = None, timeout: float = _DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if tenant_id:
            headers["X-Tenant-ID"] = tenant_id
            
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=10.0, read=timeout, write=timeout, pool=10.0),
        )
        
    async def close(self):
        """Close the underlying HTTP client."""
        await self.client.aclose()
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    # --- Vector Methods ---
        
    async def create_vector(
        self, 
        vector: List[float], 
        metadata: Optional[Dict[str, Any]] = None, 
        vector_id: Optional[str] = None,
        collection_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Insert a single vector."""
        payload = {"vector": vector}
        if metadata:
            payload["metadata"] = metadata
        if vector_id:
            payload["vector_id"] = vector_id
        if collection_id:
            payload["collection_id"] = collection_id
            
        res = await self.client.post("/vectors", json=payload)
        res.raise_for_status()
        return res.json()
        
    async def search(
        self, 
        query_vector: List[float], 
        k: int = 5, 
        method: str = "hnsw",
        collection_id: Optional[str] = None, 
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search for similar vectors."""
        payload = {"query_vector": query_vector, "k": k, "method": method}
        if collection_id:
            payload["collection_id"] = collection_id
        if filters:
            payload["filters"] = filters
            
        res = await self.client.post("/search", json=payload)
        res.raise_for_status()
        return res.json()
        
    # --- Collection & Ingestion Methods ---
        
    async def create_collection(
        self, 
        name: str, 
        dimension: int,
        collection_id: Optional[str] = None,
        modality: str = "text"
    ) -> Dict[str, Any]:
        """Create a new collection namespace."""
        payload = {"name": name, "dimension": dimension, "modality": modality}
        if collection_id:
            payload["collection_id"] = collection_id
            
        res = await self.client.post("/collections", json=payload)
        res.raise_for_status()
        return res.json()
        
    async def ingest_text(
        self, 
        collection_id: str, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Ingest raw text (server-side embedding)."""
        payload = {"text": text}
        if metadata:
            payload["metadata"] = metadata
            
        res = await self.client.post(f"/collections/{collection_id}/ingest/text", json=payload)
        res.raise_for_status()
        return res.json()
        
    async def search_text(
        self, 
        collection_id: str, 
        query: str, 
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search via natural language (server-side embedding)."""
        payload = {"query": query, "k": k}
        if filters:
            payload["filters"] = filters
            
        res = await self.client.post(f"/collections/{collection_id}/search/text", json=payload)
        res.raise_for_status()
        return res.json()
        
    # --- RAG Methods ---
    
    async def rag_query(
        self, 
        query: str, 
        collection_id: Optional[str] = None,
        strategy: str = "standard"
    ) -> Dict[str, Any]:
        """Perform a RAG pipeline query."""
        payload = {"query": query, "strategy": strategy}
        if collection_id:
            payload["collection_id"] = collection_id
            
        res = await self.client.post("/rag/query", json=payload)
        res.raise_for_status()
        return res.json()
