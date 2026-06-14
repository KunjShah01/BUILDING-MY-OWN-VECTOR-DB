"""
MCP (Model Context Protocol) Server (Phase 3: Agentic Connectors).

Exposes the vector database as MCP tools that can be used by AI agents
(Claude, Copilot, etc.) for semantic search and retrieval-augmented tasks.

Tools exposed:
  - search_vectors(query_text, collection_id, k) → search results
  - ingest_text(text, collection_id, metadata) → store a text as vector
  - get_collection_stats(collection_id) → collection information
  - hybrid_search(hybrid_query, collection_id, k) → cost-based search
  - add_memory(user_id, text, categories) → store a memory
  - search_memories(query, user_id, limit) → semantic memory search
  - chat_with_memories(message, user_id) → memory-augmented chat
  - consolidate_memories(user_id) → deduplicate memories
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MCPServer:
    """MCP server for vector database tools.

    In a production deployment, this would run as a standalone FastMCP
    application. For now, we provide the tool definitions and execution
    logic that can be mounted in the API server.

    Reference: https://modelcontextprotocol.io/
    """

    def __init__(self, api_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self._vector_service = None
        self._collection_service = None

    def set_services(self, vector_service, collection_service):
        """Inject service instances for direct in-process access."""
        self._vector_service = vector_service
        self._collection_service = collection_service

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return MCP-compatible tool definitions."""
    def _get_db(self):
        """Return a DB session from the vector service if available."""
        if self._vector_service is not None:
            return self._vector_service.db
        return None

    async def _http_get(self, path: str) -> Dict[str, Any]:
        import httpx
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        async with httpx.AsyncClient(base_url=self.api_url, headers=headers) as client:
            resp = await client.get(path)
            return resp.json()

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return MCP-compatible tool definitions."""
        return [
            {
                "name": "search_vectors",
                "description": "Search for semantically similar vectors in a collection",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query_text": {"type": "string", "description": "Search query text"},
                        "collection_id": {"type": "string", "description": "Collection to search in"},
                        "k": {"type": "integer", "description": "Number of results (1-100)", "default": 10},
                        "method": {"type": "string", "enum": ["hnsw", "ivf", "brute"], "default": "hnsw"},
                    },
                    "required": ["query_text", "collection_id"],
                },
            },
            {
                "name": "ingest_text",
                "description": "Embed text and store it as a vector in a collection",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text content to ingest"},
                        "collection_id": {"type": "string", "description": "Target collection"},
                        "metadata": {"type": "object", "description": "Optional metadata", "default": {}},
                    },
                    "required": ["text", "collection_id"],
                },
            },
            {
                "name": "get_collection_stats",
                "description": "Get statistics and info about a collection",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection_id": {"type": "string", "description": "Collection ID"},
                    },
                    "required": ["collection_id"],
                },
            },
            {
                "name": "hybrid_search",
                "description": "Cost-based hybrid search with metadata filters and semantic matching",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "hybrid_query": {
                            "type": "string",
                            "description": "Hybrid query DSL, e.g. (category = 'tech' AND price < 100) OR semantic_match(\"laptops\")",
                        },
                        "collection_id": {"type": "string", "description": "Collection to search in"},
                        "k": {"type": "integer", "description": "Number of results", "default": 10},
                    },
                    "required": ["hybrid_query", "collection_id"],
                },
            },
            {
                "name": "add_memory",
                "description": "Store a memory for a user",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "User ID"},
                        "text": {"type": "string", "description": "Memory text content"},
                        "categories": {
                            "type": "array", "items": {"type": "string"},
                            "description": "Optional category tags",
                        },
                    },
                    "required": ["user_id", "text"],
                },
            },
            {
                "name": "search_memories",
                "description": "Semantic search across user memories",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "user_id": {"type": "string", "description": "User ID"},
                        "limit": {"type": "integer", "description": "Max results (default 10)"},
                    },
                    "required": ["query", "user_id"],
                },
            },
            {
                "name": "chat_with_memories",
                "description": "Chat with memory-augmented context",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "User message"},
                        "user_id": {"type": "string", "description": "User ID"},
                    },
                    "required": ["message", "user_id"],
                },
            },
            {
                "name": "consolidate_memories",
                "description": "Deduplicate and merge similar memories",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "User ID"},
                    },
                    "required": ["user_id"],
                },
            },
            {
                "name": "list_collections",
                "description": "List all collections in the vector database",
                "inputSchema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "search_sparse",
                "description": "Search using sparse vector (SPLADE) retrieval",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection_id": {"type": "string"},
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 10},
                    },
                    "required": ["collection_id", "query"],
                },
            },
            {
                "name": "nl_search",
                "description": "Search using natural language — translate English to structured query",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_index_tuning_recommendations",
                "description": "Get AI-powered index tuning recommendations",
                "inputSchema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "get_query_cache_stats",
                "description": "Get query cache hit ratio and stats",
                "inputSchema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "flush_query_cache",
                "description": "Flush the query result cache",
                "inputSchema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "get_slow_queries",
                "description": "List recent slow queries for performance analysis",
                "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer", "default": 10}}, "required": []},
            },
        ]

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an MCP tool and return the result.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.

        Returns:
            Tool execution result.
        """
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        async with httpx.AsyncClient(base_url=self.api_url, headers=headers) as client:
            if tool_name == "search_vectors":
                text = arguments["query_text"]
                collection_id = arguments["collection_id"]
                k = arguments.get("k", 10)
                method = arguments.get("method", "hnsw")

                # Embed the text
                from services.embedding_service import embed_text
                vector = embed_text(text)

                resp = await client.post(
                    "/search",
                    json={
                        "query_vector": vector,
                        "k": k,
                        "method": method,
                    },
                    params={"collection_id": collection_id},
                )
                result = resp.json()

                # Format for MCP readability
                formatted = []
                for r in result.get("results", []):
                    meta = r.get("metadata", {}) or {}
                    formatted.append({
                        "vector_id": r.get("vector_id"),
                        "distance": r.get("distance"),
                        "text_snippet": (meta.get("text", "") or meta.get("content", ""))[:200],
                        "source": meta.get("source"),
                    })
                return {"results": formatted, "total": len(formatted)}

            elif tool_name == "ingest_text":
                text = arguments["text"]
                collection_id = arguments["collection_id"]
                metadata = arguments.get("metadata", {})

                resp = await client.post(
                    f"/collections/{collection_id}/ingest/text",
                    json={"text": text, "metadata": metadata},
                )
                return resp.json()

            elif tool_name == "get_collection_stats":
                collection_id = arguments["collection_id"]

                resp = await client.get(f"/collections/{collection_id}")
                col_data = resp.json()

                stats_resp = await client.get(
                    f"/collections/{collection_id}/index/stats"
                )
                index_stats = stats_resp.json().get("stats", {})

                return {
                    "collection": col_data.get("collection", {}),
                    "index": index_stats,
                }

            elif tool_name == "hybrid_search":
                hybrid_query = arguments["hybrid_query"]
                collection_id = arguments["collection_id"]
                k = arguments.get("k", 10)

                resp = await client.post(
                    "/search-engine/hybrid-query",
                    params={
                        "collection_id": collection_id,
                        "hybrid_query": hybrid_query,
                        "top_k": k,
                    },
                )
                return resp.json()

            elif tool_name == "add_memory":
                if not self._vector_service:
                    return {"error": "Memory service not available (in-process mode only)"}
                from services.memory_service import MemoryService
                svc = MemoryService(self._vector_service.db)
                return svc.add_memory(
                    user_id=arguments["user_id"],
                    text=arguments["text"],
                    categories=arguments.get("categories"),
                )

            elif tool_name == "search_memories":
                if not self._vector_service:
                    return {"error": "Memory service not available (in-process mode only)"}
                from services.memory_service import MemoryService
                svc = MemoryService(self._vector_service.db)
                return svc.search_memories(
                    query=arguments["query"],
                    user_id=arguments["user_id"],
                    limit=arguments.get("limit", 10),
                )

            elif tool_name == "chat_with_memories":
                if not self._vector_service:
                    return {"error": "Memory service not available (in-process mode only)"}
                from services.memory_service import MemoryService
                svc = MemoryService(self._vector_service.db)
                return svc.chat(
                    message=arguments["message"],
                    user_id=arguments["user_id"],
                )

            elif tool_name == "consolidate_memories":
                if not self._vector_service:
                    return {"error": "Memory service not available (in-process mode only)"}
                from services.memory_service import MemoryService
                svc = MemoryService(self._vector_service.db)
                return svc.consolidate(user_id=arguments["user_id"])

            elif tool_name == "list_collections":
                db = self._get_db()
                if db is None:
                    return {"error": "In-process mode requires set_services()"}
                from services.collection_service import CollectionService
                svc = CollectionService(db)
                return svc.list_collections()

            elif tool_name == "search_sparse":
                from services.sparse_service import SparseService
                db = self._get_db()
                if db is None:
                    return {"error": "In-process mode requires set_services()"}
                svc = SparseService(db)
                return svc.search_sparse(arguments["collection_id"], arguments["query"], arguments.get("k", 10))

            elif tool_name == "nl_search":
                from services.rag_service import openai_chat_completion
                import json
                prompt = f"Convert to JSON search query: {{\"text\": \"...\", \"categories\": [], \"limit\": 10}}. Query: {arguments['query']}"
                response = openai_chat_completion(messages=[{"role": "user", "content": prompt}], max_tokens=300, temperature=0.1)
                try:
                    structured = json.loads(response)
                except Exception:
                    structured = {"text": arguments["query"], "limit": arguments.get("limit", 10)}
                from services.search_engine_service import SearchEngineService
                engine = SearchEngineService(None)
                return {"success": True, "structured_query": structured, "note": "NL search via MCP"}

            elif tool_name == "get_index_tuning_recommendations":
                db = self._get_db()
                if db is None:
                    return {"error": "In-process mode requires set_services()"}
                from services.index_tuner import IndexTuner
                tuner = IndexTuner(db)
                return {"recommendations": tuner.get_recommendations()}

            elif tool_name == "get_query_cache_stats":
                from services.query_cache import query_cache
                return {"success": True, "stats": query_cache.stats()}

            elif tool_name == "flush_query_cache":
                from services.query_cache import query_cache
                query_cache.flush()
                return {"success": True, "message": "Cache flushed"}

            elif tool_name == "get_slow_queries":
                from services.slow_query_analyzer import slow_query_analyzer
                return {"success": True, "slow_queries": slow_query_analyzer.get_recent(limit=arguments.get("limit", 10))}

            else:
                return {"error": f"Unknown tool: {tool_name}"}
