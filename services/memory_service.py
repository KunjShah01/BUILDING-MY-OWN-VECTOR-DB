"""Memory service: agentic memory CRUD, search, chat, consolidation.

Stores memories in PostgreSQL via pgvector, embeds text via the existing
embedding_service, and uses the RAG service's LLM helper for chat.
"""

from __future__ import annotations

import uuid
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import text

from database.schema import Memory
from services.embedding_service import embed_text

logger = logging.getLogger(__name__)


def _mem_to_dict(m: Memory) -> Dict[str, Any]:
    return {
        "memory_id": m.memory_id,
        "user_id": m.user_id,
        "text": m.text,
        "categories": list(m.categories) if m.categories else [],
        "metadata": m.meta_data,
        "created_at": str(m.created_at) if m.created_at else None,
        "updated_at": str(m.updated_at) if m.updated_at else None,
    }


class MemoryService:
    """CRUD + search + chat + consolidation for agentic memories."""

    def __init__(self, db: Session):
        self.db = db

    @property
    def db_session(self) -> Session:
        """Expose db for MCP tool access."""
        return self.db

    def add_memory(
        self,
        user_id: str,
        text: str,
        categories: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Embed text and store as a memory."""
        memory_id = str(uuid.uuid4())
        embedding = embed_text(text)
        mem = Memory(
            memory_id=memory_id,
            user_id=user_id,
            text=text,
            embedding=embedding,
            categories=categories or [],
            meta_data=metadata or {},
        )
        self.db.add(mem)
        self.db.commit()
        self.db.refresh(mem)
        return {"success": True, "memory": _mem_to_dict(mem)}

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Fetch a single Memory ORM object by memory_id."""
        return self.db.query(Memory).filter(Memory.memory_id == memory_id).first()

    def get_memories(
        self,
        user_id: str,
        category: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List memories for a user, optionally filtered by category."""
        q = self.db.query(Memory).filter(Memory.user_id == user_id)
        if category:
            q = q.filter(Memory.categories.any(category))
        total = q.count()
        memories = q.order_by(Memory.created_at.desc()).offset(offset).limit(limit).all()
        return {
            "success": True,
            "memories": [_mem_to_dict(m) for m in memories],
            "total": total,
        }

    def update_memory(
        self,
        memory_id: str,
        user_id: str,
        text: Optional[str] = None,
        categories: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update memory text (re-embeds), categories, or metadata."""
        mem = self.db.query(Memory).filter(
            Memory.memory_id == memory_id, Memory.user_id == user_id,
        ).first()
        if not mem:
            return {"success": False, "message": "Memory not found"}
        if text is not None:
            mem.text = text
            mem.embedding = embed_text(text)
        if categories is not None:
            mem.categories = categories
        if metadata is not None:
            mem.meta_data = metadata
        self.db.commit()
        self.db.refresh(mem)
        return {"success": True, "memory": _mem_to_dict(mem)}

    def delete_memory(self, memory_id: str, user_id: str) -> Dict[str, Any]:
        """Delete a memory by ID."""
        mem = self.db.query(Memory).filter(
            Memory.memory_id == memory_id, Memory.user_id == user_id,
        ).first()
        if not mem:
            return {"success": False, "message": "Memory not found"}
        self.db.delete(mem)
        self.db.commit()
        return {"success": True, "message": "Memory deleted"}

    def search_memories(
        self,
        query: str,
        user_id: str,
        categories: Optional[List[str]] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Semantic search over memories using pgvector cosine distance."""
        query_vec = embed_text(query)
        query_vec_str = f"[{','.join(str(v) for v in query_vec)}]"

        params: Dict[str, Any] = {
            "query_vec": query_vec_str,
            "user_id": user_id,
            "limit": limit,
        }
        where_clauses = ["m.user_id = :user_id"]

        if categories:
            placeholders = [f":cat_{i}" for i in range(len(categories))]
            where_clauses.append(f"m.categories && ARRAY[{','.join(placeholders)}]")
            for i, cat in enumerate(categories):
                params[f"cat_{i}"] = cat

        where_sql = " AND ".join(where_clauses)

        sql = text(f"""
            SELECT m.memory_id, m.text, m.categories, m.metadata,
                   m.created_at, m.updated_at,
                   1 - (m.embedding <=> :query_vec::vector) AS similarity
            FROM memories m
            WHERE {where_sql}
            ORDER BY similarity DESC
            LIMIT :limit
        """)

        rows = self.db.execute(sql, params).fetchall()
        results = []
        for row in rows:
            results.append({
                "memory_id": row.memory_id,
                "text": row.text,
                "categories": list(row.categories) if row.categories else [],
                "metadata": row.metadata,
                "distance": float(round(1 - row.similarity, 6)),
                "created_at": str(row.created_at) if row.created_at else None,
            })

        return {"success": True, "results": results, "total": len(results)}

    def chat(
        self,
        message: str,
        user_id: str,
        history: Optional[List[Dict[str, str]]] = None,
        llm_model: str = "gpt-4o-mini",
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """Memory-augmented chat: retrieve relevant memories, build context, call LLM."""
        from services.rag_service import openai_chat_completion

        search_result = self.search_memories(message, user_id, limit=5)
        memories = search_result.get("results", [])

        context_parts = []
        for m in memories:
            cats = ", ".join(m.get("categories", [])) if m.get("categories") else ""
            cat_tag = f" [{cats}]" if cats else ""
            context_parts.append(f"- {m['text']}{cat_tag}")

        memory_context = "\n".join(context_parts) if context_parts else "No relevant memories found."

        system_msg = (
            "You are a helpful assistant with access to the user's personal memories. "
            "Use the following memories to provide contextually relevant answers. "
            "If the memories don't contain relevant information, just answer normally.\n\n"
            f"Relevant memories:\n{memory_context}"
        )

        messages = [{"role": "system", "content": system_msg}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": message})

        response = openai_chat_completion(
            messages=messages,
            model=llm_model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return {
            "success": True,
            "response": response,
            "memories_retrieved": len(memories),
        }

    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """Aggregated memory statistics for a user."""
        total = self.db.query(Memory).filter(Memory.user_id == user_id).count()
        by_category = {}
        rows = self.db.query(Memory.categories).filter(Memory.user_id == user_id).all()
        for row in rows:
            for cat in (row.categories or []):
                by_category[cat] = by_category.get(cat, 0) + 1
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        recent_7d = self.db.query(Memory).filter(
            Memory.user_id == user_id,
            Memory.created_at >= now - timedelta(days=7),
        ).count()
        recent_30d = self.db.query(Memory).filter(
            Memory.user_id == user_id,
            Memory.created_at >= now - timedelta(days=30),
        ).count()
        return {
            "success": True,
            "total": total,
            "by_category": by_category,
            "recent_7d": recent_7d,
            "recent_30d": recent_30d,
        }

    def batch_delete(self, user_id: str, memory_ids: List[str]) -> Dict[str, Any]:
        """Delete multiple memories at once."""
        deleted = self.db.query(Memory).filter(
            Memory.memory_id.in_(memory_ids),
            Memory.user_id == user_id,
        ).delete(synchronize_session="fetch")
        self.db.commit()
        return {"success": True, "deleted": deleted}

    async def chat_stream(
        self,
        message: str,
        user_id: str,
        history: Optional[List[Dict[str, str]]] = None,
        llm_model: str = "gpt-4o-mini",
        max_tokens: int = 500,
        temperature: float = 0.3,
    ):
        """Memory-augmented chat with SSE streaming."""
        import asyncio
        import json as _json
        from openai import AsyncOpenAI
        import os

        search_result = self.search_memories(message, user_id, limit=5)
        memories = search_result.get("results", [])

        context_parts = []
        for m in memories:
            cats = ", ".join(m.get("categories", [])) if m.get("categories") else ""
            cat_tag = f" [{cats}]" if cats else ""
            context_parts.append(f"- {m['text']}{cat_tag}")

        memory_context = "\n".join(context_parts) if context_parts else "No relevant memories found."

        system_msg = (
            "You are a helpful assistant with access to the user's personal memories. "
            "Use the following memories to provide contextually relevant answers. "
            "If the memories don't contain relevant information, just answer normally.\n\n"
            f"Relevant memories:\n{memory_context}"
        )

        yield _json.dumps({"type": "metadata", "memories_retrieved": len(memories)})

        messages = [{"role": "system", "content": system_msg}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": message})

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        stream = await client.chat.completions.create(
            model=llm_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield _json.dumps({"type": "token", "content": chunk.choices[0].delta.content})

    def consolidate(self, user_id: str) -> Dict[str, Any]:
        """LLM-based dedup: find duplicate/mergeable memories and merge them."""
        from services.rag_service import openai_chat_completion
        import json as _json

        memories = self.db.query(Memory).filter(Memory.user_id == user_id).all()
        if len(memories) < 2:
            return {
                "success": True,
                "merged": 0,
                "deleted": 0,
                "message": "Less than 2 memories, nothing to consolidate",
            }

        mem_texts = [f"{m.memory_id}: {m.text}" for m in memories]
        prompt = (
            "Analyze these memories for duplicates or merge candidates. "
            "Group similar/duplicate memory IDs and suggest which to keep "
            "and which to merge/delete.\n\n"
            + "\n".join(mem_texts)
            + "\n\nRespond as JSON: "
            '{"groups": [{"keep": "id", "merge_ids": ["id1", "id2"], '
            '"merged_text": "..."}]}'
        )

        response = openai_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.1,
        )

        merged_count = 0
        deleted_count = 0
        try:
            data = _json.loads(response)
            for group in data.get("groups", []):
                keep_id = group.get("keep")
                merge_ids = group.get("merge_ids", [])
                merged_text = group.get("merged_text", "")
                if not keep_id or not merge_ids:
                    continue
                keep_mem = self.db.query(Memory).filter(
                    Memory.memory_id == keep_id
                ).first()
                if keep_mem:
                    keep_mem.text = merged_text
                    keep_mem.embedding = embed_text(merged_text)
                    self.db.flush()
                for mid in merge_ids:
                    m = self.db.query(Memory).filter(Memory.memory_id == mid).first()
                    if m:
                        self.db.delete(m)
                        deleted_count += 1
                merged_count += 1
            self.db.commit()
        except Exception as exc:
            logger.warning("Consolidation parsing failed: %s", exc)
            return {
                "success": False,
                "merged": 0,
                "deleted": 0,
                "message": f"Consolidation error: {exc}",
            }

        return {
            "success": True,
            "merged": merged_count,
            "deleted": deleted_count,
            "message": f"Merged {merged_count} groups, deleted {deleted_count} duplicates",
        }
