"""High-level memory operations. Pure stdlib, no DSPy."""

import json
from datetime import datetime
from typing import Optional, List, Dict, Any

from mem.embedder import get_embedder
from mem.store import MemoryStore, MemoryRecord, SearchResult
from mem.llm_client import LLMClient


_default_embedder = None
_default_store = None
_default_llm = None


def _get_embedder():
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = get_embedder()
    return _default_embedder


def _get_store() -> MemoryStore:
    global _default_store
    if _default_store is None:
        import os
        path = os.environ.get("MEM_STORE_PATH", "./memory_store.json")
        _default_store = MemoryStore(persist_path=path)
        _default_store.initialize()
    return _default_store


def _get_llm() -> LLMClient:
    global _default_llm
    if _default_llm is None:
        import os
        provider = os.environ.get("MEM_LLM_PROVIDER", "ollama")
        _default_llm = LLMClient(provider=provider)
    return _default_llm


async def add_memory(
    user_id: int, memory_text: str, categories: List[str]
) -> str:
    embedder = _get_embedder()
    store = _get_store()
    vector = embedder.embed(memory_text)
    store.insert([
        MemoryRecord(
            point_id="",
            user_id=user_id,
            memory_text=memory_text,
            categories=categories,
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            embedding=vector,
        )
    ])
    return f"Memory stored: '{memory_text}'"


async def search_memories(
    query: str, user_id: int, categories: Optional[List[str]] = None, limit: int = 5
) -> str:
    embedder = _get_embedder()
    store = _get_store()
    vector = embedder.embed(query)
    results = store.search(vector, user_id=user_id, categories=categories, limit=limit)
    if not results:
        return "No memories found."
    return "\n---\n".join(
        f"{r.memory_text} (Categories: {r.categories}) Relevance: {r.score:.2f}"
        for r in results
    )


async def update_memory(
    point_id: str, user_id: int, memory_text: str, categories: List[str]
) -> str:
    store = _get_store()
    store.delete([point_id])
    embedder = _get_embedder()
    vector = embedder.embed(memory_text)
    store.insert([
        MemoryRecord(
            point_id=point_id,
            user_id=user_id,
            memory_text=memory_text,
            categories=categories,
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            embedding=vector,
        )
    ])
    return f"Memory updated to: '{memory_text}'"


async def delete_memory(point_id: str) -> str:
    store = _get_store()
    store.delete([point_id])
    return f"Memory {point_id} deleted."


async def get_categories(user_id: int) -> str:
    store = _get_store()
    cats = store.get_categories(user_id)
    if not cats:
        return "No categories found."
    return "\n".join(cats)


async def chat(
    message: str,
    user_id: int,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Memory-augmented chat. Retrieves relevant memories, responds, saves new info."""
    llm = _get_llm()
    embedder = _get_embedder()
    store = _get_store()

    history = conversation_history or []

    # Retrieve relevant memories
    query_vec = embedder.embed(message)
    memories = store.search(query_vec, user_id=user_id, limit=5)

    memory_context = ""
    if memories:
        memory_context = "Relevant memories:\n" + "\n".join(
            f"- {r.memory_text}" for r in memories
        )

    # Build system prompt with memory context
    system_prompt = (
        "You are a memory-augmented AI assistant. You have access to the "
        "following memories about this user from past conversations.\n\n"
        + (memory_context or "No relevant memories found.")
        + "\n\nRespond naturally. If the user provides new information about "
        "themselves, make a note of it (the system will save it automatically)."
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    response = llm.chat(messages)
    if response is None:
        # Fallback if no LLM available
        if memories:
            response = "Based on what I remember:\n" + "\n".join(
                f"- {r.memory_text}" for r in memories
            )
        else:
            response = "I don't have any relevant memories about that."

    return response


async def consolidate_memories(user_id: int) -> str:
    """Merge duplicate/related memories using LLM. No DSPy needed."""
    store = _get_store()
    llm = _get_llm()
    embedder = _get_embedder()

    all_memories = store.fetch_all(user_id)
    if len(all_memories) < 2:
        return "Not enough memories to consolidate."

    memory_texts = "\n".join(
        f"[{i}] {m.memory_text}" for i, m in enumerate(all_memories)
    )

    prompt = (
        "You are a memory consolidation agent. Review these memory entries "
        "and identify duplicates or related items that should be merged.\n\n"
        f"{memory_texts}\n\n"
        "Return a JSON list of actions:\n"
        '- {"action": "merge", "indices": [0, 1], "text": "merged text"}\n'
        '- {"action": "delete", "indices": [2]}\n'
        '- {"action": "noop"}\n'
        "Only merge/suggest if they are clearly related."
    )

    result = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2048,
    )

    if not result:
        return "Consolidation skipped (no LLM response)."

    return result
