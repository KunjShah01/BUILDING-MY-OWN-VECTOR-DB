"""MCP server exposing memory tools. Uses zero-dep protocol and pipeline."""

from mem_mcp.protocol import MCPServer
from mem.pipeline import (
    add_memory,
    search_memories,
    update_memory,
    delete_memory,
    get_categories,
    chat,
)

mcp = MCPServer("agentic-memory")


@mcp.tool(
    "add_memory",
    "Store a new memory fact for a user",
    {
        "type": "object",
        "properties": {
            "user_id": {"type": "integer", "description": "User's unique identifier"},
            "memory_text": {"type": "string", "description": "The factual statement to remember"},
            "categories": {"type": "array", "items": {"type": "string"}, "description": "Category tags"},
        },
        "required": ["user_id", "memory_text", "categories"],
    },
)
async def tool_add_memory(user_id: int, memory_text: str, categories: list):
    return await add_memory(user_id, memory_text, categories)


@mcp.tool(
    "search_memories",
    "Search stored memories by semantic similarity",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language query"},
            "user_id": {"type": "integer", "description": "User's unique identifier"},
            "categories": {"type": "array", "items": {"type": "string"}, "description": "Filter by categories"},
            "limit": {"type": "integer", "description": "Max results (default 5)"},
        },
        "required": ["query", "user_id"],
    },
)
async def tool_search_memories(query: str, user_id: int, categories: list = None, limit: int = 5):
    return await search_memories(query, user_id, categories, limit)


@mcp.tool(
    "update_memory",
    "Replace an existing memory",
    {
        "type": "object",
        "properties": {
            "point_id": {"type": "string", "description": "Memory ID to update"},
            "user_id": {"type": "integer", "description": "User's unique identifier"},
            "memory_text": {"type": "string", "description": "New factual statement"},
            "categories": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["point_id", "user_id", "memory_text", "categories"],
    },
)
async def tool_update_memory(point_id: str, user_id: int, memory_text: str, categories: list):
    return await update_memory(point_id, user_id, memory_text, categories)


@mcp.tool(
    "delete_memory",
    "Permanently remove a memory",
    {
        "type": "object",
        "properties": {
            "point_id": {"type": "string", "description": "Memory ID to delete"},
        },
        "required": ["point_id"],
    },
)
async def tool_delete_memory(point_id: str):
    return await delete_memory(point_id)


@mcp.tool(
    "get_categories",
    "List all memory categories for a user",
    {
        "type": "object",
        "properties": {
            "user_id": {"type": "integer", "description": "User's unique identifier"},
        },
        "required": ["user_id"],
    },
)
async def tool_get_categories(user_id: int):
    return await get_categories(user_id)


@mcp.tool(
    "chat",
    "Memory-augmented conversation",
    {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "User's message"},
            "user_id": {"type": "integer", "description": "User's unique identifier"},
            "conversation_history": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
                "description": "Previous messages",
            },
        },
        "required": ["message", "user_id"],
    },
)
async def tool_chat(message: str, user_id: int, conversation_history: list = None):
    return await chat(message, user_id, conversation_history)
