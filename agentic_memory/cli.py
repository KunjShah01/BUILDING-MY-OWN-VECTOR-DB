"""Agentic memory CLI — backed by the vector DB API.

Usage:
    agentic-memory add "I like pizza" --categories food
    agentic-memory search "what do I like" --user default
    agentic-memory list --user default
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError:
    httpx = None


class MemoryAPIClient:
    """Thin HTTP client for the vector DB /memories endpoints."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def _request(self, method: str, path: str, **kwargs) -> Any:
        if httpx is None:
            print("Error: httpx is required. pip install httpx", file=sys.stderr)
            sys.exit(1)
        url = f"{self.base_url}{path}"
        r = httpx.request(method, url, headers=self._headers(), **kwargs)
        r.raise_for_status()
        return r.json()

    def add(self, text: str, user_id: str = "default", categories: Optional[List[str]] = None):
        return self._request("POST", "/memories", json={
            "text": text, "categories": categories or [],
        }, params={"user_id": user_id})

    def search(self, query: str, user_id: str = "default", limit: int = 10):
        return self._request("POST", "/memories/search", json={
            "query": query, "user_id": user_id, "limit": limit,
        })

    def chat(self, message: str, user_id: str = "default"):
        return self._request("POST", "/memories/chat", json={
            "message": message, "user_id": user_id,
        })

    def consolidate(self, user_id: str = "default"):
        return self._request("POST", "/memories/consolidate", json={
            "user_id": user_id,
        })

    def list(self, user_id: str = "default", limit: int = 50):
        return self._request("GET", f"/memories?user_id={user_id}&limit={limit}")

    def delete(self, memory_id: str, user_id: str = "default"):
        return self._request("DELETE", f"/memories/{memory_id}?user_id={user_id}")


def main():
    parser = argparse.ArgumentParser(description="Agentic Memory CLI")
    parser.add_argument("--url", default="http://localhost:8000", help="Vector DB API URL")
    parser.add_argument("--api-key", help="API key")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Add a memory")
    p_add.add_argument("text", help="Memory text")
    p_add.add_argument("--user", default="default")
    p_add.add_argument("--categories", nargs="*", default=[])

    p_search = sub.add_parser("search", help="Search memories")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--user", default="default")
    p_search.add_argument("--limit", type=int, default=10)

    p_chat = sub.add_parser("chat", help="Chat with memories")
    p_chat.add_argument("message", help="Your message")
    p_chat.add_argument("--user", default="default")

    p_consolidate = sub.add_parser("consolidate", help="Consolidate memories")
    p_consolidate.add_argument("--user", default="default")

    p_list = sub.add_parser("list", help="List memories")
    p_list.add_argument("--user", default="default")
    p_list.add_argument("--limit", type=int, default=50)

    p_delete = sub.add_parser("delete", help="Delete a memory")
    p_delete.add_argument("memory_id", help="Memory ID")
    p_delete.add_argument("--user", default="default")

    args = parser.parse_args()
    client = MemoryAPIClient(base_url=args.url, api_key=args.api_key)

    if args.command == "add":
        result = client.add(args.text, args.user, args.categories)
        print(json.dumps(result, indent=2))
    elif args.command == "search":
        result = client.search(args.query, args.user, args.limit)
        print(json.dumps(result, indent=2))
    elif args.command == "chat":
        result = client.chat(args.message, args.user)
        print(result.get("response", json.dumps(result, indent=2)))
    elif args.command == "consolidate":
        result = client.consolidate(args.user)
        print(json.dumps(result, indent=2))
    elif args.command == "list":
        result = client.list(args.user, args.limit)
        print(json.dumps(result, indent=2))
    elif args.command == "delete":
        result = client.delete(args.memory_id, args.user)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
