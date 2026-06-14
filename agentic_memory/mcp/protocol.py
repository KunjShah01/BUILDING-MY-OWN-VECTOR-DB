"""MCP JSON-RPC 2.0 protocol over stdio. Zero external deps."""

import json
import sys
from typing import Any, Callable, Dict, List, Optional


class MCPServer:
    """MCP protocol server implementing JSON-RPC 2.0 over stdin/stdout.

    Supports stdio transport. No FastMCP, no external deps.
    """

    def __init__(self, name: str = "agentic-memory"):
        self.name = name
        self.tools: Dict[str, Callable] = {}
        self.tool_meta: Dict[str, dict] = {}

    def tool(self, name: str, description: str = "", parameters: dict = None):
        def decorator(func):
            self.tools[name] = func
            self.tool_meta[name] = {
                "name": name,
                "description": description or func.__doc__ or "",
                "inputSchema": parameters or {"type": "object", "properties": {}},
            }
            return func
        return decorator

    async def _handle_request(self, request: dict) -> Optional[dict]:
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": self.name, "version": "0.1.0"},
                },
            }
        elif method == "notifications/initialized":
            return None  # No response for notifications
        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": list(self.tool_meta.values())},
            }
        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            handler = self.tools.get(tool_name)
            if not handler:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Tool not found: {tool_name}"},
                }
            try:
                # Support both sync and async handlers
                result = handler(**arguments)
                if hasattr(result, "__await__"):
                    result = await result
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": str(result)}]},
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32603, "message": str(e)},
                }
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

    async def run_stdio_async(self):
        """Read JSON-RPC from stdin, write responses to stdout."""
        print(f"MCP server '{self.name}' running on stdio", file=sys.stderr)
        sys.stderr.flush()

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = await self._handle_request(request)
                if response is not None:
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()
            except json.JSONDecodeError:
                error_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }
                sys.stdout.write(json.dumps(error_resp) + "\n")
                sys.stdout.flush()
