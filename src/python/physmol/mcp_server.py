"""Minimal MCP-style stdio server for PHYSMOL tools.

This server speaks JSON-RPC over stdin/stdout and implements the common MCP
methods an agent needs first: initialize, tools/list, and tools/call.

Run:
    python -m physmol.mcp_server
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Optional

from .agent_bridge import PhysmolAgentBridge, SmallLLMClient
from .evolution import to_jsonable


SERVER_NAME = "physmol-cognitive-tools"
SERVER_VERSION = "0.1.0"


class MCPStdioServer:
    """Small JSON-RPC server exposing PHYSMOL tools."""

    def __init__(self, bridge: PhysmolAgentBridge):
        self.bridge = bridge

    def serve_forever(self):
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = self.handle(request)
            except Exception as exc:
                response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32603, "message": str(exc)},
                }
            if response is not None:
                _write_response(response)

    def handle(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params") or {}

        # Notifications have no id and should not produce a response.
        if req_id is None and method.startswith("notifications/"):
            return None

        if method == "initialize":
            return _result(req_id, {
                "protocolVersion": "2025-06-18",
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION,
                },
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"subscribe": False, "listChanged": False},
                },
            })

        if method == "tools/list":
            return _result(req_id, {"tools": self.bridge.list_tools()})

        if method == "tools/call":
            name = params.get("name", "")
            arguments = params.get("arguments") or {}
            result = self.bridge.call_tool(name, arguments)
            return _result(req_id, {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(to_jsonable(result), ensure_ascii=False, indent=2),
                    }
                ],
                "structuredContent": to_jsonable(result),
                "isError": False,
            })

        if method == "resources/list":
            return _result(req_id, {"resources": []})

        if method == "prompts/list":
            return _result(req_id, {"prompts": []})

        if method == "ping":
            return _result(req_id, {})

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"},
        }


def build_bridge_from_args(args) -> PhysmolAgentBridge:
    llm = None
    if args.llm_endpoint and args.llm_model:
        llm = SmallLLMClient(
            provider=args.llm_provider,
            endpoint=args.llm_endpoint,
            model=args.llm_model,
            api_key=args.llm_api_key,
        )
    return PhysmolAgentBridge(
        vsa_dim=args.vsa_dim,
        broca_checkpoint=args.broca_checkpoint,
        learning_dir=args.learning_dir,
        trace_dir=args.trace_dir,
        llm_client=llm,
    )


def main():
    parser = argparse.ArgumentParser(description="Run PHYSMOL MCP-style stdio server")
    parser.add_argument("--vsa-dim", type=int, default=4096)
    parser.add_argument("--broca-checkpoint", default="./checkpoints/broca/model")
    parser.add_argument("--learning-dir", default="./checkpoints/learning")
    parser.add_argument("--trace-dir", default="./checkpoints/evolution")
    parser.add_argument("--llm-provider", default="openai", choices=["openai", "ollama"])
    parser.add_argument("--llm-endpoint", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--llm-api-key", default="")
    args = parser.parse_args()

    server = MCPStdioServer(build_bridge_from_args(args))
    server.serve_forever()


def _result(req_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": to_jsonable(result)}


def _write_response(response: Dict[str, Any]):
    sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
