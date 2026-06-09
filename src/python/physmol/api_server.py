"""OpenAI-compatible API server for PHYSMOL.

Exposes PHYSMOL's cognitive engine as a /v1/chat/completions endpoint,
so Hermes, OpenCode, or any OpenAI-compatible client can use it.

Usage:
    python -m physmol.api_server [--port 8420] [--host 127.0.0.1]

Then configure in Hermes config.yaml:
    custom_providers:
      physmol:
        base_url: http://127.0.0.1:8420/v1
        api_key: none
        models:
          - physmol-cognitive
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

from .agent_bridge import PhysmolAgentBridge
from .evolution import to_jsonable

# Global bridge instance (initialized once)
_bridge: Optional[PhysmolAgentBridge] = None


def get_bridge() -> PhysmolAgentBridge:
    global _bridge
    if _bridge is None:
        _bridge = PhysmolAgentBridge()
    return _bridge


def chat_completions(body: Dict[str, Any]) -> Dict[str, Any]:
    """Handle /v1/chat/completions request."""
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    # Extract the last user message as the query
    user_text = ""
    system_text = ""
    for msg in messages:
        if msg.get("role") == "system":
            system_text = msg.get("content", "")
        elif msg.get("role") == "user":
            user_text = msg.get("content", "")

    if not user_text:
        user_text = system_text or "status"

    bridge = get_bridge()

    # Check for tool/function call requests
    tools = body.get("tools", [])
    if tools:
        return _handle_tool_request(bridge, user_text, tools, body)

    # Regular chat completion
    try:
        response_text = bridge.chat(user_text, use_llm=True)
    except Exception as e:
        response_text = f"PHYSMOL error: {e}"

    model_name = body.get("model", "physmol-cognitive")
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    result = {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(user_text.split()),
            "completion_tokens": len(response_text.split()),
            "total_tokens": len(user_text.split()) + len(response_text.split()),
        },
    }

    if stream:
        return _stream_response(completion_id, model_name, response_text)
    return result


def _handle_tool_request(bridge, user_text, tools, body):
    """Handle function/tool call requests by routing to PHYSMOL tools."""
    model_name = body.get("model", "physmol-cognitive")
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # Map OpenAI tool names to PHYSMOL tool names
    tool_map = {
        "explain_concept": "explain",
        "predict": "predict",
        "reason": "reason",
        "knowledge_status": "knowledge_status",
        "status": "status",
    }

    # Try to extract tool call from the message
    tool_calls = []
    response_text = ""

    for tool in tools:
        func_name = tool.get("function", {}).get("name", "")
        physmol_name = tool_map.get(func_name, func_name)

        # If user message contains a question, try to auto-route
        if user_text:
            try:
                result = bridge.call_tool(physmol_name, {"text": user_text})
                result_str = json.dumps(to_jsonable(result), ensure_ascii=False, indent=2)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps({"text": user_text}, ensure_ascii=False),
                    },
                })
                response_text = result_str
            except Exception:
                continue

    if not tool_calls:
        # Fallback to regular chat
        try:
            response_text = bridge.chat(user_text, use_llm=True)
        except Exception as e:
            response_text = f"PHYSMOL error: {e}"

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _stream_response(completion_id, model_name, text):
    """Generator for streaming response (SSE format)."""
    chunks = []
    words = text.split()
    for i, word in enumerate(words):
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"content": word + " " if i < len(words) - 1 else word},
                "finish_reason": None,
            }],
        }
        chunks.append(chunk)

    # Final chunk
    chunks.append({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    })
    return chunks


def list_models() -> Dict[str, Any]:
    """Handle /v1/models request."""
    return {
        "object": "list",
        "data": [
            {
                "id": "physmol-cognitive",
                "object": "model",
                "created": 1700000000,
                "owned_by": "physmol",
                "permission": [],
            }
        ],
    }


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle each request in a new thread — prevents one slow Ollama call from blocking everything."""
    daemon_threads = True


class PhysmolAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OpenAI-compatible API."""

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/v1/models":
            self._json_response(list_models())
        elif parsed.path == "/health":
            self._json_response({"status": "ok"})
        elif parsed.path == "/v1/tools":
            bridge = get_bridge()
            tools = bridge.list_tools()
            self._json_response({"tools": tools})
        else:
            self._json_response({"error": "Not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError:
            self._json_response({"error": "Invalid JSON"}, 400)
            return

        if parsed.path == "/v1/chat/completions":
            try:
                result = chat_completions(body)
                # Check if it's a streaming response (list of chunks)
                if isinstance(result, list):
                    self._stream_response(result)
                else:
                    self._json_response(result)
            except Exception as e:
                self._json_response({"error": str(e)}, 500)
        else:
            self._json_response({"error": "Not found"}, 404)

    def _json_response(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def _stream_response(self, chunks):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        for chunk in chunks:
            line = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            self.wfile.write(line.encode("utf-8"))
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format, *args):
        # Suppress default logging
        pass


def main():
    parser = argparse.ArgumentParser(description="PHYSMOL OpenAI-compatible API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8420)
    args = parser.parse_args()

    # Initialize bridge eagerly
    print(f"Initializing PHYSMOL cognitive engine...")
    get_bridge()
    print(f"PHYSMOL API server ready at http://{args.host}:{args.port}")
    print(f"Endpoints:")
    print(f"  GET  /v1/models          - List available models")
    print(f"  POST /v1/chat/completions - Chat completions")
    print(f"  GET  /v1/tools            - List PHYSMOL tools")
    print(f"  GET  /health              - Health check")

    server = ThreadedHTTPServer((args.host, args.port), PhysmolAPIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
