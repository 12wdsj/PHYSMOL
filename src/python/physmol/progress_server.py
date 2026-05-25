"""Tiny HTTP dashboard for watching PHYSMOL training progress."""

from __future__ import annotations

import argparse
import html
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>PHYSMOL Training Progress</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 32px; color: #17202a; }
    .bar { width: 100%; max-width: 820px; height: 20px; background: #e8edf3; border-radius: 6px; overflow: hidden; }
    .fill { height: 100%; background: #1f8a70; width: 0%; transition: width .4s; }
    pre { background: #f6f8fa; padding: 16px; border-radius: 6px; max-width: 820px; overflow: auto; }
    .muted { color: #667085; }
  </style>
</head>
<body>
  <h1>PHYSMOL Training Progress</h1>
  <p class="muted" id="path"></p>
  <h2 id="phase">idle</h2>
  <div class="bar"><div class="fill" id="fill"></div></div>
  <p id="summary"></p>
  <pre id="metrics">{}</pre>
  <script>
    async function refresh() {
      const r = await fetch('/progress.json?ts=' + Date.now());
      const p = await r.json();
      document.getElementById('path').textContent = p.path || '';
      document.getElementById('phase').textContent = p.phase || 'idle';
      document.getElementById('fill').style.width = (p.percent || 0) + '%';
      document.getElementById('summary').textContent =
        `${p.step || 0}/${p.total_steps || 0} (${(p.percent || 0).toFixed(1)}%) - ${p.message || ''}`;
      document.getElementById('metrics').textContent = JSON.stringify(p.metrics || {}, null, 2);
    }
    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>
"""


class ProgressHandler(BaseHTTPRequestHandler):
    progress_path: str = ""

    def do_GET(self):
        if self.path.startswith("/progress.json"):
            self._send_json(self._read_progress())
            return
        self._send_html(HTML)

    def log_message(self, format, *args):
        return

    def _read_progress(self) -> dict:
        if not os.path.exists(self.progress_path):
            return {
                "phase": "idle",
                "step": 0,
                "total_steps": 0,
                "percent": 0.0,
                "metrics": {},
                "message": "Waiting for progress file.",
                "path": self.progress_path,
            }
        with open(self.progress_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["path"] = self.progress_path
        return data

    def _send_json(self, data: dict):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_html(self, text: str):
        payload = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    parser = argparse.ArgumentParser(description="Serve PHYSMOL progress dashboard")
    parser.add_argument("--progress", default="./checkpoints/abstract_training/progress.json")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    ProgressHandler.progress_path = os.path.abspath(args.progress)
    server = ThreadingHTTPServer((args.host, args.port), ProgressHandler)
    print(f"Serving PHYSMOL progress at http://{args.host}:{args.port}")
    print(f"Progress file: {ProgressHandler.progress_path}")
    server.serve_forever()


if __name__ == "__main__":
    main()
