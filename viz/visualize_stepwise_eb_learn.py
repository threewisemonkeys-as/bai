#!/usr/bin/env python3
"""Visualize stepwise_eb_learn.py logs with a local web server."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from stepwise_eb_viewer_data import (
    is_safe_step_image_path,
    load_combined_trajectory,
    load_experiment_timeline,
    load_log_dir,
    load_qa_timeline,
    load_step_detail,
    load_trajectory,
    resolve_log_dir,
)


ASSET_DIR = Path(__file__).resolve().parent / "stepwise_eb_learn"
INDEX_HTML_PATH = ASSET_DIR / "index.html"
APP_JS_PATH = ASSET_DIR / "app.js"


def load_asset(path):
    with open(path, "r") as f:
        return f.read()


def render_index_html():
    config = {
        "mode": "dynamic",
        "allowDynamicInput": True,
        "dataIndexPath": None,
    }
    html = load_asset(INDEX_HTML_PATH)
    injection = (
        "<script>"
        f"window.STEPWISE_EB_VIEWER_CONFIG = {json.dumps(config)};"
        "</script>"
    )
    return html.replace("<!-- CONFIG_INJECTION -->", injection)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _get_log_dir(self, params):
        raw = params.get("log_dir", [None])[0]
        if not raw:
            raise ValueError("Missing log_dir parameter")
        return resolve_log_dir(raw)

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            params = parse_qs(parsed.query)

            if path in ("/", "/index.html"):
                self._html_response(render_index_html())
            elif path == "/app.js":
                self._text_response(load_asset(APP_JS_PATH), "application/javascript; charset=utf-8")
            elif path == "/api/data":
                log_dir = self._get_log_dir(params)
                self._json_response(load_log_dir(log_dir))
            elif path == "/api/step_detail":
                log_dir = self._get_log_dir(params)
                episode = int(params.get("episode", [0])[0])
                step = int(params.get("step", [0])[0])
                self._json_response(load_step_detail(log_dir, episode, step))
            elif path == "/api/trajectory":
                log_dir = self._get_log_dir(params)
                episode = int(params.get("episode", [0])[0])
                self._json_response(load_trajectory(log_dir, episode))
            elif path == "/api/combined_trajectory":
                log_dir = self._get_log_dir(params)
                self._json_response(load_combined_trajectory(log_dir))
            elif path == "/api/experiment_timeline":
                log_dir = self._get_log_dir(params)
                self._json_response(load_experiment_timeline(log_dir))
            elif path == "/api/qa_timeline":
                log_dir = self._get_log_dir(params)
                self._json_response(load_qa_timeline(log_dir))
            elif path == "/api/step_image":
                log_dir = self._get_log_dir(params)
                episode = int(params.get("episode", [0])[0])
                step = int(params.get("step", [0])[0])
                name = params.get("name", ["obs_before.png"])[0]
                if name not in ("obs_before.png", "obs_after.png") and not is_safe_step_image_path(name):
                    self.send_error(400)
                    return
                step_dir = (Path(log_dir) / f"episode_{episode}" / f"step_{step:03d}").resolve()
                img_path = (step_dir / name).resolve()
                if step_dir not in img_path.parents and img_path != step_dir:
                    self.send_error(400)
                    return
                if img_path.is_file():
                    self._image_response(img_path)
                else:
                    self.send_error(404)
            else:
                self.send_error(404)
        except Exception as exc:
            err = f"[visualize] request failed for {self.path}: {exc}\n{traceback.format_exc()}"
            print(err, file=sys.stderr, flush=True)
            try:
                self._json_response({"error": str(exc)})
            except Exception:
                pass

    def _json_response(self, obj):
        data = json.dumps(obj, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _html_response(self, html):
        data = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _text_response(self, text, content_type):
        data = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _image_response(self, img_path):
        data = img_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser(description="Visualize stepwise_eb_learn.py logs")
    parser.add_argument(
        "log_dir",
        nargs="?",
        default=None,
        help="Path to log directory (optional; can also be set in the browser)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8766, help="HTTP port (default: 8766)")
    parser.add_argument("--open-browser", action="store_true", help="Open browser automatically")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), Handler)
    display_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    url = f"http://{display_host}:{args.port}"

    if args.log_dir:
        try:
            resolved = resolve_log_dir(args.log_dir)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Log dir resolved to: {resolved}")
        data = load_log_dir(resolved)
        print(f"Found {len(data['episodes'])} episodes, {len(data['steps'])} steps")
        url += f"?log_dir={quote(resolved, safe='')}"

    print(f"Serving at {url}")

    if args.open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
