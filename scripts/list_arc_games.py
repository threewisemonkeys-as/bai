#!/usr/bin/env python3
"""List available ARC-AGI-3 games via the discovery endpoint.

Hits GET https://three.arcprize.org/api/games with an X-API-Key header and
prints the games your key can access. Anonymous access (no key) returns the
3 public games; a valid key unlocks the rest.

Usage:
    uv run scripts/list_arc_games.py
    ARC_API_KEY=... python scripts/list_arc_games.py
    python scripts/list_arc_games.py --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

try:  # load .env if python-dotenv is available (matches rest of repo)
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

BASE_URL = os.environ.get("ARC_BASE_URL", "https://three.arcprize.org")
ENDPOINT = f"{BASE_URL.rstrip('/')}/api/games"

# Check the env var names this repo / the ARC toolkit might use, in order.
KEY_ENV_VARS = ("ARC_API_KEY", "ARC_AGI_API_KEY", "ARCPRIZE_API_KEY")


def get_api_key() -> str | None:
    for name in KEY_ENV_VARS:
        val = os.environ.get(name)
        if val:
            return val
    return None


def fetch_games(api_key: str | None, timeout: float = 30.0) -> list[dict]:
    req = urllib.request.Request(ENDPOINT, method="GET")
    req.add_header("Accept", "application/json")
    if api_key:
        req.add_header("X-API-Key", api_key)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="List available ARC-AGI-3 games")
    parser.add_argument("--json", action="store_true", help="emit raw JSON")
    args = parser.parse_args()

    api_key = get_api_key()
    if not api_key:
        print(
            f"[!] No API key found (checked {', '.join(KEY_ENV_VARS)}). "
            "Fetching anonymous game list only.\n",
            file=sys.stderr,
        )

    try:
        games = fetch_games(api_key)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        print(f"HTTP {e.code} {e.reason} from {ENDPOINT}\n{body}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"Request failed: {e.reason}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(games, indent=2))
        return 0

    print(f"Endpoint : {ENDPOINT}")
    print(f"Auth     : {'X-API-Key set' if api_key else 'anonymous'}")
    print(f"Games    : {len(games)}\n")
    width = max((len(g.get("game_id", "")) for g in games), default=8)
    for g in sorted(games, key=lambda g: g.get("title", "")):
        print(f"  {g.get('game_id', '?'):<{width}}  {g.get('title', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
