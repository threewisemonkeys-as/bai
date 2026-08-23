#!/usr/bin/env python3
"""Expose a logged-in Claude CLI as an isolated OpenAI-compatible chat endpoint.

Each request runs with a fresh empty HOME, Claude config directory, project directory, and
working directory. Only the subscription credential is copied into that temporary config.
Claude Code filesystem settings, CLAUDE.md files, memory, tools, hooks, plugins, skills, MCP
servers, and session persistence are disabled.

    uv run python offline_learning/scripts/claude_cli_proxy.py
    curl -s http://127.0.0.1:8000/healthz
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

DEFAULT_SYSTEM_PROMPT = "You are Claude, an AI assistant."
MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
ANSI_RE = re.compile(r"(?:\x1B[@-_][0-?]*[ -/]*[@-~]|\r)")
CLI_SETTINGS = json.dumps({"disableAllHooks": True}, separators=(",", ":"))
AUTH_ENV_KEYS = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_OAUTH_TOKEN",
}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "sonnet"
    messages: list[ChatMessage] | None = None
    prompt: str | None = None


def render_prompt(request: ChatRequest) -> str:
    """Accept standard chat messages while retaining the supplied prompt-only shorthand."""
    if request.messages:
        if len(request.messages) == 1 and request.messages[0].role == "user":
            prompt = request.messages[0].content
        else:
            prompt = "\n\n".join(
                f"<{message.role}>\n{message.content}\n</{message.role}>"
                for message in request.messages
            )
    else:
        prompt = request.prompt or ""
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Provide a non-empty messages list or prompt")
    return prompt


def scrub_terminal_output(raw_text: str) -> str:
    """Remove terminal control sequences that should not appear in an API response."""
    if not raw_text:
        return ""
    clean_text = ANSI_RE.sub("", raw_text)
    for phrase in (
        "Analyzing project structure...",
        "Checking workspace layout...",
        "Claude Code loaded.",
        "✔ Done",
        "Working...",
    ):
        clean_text = clean_text.replace(phrase, "")
    return clean_text.strip()


def claude_command(cli_path: str, model: str, system_prompt: str) -> list[str]:
    if not MODEL_RE.fullmatch(model):
        raise HTTPException(status_code=400, detail=f"Invalid Claude model name: {model!r}")
    return [
        cli_path,
        "--print",
        "--model",
        model,
        "--output-format",
        "json",
        "--no-session-persistence",
        "--safe-mode",
        "--setting-sources",
        "",
        "--settings",
        CLI_SETTINGS,
        "--tools",
        "",
        "--disable-slash-commands",
        "--permission-mode",
        "dontAsk",
        "--system-prompt",
        system_prompt,
    ]


def _default_credentials_file() -> Path:
    override = os.getenv("CLAUDE_PROXY_CREDENTIALS_FILE")
    if override:
        return Path(override).expanduser()
    config_dir = os.getenv("CLAUDE_CONFIG_DIR")
    if config_dir:
        return Path(config_dir).expanduser() / ".credentials.json"
    return Path.home() / ".claude" / ".credentials.json"


@contextmanager
def isolated_cli_environment(
    runtime_root: Path,
    credentials_file: Path,
) -> Iterator[tuple[Path, dict[str, str]]]:
    """Create request-local config/project state and remove it after the CLI exits."""
    request_root = Path(tempfile.mkdtemp(prefix="request-", dir=runtime_root))
    request_root.chmod(0o700)
    config_dir = request_root / "config"
    project_dir = request_root / "project"
    home_dir = request_root / "home"
    cache_dir = request_root / "cache"
    data_dir = request_root / "data"
    temp_dir = request_root / "tmp"
    try:
        for directory in (config_dir, project_dir, home_dir, cache_dir, data_dir, temp_dir):
            directory.mkdir(mode=0o700)

        # Authentication is the only user-level Claude file admitted into the request sandbox.
        if credentials_file.is_file():
            destination = config_dir / ".credentials.json"
            shutil.copyfile(credentials_file, destination)
            destination.chmod(0o600)

        env = os.environ.copy()
        for key in tuple(env):
            if (
                key.startswith(("CLAUDE_", "ANTHROPIC_"))
                and key not in AUTH_ENV_KEYS
            ):
                env.pop(key)
        for key in ("PWD", "OLDPWD", "INIT_CWD"):
            env.pop(key, None)
        env.update({
            "HOME": str(home_dir),
            "PWD": str(project_dir),
            "XDG_CONFIG_HOME": str(home_dir / ".config"),
            "XDG_CACHE_HOME": str(cache_dir),
            "XDG_DATA_HOME": str(data_dir),
            "TMPDIR": str(temp_dir),
            "CLAUDE_CONFIG_DIR": str(config_dir),
            "CLAUDE_PROJECT_DIR": str(project_dir),
            "CLAUDE_CODE_TMPDIR": str(temp_dir),
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_SKIP_PROMPT_HISTORY": "1",
            "CLAUDE_CODE_DISABLE_AGENT_VIEW": "1",
            "CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD": "0",
            "DISABLE_AUTOUPDATER": "1",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "DISABLE_BUG_COMMAND": "1",
        })
        yield project_dir, env
    finally:
        shutil.rmtree(request_root, ignore_errors=True)


def openai_response(payload: dict[str, Any], model: str) -> dict[str, Any]:
    if payload.get("is_error"):
        detail = scrub_terminal_output(str(payload.get("result", payload)))
        raise HTTPException(status_code=502, detail=f"Claude CLI error: {detail}")
    content = payload.get("result")
    if not isinstance(content, str) or not content.strip():
        raise HTTPException(status_code=502, detail="Claude CLI returned no result text")

    cli_usage = payload.get("usage") or {}
    prompt_tokens = int(cli_usage.get("input_tokens") or 0)
    prompt_tokens += int(cli_usage.get("cache_creation_input_tokens") or 0)
    prompt_tokens += int(cli_usage.get("cache_read_input_tokens") or 0)
    completion_tokens = int(cli_usage.get("output_tokens") or 0)
    return {
        "id": f"chatcmpl-claude-cli-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": scrub_terminal_output(content),
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost": float(payload.get("total_cost_usd") or 0.0),
        },
    }


def create_app(
    *,
    cli_path: str | None = None,
    credentials_file: str | Path | None = None,
    runtime_parent: str | Path | None = None,
    max_concurrency: int = 4,
    timeout_s: float = 600.0,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> FastAPI:
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")

    resolved_cli = cli_path or shutil.which("claude") or "claude"
    resolved_credentials = (
        Path(credentials_file).expanduser() if credentials_file else _default_credentials_file()
    )
    resolved_runtime_parent = Path(runtime_parent or tempfile.gettempdir())
    semaphore = asyncio.Semaphore(max_concurrency)

    @asynccontextmanager
    async def lifespan(api: FastAPI):
        runtime_root = Path(tempfile.mkdtemp(
            prefix=f"claude-cli-proxy-{os.getuid()}-",
            dir=resolved_runtime_parent,
        ))
        runtime_root.chmod(0o700)
        api.state.runtime_root = runtime_root
        try:
            yield
        finally:
            shutil.rmtree(runtime_root, ignore_errors=True)

    api = FastAPI(title="Isolated Claude CLI subscription proxy", lifespan=lifespan)

    @api.get("/healthz")
    async def healthz() -> dict[str, Any]:
        executable = (
            Path(resolved_cli).is_file()
            if os.path.sep in resolved_cli
            else shutil.which(resolved_cli) is not None
        )
        return {
            "status": "ok" if executable else "error",
            "claude_cli": resolved_cli,
            "claude_cli_found": executable,
            "credential_file_found": resolved_credentials.is_file(),
            "max_concurrency": max_concurrency,
            "timeout_s": timeout_s,
            "isolation": {
                "fresh_config_per_request": True,
                "fresh_project_per_request": True,
                "filesystem_setting_sources": [],
                "tools": False,
                "user_project_hooks": False,
                "session_persistence": False,
            },
        }

    @api.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest) -> dict[str, Any]:
        prompt = render_prompt(request)
        command = claude_command(resolved_cli, request.model, system_prompt)
        try:
            async with semaphore:
                with isolated_cli_environment(
                    api.state.runtime_root,
                    resolved_credentials,
                ) as (project_dir, env):
                    process = await asyncio.create_subprocess_exec(
                        *command,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=project_dir,
                        env=env,
                    )
                    try:
                        stdout, stderr = await asyncio.wait_for(
                            process.communicate(prompt.encode("utf-8")),
                            timeout=timeout_s,
                        )
                    except TimeoutError:
                        process.kill()
                        await process.communicate()
                        raise HTTPException(
                            status_code=504,
                            detail=f"Claude CLI timed out after {timeout_s:g}s",
                        )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Claude CLI executable not found: {resolved_cli}",
            ) from exc

        stderr_text = scrub_terminal_output(stderr.decode("utf-8", errors="replace"))
        stdout_text = scrub_terminal_output(stdout.decode("utf-8", errors="replace"))
        if process.returncode != 0:
            detail = stderr_text or stdout_text or f"exit status {process.returncode}"
            raise HTTPException(status_code=502, detail=f"Claude CLI failed: {detail[-4000:]}")
        try:
            payload = json.loads(stdout_text)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Claude CLI returned invalid JSON: {stdout_text[-1000:]}",
            ) from exc
        return openai_response(payload, request.model)

    return api


app = create_app(
    cli_path=os.getenv("CLAUDE_CLI") or None,
    credentials_file=os.getenv("CLAUDE_PROXY_CREDENTIALS_FILE") or None,
    runtime_parent=os.getenv("CLAUDE_PROXY_RUNTIME_PARENT") or None,
    max_concurrency=int(os.getenv("CLAUDE_PROXY_MAX_CONCURRENCY", "4")),
    timeout_s=float(os.getenv("CLAUDE_PROXY_TIMEOUT_SECONDS", "600")),
    system_prompt=os.getenv("CLAUDE_PROXY_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--claude-cli", default=os.getenv("CLAUDE_CLI") or None)
    parser.add_argument(
        "--credentials-file",
        default=os.getenv("CLAUDE_PROXY_CREDENTIALS_FILE") or None,
        help="subscription credential copied into each request-local config",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=int(os.getenv("CLAUDE_PROXY_MAX_CONCURRENCY", "4")),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("CLAUDE_PROXY_TIMEOUT_SECONDS", "600")),
    )
    args = parser.parse_args()

    import uvicorn

    direct_app = create_app(
        cli_path=args.claude_cli,
        credentials_file=args.credentials_file,
        runtime_parent=os.getenv("CLAUDE_PROXY_RUNTIME_PARENT") or None,
        max_concurrency=args.max_concurrency,
        timeout_s=args.timeout,
        system_prompt=os.getenv("CLAUDE_PROXY_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
    )
    uvicorn.run(direct_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
