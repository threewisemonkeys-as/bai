"""The planner's LLM transport: SSE assembly, and the deadline it exists to fix.

`llm_call` streams because a NON-streamed OpenRouter completion sends no bytes until
generation finishes, which turns httpx's per-read timeout into a hard total deadline on
output length. These tests pin the three things that would silently break the eval if the
transport regressed: the hidden reasoning must survive (it is `delta.reasoning` when
streamed, `message.reasoning` when not), keep-alive comment lines must not be parsed as
data, and the local Claude proxy -- which speaks no SSE -- must keep getting a plain POST.
"""
import asyncio
import json

import httpx
import pytest

from offline_learning.scripts import eval_coverage_plan as E


def _sse(*chunks: dict, done: bool = True, keepalives: int = 2) -> bytes:
    body = b": OPENROUTER PROCESSING\n\n" * keepalives
    for c in chunks:
        body += b"data: " + json.dumps(c).encode() + b"\n\n"
    if done:
        body += b"data: [DONE]\n\n"
    return body


def _chunk(content="", reasoning="", **extra) -> dict:
    return {"provider": "Alibaba",
            "choices": [{"index": 0, "delta": {"content": content, "reasoning": reasoning}}],
            **extra}


def _run(llm, handler, attempts=1):
    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient

    def patched(*a, **kw):
        kw["transport"] = transport
        return real(*a, **kw)

    E.httpx.AsyncClient = patched
    try:
        return asyncio.run(E.llm_call("prompt", asyncio.Semaphore(1), llm, attempts=attempts))
    finally:
        E.httpx.AsyncClient = real


OR = E.LLMConfig(backend="openrouter", url="https://openrouter.test/v1/chat/completions",
                 model="deepseek/deepseek-v4-flash", api_key="k")
CLAUDE = E.LLMConfig(backend="claude", url="http://127.0.0.1:8000/v1/chat/completions",
                     model="sonnet")


def test_streamed_call_assembles_content_reasoning_and_cost():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=_sse(
            _chunk(content="<plan>", reasoning="thinking "),
            _chunk(content="up\n", reasoning="harder"),
            _chunk(content="</plan>"),
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
             "usage": {"cost": 0.0123, "completion_tokens": 40, "prompt_tokens": 900,
                       "completion_tokens_details": {"reasoning_tokens": 30}}},
        ), headers={"content-type": "text/event-stream"})

    E.CALL_STATS.clear()
    text, reasoning, cost, errors = _run(OR, handler)

    assert seen["body"]["stream"] is True          # the whole point
    assert text == "<plan>up\n</plan>"
    assert reasoning == "thinking harder"          # delta.reasoning, not message.reasoning
    assert cost == pytest.approx(0.0123)
    assert errors == []
    stat = E.CALL_STATS[-1]
    assert stat["streamed"] and stat["finish_reason"] == "stop"
    assert stat["provider"] == "Alibaba" and stat["reasoning_tokens"] == 30
    assert stat["ttfb_s"] is not None and stat["max_gap_s"] is not None


def test_keepalive_comments_are_not_data():
    """`: OPENROUTER PROCESSING` arrives during prefill -- it resets the stall clock and
    must never reach the JSON parser."""
    def handler(request):
        return httpx.Response(200, content=_sse(_chunk(content="ok"), keepalives=50),
                              headers={"content-type": "text/event-stream"})

    text, _r, _c, errors = _run(OR, handler)
    assert text == "ok" and errors == []


def test_mid_stream_error_chunk_is_raised_not_returned_as_text():
    def handler(request):
        return httpx.Response(200, content=_sse(
            _chunk(content="partial"), {"error": {"message": "upstream exploded"}},
        ), headers={"content-type": "text/event-stream"})

    text, _r, _c, errors = _run(OR, handler, attempts=2)
    assert text == ""
    assert len(errors) == 2 and "upstream exploded" in errors[0]


def test_http_error_reports_the_body():
    def handler(request):
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    text, _r, _c, errors = _run(OR, handler, attempts=1)
    assert text == "" and "429" in errors[0] and "rate limited" in errors[0]


def test_claude_proxy_is_not_streamed():
    """The local CLI proxy answers with one JSON body and speaks no SSE."""
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "<plan>noop</plan>",
                                     "reasoning": "brief"}, "finish_reason": "stop"}],
            "usage": {"cost": 0.5}})

    text, reasoning, cost, errors = _run(CLAUDE, handler)
    assert "stream" not in seen["body"]
    assert text == "<plan>noop</plan>" and reasoning == "brief"
    assert cost == pytest.approx(0.5) and errors == []
    assert E.CALL_STATS[-1]["streamed"] is False


def test_stream_can_be_disabled_by_env(monkeypatch):
    monkeypatch.setenv("LLM_STREAM", "0")
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    text, _r, _c, _e = _run(OR, handler)
    assert "stream" not in seen["body"] and text == "x"


def test_empty_completion_is_retried_then_reported():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, content=_sse(
            {"choices": [{"index": 0, "delta": {"content": ""},
                          "finish_reason": "length"}]}),
            headers={"content-type": "text/event-stream"})

    text, _r, cost, errors = _run(OR, handler, attempts=3)
    assert text == "" and cost == 0.0
    assert len(calls) == 3
    assert all("finish=length" in e for e in errors)
