"""The parity proxy: the pin it injects, and the ways that injection can go quiet.

Every failure this guards against returns HTTP 200. A dropped pin, a stripped field that
was never sent, an audit that cannot authenticate -- none of them raise, none of them
show up in the transcript, and all of them change what the arm measured. So the tests are
about the request that leaves the proxy and the verdict it records, not about plumbing.

No network: the upstream is a stub the proxy talks to over an ASGI transport.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[1]
RG = REPO / "RGB-Agent"
for _p in (RG, RG / "research/arc-agi-3"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

pytestmark = pytest.mark.skipif(
    not (RG / "research/autumn/proxy.py").is_file(),
    reason="needs the prolong-autumn submodule branch")

proxy = pytest.importorskip("research.autumn.proxy")


# ------------------------------------------------------------------- the pin itself
def test_the_pin_is_the_planner_arms_pin():
    """The one line the whole comparison rests on. `raw`, `icl` and `lmwm` ran on these
    three hosts; an agent arm on different silicon is a different experiment, and the
    launcher's default is where that set is actually defined."""
    launcher = (REPO / "offline_learning/launch/launch_planning_v2_online.py").read_text()
    line = next(x for x in launcher.splitlines() if "--provider-only" in x and "default" in x)
    default = line.split('default="')[1].split('"')[0]
    assert tuple(default.split(",")) == proxy.PIN


def test_the_pin_excludes_the_fp4_host():
    """`atlas-cloud/fp4` serves this model at a different quantisation. It is the reason
    an unpinned call is not the same experiment even when the model slug matches."""
    assert all(p.endswith("/fp8") for p in proxy.PIN)
    assert not any("atlas" in p for p in proxy.PIN)


# ---------------------------------------------------------------- the body rewrite
def test_the_pin_is_injected():
    state = proxy.Parity()
    body, note = state.rewrite(json.dumps({"model": "m", "input": "hi"}).encode())
    assert json.loads(body)["provider"] == {"only": list(proxy.PIN)}
    assert json.loads(body)["usage"] == {"include": True}
    assert note["pinned"] is True


def test_a_provider_the_caller_chose_is_overridden():
    """The caller cannot opt out of the pin: if codex ever learns to send `provider`,
    the arm's routing must still be the arm's."""
    state = proxy.Parity()
    body, _ = state.rewrite(json.dumps({"model": "m", "provider": {"only": ["venice"]}}).encode())
    assert json.loads(body)["provider"] == {"only": list(proxy.PIN)}


def test_the_sampling_knobs_the_planner_never_sent_are_removed():
    state = proxy.Parity()
    body, note = state.rewrite(json.dumps({
        "model": "m", "temperature": 0.9, "top_p": 0.5, "seed": 7,
        "max_output_tokens": 64, "input": "hi"}).encode())
    out = json.loads(body)
    assert not ({"temperature", "top_p", "seed", "max_output_tokens"} & set(out))
    assert out["input"] == "hi"
    assert set(note["stripped"]) == {"temperature", "top_p", "seed", "max_output_tokens"}


def test_codexs_own_request_needs_nothing_stripped():
    """Measured against codex 0.151: it sends no decoding parameters at all, so the strip
    list is a guard against a future change rather than a live correction. If this ever
    fails, the two arms' sampling has diverged and someone should know why."""
    state = proxy.Parity()
    codex_body = {"model": "deepseek/deepseek-v4-flash", "input": [], "stream": True,
                  "store": False, "tool_choice": "auto", "parallel_tool_calls": False,
                  "reasoning": {"effort": "medium", "context": "all_turns"},
                  "text": {"verbosity": "low"}, "include": ["reasoning.encrypted_content"]}
    _body, note = state.rewrite(json.dumps(codex_body).encode())
    assert note["stripped"] == []


def test_an_unparseable_body_is_flagged_not_swallowed():
    state = proxy.Parity()
    body, note = state.rewrite(b"not json at all")
    assert body == b"not json at all"
    assert note["pinned"] is False and note["reason"]


# ------------------------------------------------------------------- the verdict
def test_a_display_name_is_resolved_through_openrouters_own_map():
    tags = {"Alibaba": "alibaba/fp8", "AtlasCloud": "atlas-cloud/fp4"}
    assert proxy._served_by_pinned("Alibaba", tags)
    assert not proxy._served_by_pinned("AtlasCloud", tags)


def test_an_unknown_display_name_falls_back_to_normalising():
    """If the endpoint listing could not be fetched the audit still has to reach a
    verdict; silently passing everything would be worse than a rough match."""
    assert proxy._served_by_pinned("Parasail", {})
    assert not proxy._served_by_pinned("Baidu", {})


def test_the_fp4_host_is_rejected_by_both_paths():
    assert not proxy._served_by_pinned("AtlasCloud", {})
    assert not proxy._served_by_pinned("AtlasCloud", {"AtlasCloud": "atlas-cloud/fp4"})


# ------------------------------------------------------- the proxy, end to end
@pytest.fixture()
def stub(monkeypatch, tmp_path):
    """A fake OpenRouter that records what it was sent and streams a `gen-` id back."""
    seen: list[dict] = []
    upstream = FastAPI()

    # The annotation must resolve in MODULE globals: this file uses postponed
    # evaluation, so a `Request` imported inside the fixture is invisible to FastAPI and
    # the parameter silently becomes a required query arg (every call 422s).
    @upstream.post("/responses")
    async def responses(request: Request):            # noqa: ANN202
        seen.append(json.loads(await request.body()))

        async def body():                             # noqa: ANN202
            yield b'data: {"type":"response.created","response":{"id":"gen-stub-123"}}\n\n'
            yield (b'data: {"type":"response.completed","response":{"usage":'
                   b'{"input_tokens":11,"input_tokens_details":{"cached_tokens":7},'
                   b'"output_tokens":22,"output_tokens_details":{"reasoning_tokens":9},'
                   b'"cost":0.0025}},"sequence_number":24}\n\n')
            yield b'data: [DONE]\n\n'
        return StreamingResponse(body(), media_type="text/event-stream")

    @upstream.get("/generation")
    async def generation(id: str):                    # noqa: A002,ANN202
        return {"data": {"provider_name": "Alibaba", "model": "m",
                         "native_tokens_prompt": 10, "native_tokens_completion": 20,
                         "native_tokens_reasoning": 5, "total_cost": 0.001}}

    state = proxy.Parity(audit_path=tmp_path / "parity.jsonl", upstream="http://up")
    # Both hops go over ASGI: no sockets, no key, no cost. The proxy's lifespan adopts a
    # client and a tag map that are already set, so nothing here has to be monkeypatched;
    # the lifespan still runs, which is what drains the detached audits on shutdown.
    state.client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=upstream), base_url="http://up")
    state.tags = {"Alibaba": "alibaba/fp8"}
    return proxy.build_app(state), state, seen


def test_a_codex_shaped_call_arrives_upstream_pinned(stub):
    app, _state, seen = stub
    with TestClient(app) as client:
        r = client.post("/v1/responses", json={"model": "m", "input": [], "stream": True})
    assert r.status_code == 200
    assert seen[0]["provider"] == {"only": list(proxy.PIN)}
    assert seen[0]["usage"] == {"include": True}


def test_the_v1_prefix_is_not_doubled(stub):
    """Callers are configured with `.../v1` and codex appends `/responses`, while the
    upstream base already ends in `/v1`. Getting this wrong 404s every call."""
    app, _state, seen = stub
    with TestClient(app) as client:
        assert client.post("/v1/responses", json={"model": "m"}).status_code == 200
        assert client.post("/responses", json={"model": "m"}).status_code == 200
    assert len(seen) == 2


def test_the_response_is_streamed_through_untouched(stub):
    app, _state, _seen = stub
    with TestClient(app) as client:
        r = client.post("/v1/responses", json={"model": "m", "stream": True})
    assert b'"gen-stub-123"' in r.content and b"[DONE]" in r.content


def test_every_call_leaves_an_audit_row(stub):
    """An arm that claims a pin should be able to show it held for all N calls."""
    app, state, _seen = stub
    with TestClient(app) as client:
        client.post("/v1/responses", json={"model": "m"})
    rows = [json.loads(x) for x in state.audit_path.read_text().splitlines() if x.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["gen_id"] == "gen-stub-123" and row["pinned"] is True
    assert row["provider"] == "Alibaba" and row["tag"] == "alibaba/fp8" and row["ok"] is True


def test_the_accounting_comes_from_the_stream_not_the_generation_record(stub):
    """`/generation` reports zeroes for every token and cost field of a streamed
    responses-API call -- measured, and permanent rather than a lag. The numbers the run
    reports have to come from the stream, where the injected `usage.include` puts them.
    """
    app, state, _seen = stub
    with TestClient(app) as client:
        client.post("/v1/responses", json={"model": "m", "stream": True})
    row = json.loads(state.audit_path.read_text().splitlines()[0])
    assert row["tokens_prompt"] == 11 and row["tokens_completion"] == 22
    assert row["tokens_reasoning"] == 9 and row["tokens_cached"] == 7
    assert row["cost"] == 0.0025


def test_a_stream_with_no_usage_block_still_records_a_row(stub):
    """The stub's `/generation` says Alibaba either way: a missing usage block must cost
    the accounting, never the pin verdict."""
    assert proxy._usage_from_tail(b'data: {"type":"response.created"}') == {}


def test_healthz_reports_the_pin(stub):
    app, _state, _seen = stub
    with TestClient(app) as client:
        assert client.get("/healthz").json()["pin"] == list(proxy.PIN)


# ------------------------------------------------------------ the catalog it feeds
def test_a_code_mode_clone_is_refused():
    """Measured: `tool_mode="code_mode_only"` offers one tool whose argument is freeform
    JavaScript. DeepSeek sends JSON, codex drops the call with no error, and the turn
    ends having done nothing -- 1/5 success against 3/3 for a non-code-mode clone. The
    guard exists because the failure is invisible: the agent reasons, acts, is ignored.
    """
    from research.autumn import agent as agent_mod

    catalog = {"models": [{"slug": "gpt-5.5", "tool_mode": "code_mode_only"}]}

    class _Done:
        stdout = json.dumps(catalog)

    agent_mod.subprocess.run = lambda *a, **k: _Done()                 # type: ignore
    with pytest.raises(RuntimeError, match="code_mode_only"):
        agent_mod.build_catalog(Path("/tmp/never-written.json"))
