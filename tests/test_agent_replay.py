"""The replay record: capturing a session, and rendering it.

The agent arm's claim is that a coding agent read a corpus of recorded transitions and
worked something out. A row saying `success: true` does not support that claim, so the
run has to keep the working -- and the working is scattered across two sources that each
hold half of it:

  * codex's `--json` stream has the shell commands and the messages, and **no reasoning
    at all** (measured on this build: `show_raw_agent_reasoning` and
    `model_reasoning_summary` change nothing, no `reasoning` item is ever emitted);
  * the parity proxy sees the raw SSE, where reasoning arrives as
    `response.reasoning_text.done` frames -- and sees no tool calls, because those never
    reach the wire as anything but function-call arguments.

Neither is recoverable after the run. These tests hold the join between them, and the
page's own frame-reconstruction, because a replay that silently shows the wrong board or
no thinking looks exactly like a run that did neither.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RG = REPO / "RGB-Agent"
for _p in (RG, RG / "research/arc-agi-3", REPO / "offline_learning/scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

pytestmark = pytest.mark.skipif(
    not (RG / "research/autumn/proxy.py").is_file(),
    reason="needs the prolong-autumn submodule branch")

import httpx                                                        # noqa: E402
from fastapi import FastAPI, Request                                # noqa: E402
from fastapi.responses import StreamingResponse                     # noqa: E402
from fastapi.testclient import TestClient                           # noqa: E402

from research.autumn import proxy                                   # noqa: E402
from research.autumn.agent import _interleave, _text_of             # noqa: E402

REASON = "The drives show click recolours the whole board, not one cell."


# ------------------------------------------------------- the proxy's half: reasoning
@pytest.fixture()
def sse(tmp_path):
    """An upstream that streams reasoning the way OpenRouter actually streams it."""
    upstream = FastAPI()

    @upstream.post("/responses")
    async def responses(request: Request):            # noqa: ANN202
        await request.body()

        async def body():                             # noqa: ANN202
            yield b'data: {"type":"response.created","response":{"id":"gen-stub-abc"}}\n\n'
            # deltas are noise for us; the completed block is the record. It is split
            # mid-frame on purpose: chunks do not respect line boundaries.
            frame = (b'data: {"type":"response.reasoning_text.done","text":'
                     + json.dumps(REASON).encode() + b',"item_id":"rs_1"}\n\n')
            yield frame[:30]
            yield frame[30:]
            yield b'data: {"type":"response.completed","response":{"usage":{}}}\n\n'
        return StreamingResponse(body(), media_type="text/event-stream")

    state = proxy.Parity(audit_path=tmp_path / "parity.jsonl",
                         transcript=tmp_path / "reasoning.jsonl",
                         upstream="http://up")
    state.client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=upstream), base_url="http://up")
    state.tags = {"Alibaba": "alibaba/fp8"}
    return proxy.build_app(state), state


def test_the_proxy_recovers_reasoning_codex_never_emits(sse):
    app, state = sse
    with TestClient(app) as client:
        r = client.post("/v1/responses", json={"model": "m", "input": [], "stream": True})
    assert r.status_code == 200
    rows = [json.loads(x) for x in state.transcript.read_text().splitlines() if x.strip()]
    assert len(rows) == 1
    # the frame was delivered in two pieces and still came back whole
    assert rows[0]["reasoning"] == [REASON]
    assert rows[0]["seq"] == 1


def test_a_run_without_a_transcript_still_streams(tmp_path):
    """The transcript is a view of the run, never a condition of it."""
    upstream = FastAPI()

    @upstream.post("/responses")
    async def responses(request: Request):            # noqa: ANN202
        await request.body()

        async def body():                             # noqa: ANN202
            yield b'data: {"type":"response.created","response":{"id":"gen-x"}}\n\n'
        return StreamingResponse(body(), media_type="text/event-stream")

    state = proxy.Parity(audit_path=tmp_path / "p.jsonl", upstream="http://up")
    state.client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=upstream), base_url="http://up")
    state.tags = {}
    with TestClient(proxy.build_app(state)) as client:
        assert client.post("/v1/responses", json={"model": "m"}).status_code == 200
    assert state.transcript is None


# --------------------------------------------------------------------- the join
def test_reasoning_lands_in_front_of_the_action_it_asked_for():
    events = [{"kind": "command", "command": "ls"},
              {"kind": "command", "command": "cat x"},
              {"kind": "message", "text": "done"}]
    out = _interleave(events, [["why I list"], ["why I read"], ["why I answer"]])
    assert [e["kind"] for e in out] == [
        "reasoning", "command", "reasoning", "command", "message", "reasoning"]
    assert out[0]["text"] == "why I list"
    # the surplus block is the final call, which answered instead of acting
    assert out[-1]["text"] == "why I answer"


def test_a_turn_with_no_reasoning_is_left_exactly_as_it_was():
    """A build that stops emitting reasoning must degrade to today's transcript, not to
    a reordered one."""
    events = [{"kind": "command", "command": "ls"}, {"kind": "message", "text": "hi"}]
    assert _interleave(events, []) == events


def test_more_actions_than_reasoning_blocks_keeps_every_action():
    events = [{"kind": "command", "command": f"c{i}"} for i in range(5)]
    out = _interleave(events, [["only one"]])
    assert sum(1 for e in out if e["kind"] == "command") == 5
    assert sum(1 for e in out if e["kind"] == "reasoning") == 1


@pytest.mark.parametrize("item,expected", [
    ({"text": "a"}, "a"),
    ({"summary": [{"text": "b"}]}, "b"),
    ({"content": [{"type": "reasoning_text", "text": "c"}]}, "c"),
    ({"content": ["d"]}, "d"),
    ({}, ""),
])
def test_reasoning_text_is_found_wherever_the_build_put_it(item, expected):
    """The shape has moved between codex releases, and an empty trace looks exactly like
    a model that did not think -- which is the thing F12 exists to detect."""
    assert _text_of(item) == expected


# ------------------------------------------------------------------- the page
def _trace(uid="g:one:s0", start=None, steps=None):
    start = start or [["black", "white"], ["gold", "black"]]
    return {
        "task_uid": uid, "game": "bt3gb", "label": "BT3GB", "nl_goal": "Make it night.",
        "action_cap": 4, "start_grid": json.dumps(start), "dims": [2, 2],
        "alphabet": ["noop", "click"], "success": True, "reached_at": 2,
        "live_success": True, "failed_reason": None, "actions_used": 2,
        "study_rounds_used": 1, "wall_s": 12.5,
        "usage": {"calls": 2, "in": 100, "out": 20, "reasoning": 9, "cache_read": 80},
        "steps": steps or [
            {"n": 1, "action": "noop", "grid_after": json.dumps(
                [["black", "gold"], ["gold", "black"]]),
             "reached": False, "terminated": False, "remaining": 3, "turn": 0,
             "plan_index": 1, "plan_total": 2},
            {"n": 2, "action": "click 0 0", "grid_after": json.dumps(
                [["gray", "gray"], ["gray", "gray"]]),
             "reached": True, "terminated": False, "remaining": 2, "turn": 0,
             "plan_index": 2, "plan_total": 2},
        ],
        "turns": [{
            "i": 0, "attempt": 0, "kind": "plan", "n": 0, "remaining": 4,
            "prompt": "Plan the next actions.", "plan": ["noop", "click 0 0"],
            "rejected": [], "wall_s": 11.0,
            "tokens": {"input_tokens": 100, "cached_tokens": 80,
                       "output_tokens": 20, "reasoning_tokens": 9},
            "events": [{"kind": "reasoning", "text": REASON},
                       {"kind": "command", "command": "cat drives/t000.json",
                        "exit_code": 0, "output": "{}", "truncated": False},
                       {"kind": "file_change",
                        "changes": [{"path": "actions.json", "type": "add"}], "n": 1}],
        }],
    }


def _rebuild(problem, i):
    """The page's own frameAt(), in Python. If this and the JS ever disagree the page is
    wrong in a way no assertion in this file would catch -- so it is transcribed, not
    reimplemented."""
    if i < 0:
        return problem["startFrame"].replace("|", "")
    cur = None
    for k in range(i + 1):
        f = problem["frames"][k]
        if "full" in f:
            cur = f["full"]
            continue
        if cur is None:
            cur = problem["startFrame"].replace("|", "")
        arr = list(cur)
        for pos, ch in f["d"]:
            arr[pos] = ch
        cur = "".join(arr)
    return cur


def test_the_page_reconstructs_every_recorded_board_exactly():
    import viz_agent_replay as viz
    from viz_plan_replay import Lines

    tr = _trace()
    data = viz.build([tr], Lines())
    p = data["problems"][0]
    names = data["names"]
    for i, step in enumerate(tr["steps"]):
        truth = [c for row in json.loads(step["grid_after"]) for c in row]
        assert [names[ch] for ch in _rebuild(p, i)] == truth, f"board {i} is wrong"
    assert [names[ch] for ch in p["startFrame"].replace("|", "")] == \
        [c for row in json.loads(tr["start_grid"]) for c in row]


def test_every_field_the_page_reads_is_written():
    import viz_agent_replay as viz
    from viz_plan_replay import Lines

    data = viz.build([_trace()], Lines())
    p = data["problems"][0]
    for key in ("uid", "game", "human", "label", "goal", "cap", "rows", "cols",
                "startFrame", "success", "liveSuccess", "reachedAt", "used", "failed",
                "studies", "wall", "usage", "frames", "steps", "turns"):
        assert key in p, key
    turn = p["turns"][0]
    for key in ("i", "kind", "attempt", "n", "remaining", "plan", "rejected", "wall",
                "tok", "prompt", "events", "nthink", "ncmd", "nfile"):
        assert key in turn, key
    for key in ("n", "a", "t", "r", "x", "rem", "pi", "pt"):
        assert key in p["steps"][0], key
    assert [e["k"] for e in turn["events"]] == ["think", "cmd", "file"]
    assert (turn["nthink"], turn["ncmd"], turn["nfile"]) == (1, 1, 1)


def test_the_step_names_the_turn_that_planned_it():
    """The link the whole page turns on: scrubbing the board moves the transcript to the
    thinking that produced the frame."""
    import viz_agent_replay as viz
    from viz_plan_replay import Lines

    p = viz.build([_trace()], Lines())["problems"][0]
    assert {s["t"] for s in p["steps"]} == {0}
    assert all(0 <= s["t"] < len(p["turns"]) for s in p["steps"])


def test_a_run_recorded_before_traces_existed_still_renders():
    import viz_agent_replay as viz
    from viz_plan_replay import Lines

    bare = _trace()
    bare["turns"], bare["steps"] = [], []
    p = viz.build([bare], Lines())["problems"][0]
    assert p["frames"] == [] and p["turns"] == []


def test_the_summary_reports_the_bill_not_only_the_score():
    """Inference compute is the one axis the study does not match across arms, so the
    page that shows the trajectories shows what they cost."""
    import viz_agent_replay as viz

    s = viz.summary([_trace("g:one:s0"), _trace("g:two:s0")])
    assert s["n"] == 2 and s["wins"] == 2
    assert s["calls"] == 4 and s["in"] == 200 and s["reasoning"] == 18
    assert s["games"][0]["human"]
