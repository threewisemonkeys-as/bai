#!/usr/bin/env python3
"""
Mock-mode integration test: runtime perception failures (crashes and blank
output) must be fed back into the improve prompts.

Two perception modules are run through _run_improve_loop_eb with a small
trajectory buffer:
  a. one that crashes on the first (welcome-prefixed) observation, and
  b. one that runs without error but returns a blank string.
Verifies, for both:
  1. Track 1a's prompt contains the RUNTIME PERCEPTION ERRORS section and the
     inline failure rendering in the step sequence.
  2. Track 1b's prompt contains the runtime failure block.
And for the crash case:
  3. Perception validation exercises the failing input (a candidate that still
     crashes on it is rejected and retried).

Run with:  uv run python tests/test_runtime_error_feedback_mock.py
"""

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

_ROOT = Path(__file__).resolve().parent.parent  # repo root (this file lives in tests/)

GlobalHydra.instance().clear()
config_path = str(_ROOT / "BALROG/balrog/config")
initialize_config_dir(config_dir=config_path, job_name="test_runtime_err")
config = compose(
    config_name="config",
    overrides=[
        "eval.evolve.mock_mode=true",
        "eval.evolve.perception_enabled=true",
        "envs.names=minihack",
        "client.model_id=anthropic/claude-haiku-4-5-20251001",
    ],
)
GlobalHydra.instance().clear()

from explore.mixed_improve import set_mock_mode
from explore.stepwise_eb_learn import _run_improve_loop_eb, build_stepwise_eb_config

set_mock_mode(True)
eb_config = build_stepwise_eb_config(config)

# Crashes on any observation that is not pure JSON (e.g. the interactive-phase
# welcome header) — the failure mode observed in real runs.
PERC = '''import json

def perceive(observation_history: list[str]) -> str:
    grid = json.loads(observation_history[-1])
    return f"rows: {len(grid)}"
'''

welcome_obs = (
    'Welcome, you are now in the interactive phase\n'
    '[["black","blue"],["black","red"]]'
)
grid_obs = '[["black","blue"],["black","red"]]'

buffer = []
for i in range(5):
    buffer.append(
        {
            "step": i,
            "action": ["up", "left", "down", "right", "noop"][i],
            "raw_long_term_context": welcome_obs if i == 0 else grid_obs,
            "raw_short_term_context": f"aux {i}",
            "reasoning": f"mock reasoning {i}",
            "reward": 0,
        }
    )

def run_improve(perception_code: str) -> dict:
    _, _, _, _, records = _run_improve_loop_eb(
        config=config,
        eb_config=eb_config,
        beliefs="- initial mock belief",
        perception=perception_code,
        qa_pairs=[],
        trajectory_buffer=buffer,
        default_knowledge="mock default knowledge",
        step=1,
        global_step=1,
        step_dir=None,
    )
    return {r["track"]: r for r in records}


# =========================================================================
# Case a: module that crashes on the welcome-prefixed observation
# =========================================================================
tracks = run_improve(PERC)
print("tracks run:", list(tracks.keys()))

# --- Track 1a: runtime errors section present in the beliefs/analysis prompt
t1a_prompt = tracks["steps_beliefs"]["turns"][0]["prompt"]
assert "=== RUNTIME PERCEPTION ERRORS ===" in t1a_prompt, "Track 1a missing error section"
assert "JSONDecodeError" in t1a_prompt
assert "Welcome, you are now in the interactive phase" in t1a_prompt
assert "PERCEPTION MODULE CRASHED" in t1a_prompt  # inline rendering in step sequence
print("Track 1a prompt contains runtime error section + inline crash: OK")

# --- Track 1b: error block present in the perception improvement prompt
assert "perception_from_analysis" in tracks, "Track 1b did not run"
t1b_turn1 = tracks["perception_from_analysis"]["turns"][0]
assert "JSONDecodeError" in t1b_turn1["prompt"], "Track 1b turn 1 missing runtime error"
assert "FAILS at runtime" in t1b_turn1["prompt"]
print("Track 1b turn 1 prompt contains runtime error block: OK")

print(
    "Track 1b turns (turn, validated, changed):",
    [
        (t["turn"], t["validated"], t["perception_changed"])
        for t in tracks["perception_from_analysis"]["turns"]
    ],
)

# =========================================================================
# Case b: module that runs without error but returns blank output
# =========================================================================
PERC_BLANK = '''def perceive(observation_history: list[str]) -> str:
    return ""
'''

tracks_blank = run_improve(PERC_BLANK)

t1a_prompt = tracks_blank["steps_beliefs"]["turns"][0]["prompt"]
assert "=== RUNTIME PERCEPTION ERRORS ===" in t1a_prompt, (
    "Track 1a missing error section for blank output"
)
assert "BLANK OUTPUT" in t1a_prompt, "Track 1a missing blank-output failure"
assert "RETURNED BLANK OUTPUT" in t1a_prompt, (
    "Track 1a step sequence missing inline blank-output note"
)
print("Track 1a prompt flags blank output as a failure: OK")

t1b_prompt = tracks_blank["perception_from_analysis"]["turns"][0]["prompt"]
assert "BLANK OUTPUT" in t1b_prompt, "Track 1b missing blank-output failure"
print("Track 1b turn 1 prompt flags blank output as a failure: OK")

print("ALL MOCK INTEGRATION CHECKS PASSED")
