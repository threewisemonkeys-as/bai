#!/usr/bin/env python3
"""
End-to-end smoke test: inject invalid actions into a mock stepwise_eb_learn episode
and verify that the Auxiliary Observation section in agent_messages does NOT nest.

Run with:  uv run python tests/test_invalid_action_mock.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# ---- Hydra config loading ----
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parent.parent  # repo root (this file lives in tests/)
GlobalHydra.instance().clear()
config_path = str(_ROOT / "BALROG/balrog/config")
initialize_config_dir(config_dir=config_path, job_name="test_invalid_action")

overrides = [
    "eval.output_dir=logs/test_invalid_action_mock",
    "eval.evolve.mock_mode=true",
    "eval.evolve.n_environment_steps=6",
    "eval.evolve.improve_interval=999",
    "eval.evolve.experiment_interval=999",
    "eval.evolve.artifact_update_interval=999",
    "eval.feedback_on_invalid_action=true",
    "eval.num_workers=1",
    "eval.num_episodes=1",
    "envs.names=minihack",
    f"eval.perception_path={_ROOT}/curated/perc/perc.py",
    "client.model_id=anthropic/claude-haiku-4-5-20251001",
]

config = compose(config_name="config", overrides=overrides)
GlobalHydra.instance().clear()

# ---- Inject invalid actions via mock action provider ----
import balrog.client as _client

_invalid_budget = {"remaining": 3}
_real_provider_ref = [None]


def _invalid_then_valid_provider():
    if _invalid_budget["remaining"] > 0:
        _invalid_budget["remaining"] -= 1
        print(f"  [mock] injecting invalid action (budget left: {_invalid_budget['remaining']})")
        return "DEFINITELY_NOT_A_VALID_ACTION"
    # Fall back to the real provider
    real = _real_provider_ref[0]
    if real is not None:
        return real()
    return "wait"


_original_set_mock_action_provider = _client.set_mock_action_provider


def _patched_set_mock_action_provider(fn):
    _real_provider_ref[0] = fn
    _original_set_mock_action_provider(_invalid_then_valid_provider)


_client.set_mock_action_provider = _patched_set_mock_action_provider

# Patch the same symbol in stepwise_eb_learn's namespace too
import explore.stepwise_eb_learn as _seb
_seb.set_mock_action_provider = _patched_set_mock_action_provider

# ---- Run ----
from explore.stepwise_eb_learn import build_stepwise_eb_config, stepwise_eb_learn

with tempfile.TemporaryDirectory(prefix="test_invalid_action_") as tmpdir:
    eb_config = build_stepwise_eb_config(config)

    # Force output to our temp dir
    output_dir = tmpdir

    print(f"\nRunning stepwise_eb_learn in mock mode with {_invalid_budget['remaining']} invalid actions …")
    print(f"Output dir: {output_dir}\n")

    stepwise_eb_learn(
        eb_config=eb_config,
        config=config,
        original_cwd=str(_ROOT),
        output_dir=output_dir,
    )

    # ---- Check results ----
    print("\n=== Checking agent_messages.json files for nested Auxiliary Observations ===\n")

    agent_msg_files = sorted(Path(output_dir).rglob("agent_messages.json"))
    if not agent_msg_files:
        print("ERROR: No agent_messages.json files found!")
        sys.exit(1)

    failures = []
    for f in agent_msg_files:
        msgs = json.loads(f.read_text())
        for i, msg in enumerate(msgs):
            if msg["role"] != "user":
                continue
            content = msg["content"]
            text = content if isinstance(content, str) else str(content)
            # Count nesting depth
            depth = text.count("Start of Auxilliary Observation")
            if depth > 1:
                failures.append((f, i, depth))
                print(f"  FAIL  {f.relative_to(output_dir)}  msg[{i}]: Auxiliary nested {depth} times")
            elif depth == 1:
                print(f"  OK    {f.relative_to(output_dir)}  msg[{i}]: Auxiliary present (depth=1) ✓")

    print()
    if failures:
        print(f"FAILED: {len(failures)} message(s) had nested Auxiliary Observations.")
        sys.exit(1)
    else:
        print("PASSED: No nested Auxiliary Observations found.")
