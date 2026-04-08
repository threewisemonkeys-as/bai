#!/usr/bin/env python3
"""Step-by-step interactive player controlled via files."""
import json
import os
import sys
import time
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "BALROG"))

import gym
import minihack  # noqa: F401

from balrog.environments.nle import NLELanguageWrapper
from balrog.environments.minihack import get_task_goal, get_available_actions
from balrog.environments.wrappers import NLETimeLimit

OBSERVATION_KEYS = [
    "glyphs", "blstats", "tty_chars", "inv_letters", "inv_strs",
    "tty_cursor", "tty_colors", "screen_descriptions",
]

STATE_FILE = "/tmp/bai_state.json"
ACTION_FILE = "/tmp/bai_action.txt"

def parse_sections(text):
    sections = {}
    parts = re.split(r'^([a-zA-Z][a-zA-Z ]*):[ ]*\n', text, flags=re.MULTILINE)
    for i in range(1, len(parts) - 1, 2):
        key = parts[i].strip().lower()
        val = parts[i + 1].rstrip("\n")
        sections[key] = val
    return sections

def obs_to_dict(obs):
    text = obs["text"]
    lt = text.get("long_term_context", "")
    st = text.get("short_term_context", "")
    lt_s = parse_sections(lt)
    st_s = parse_sections(st)
    return {
        "message": lt_s.get("message", "").strip(),
        "cursor": lt_s.get("cursor", "").strip(),
        "map": lt_s.get("map", "").rstrip(),
        "statistics": st_s.get("statistics", "").strip(),
        "inventory": st_s.get("inventory", "").strip(),
    }

def write_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def main():
    # Clean up
    for f in [STATE_FILE, ACTION_FILE]:
        if os.path.exists(f):
            os.remove(f)

    env = gym.make("MiniHack-Quest-Easy-v0", observation_keys=OBSERVATION_KEYS)
    env = NLELanguageWrapper(env, vlm=False, include_lang_obs=False, include_perc_obs=False)
    env = NLETimeLimit(env)

    lang_env = env
    while lang_env is not None:
        if isinstance(lang_env, NLELanguageWrapper):
            break
        lang_env = getattr(lang_env, "env", None)

    actions = get_available_actions(lang_env)
    goal = get_task_goal("MiniHack-Quest-Easy-v0")

    obs = env.reset()
    step_num = 0
    total_reward = 0.0
    done = False

    state = {
        "step": step_num,
        "reward": 0,
        "total_reward": total_reward,
        "done": done,
        "goal": goal,
        "observation": obs_to_dict(obs),
        "status": "waiting_for_action",
    }
    write_state(state)
    print(f"Environment ready. State written to {STATE_FILE}", flush=True)

    while not done:
        # Wait for action file
        while not os.path.exists(ACTION_FILE):
            time.sleep(0.2)

        time.sleep(0.1)  # ensure file is fully written
        with open(ACTION_FILE, "r") as f:
            action = f.read().strip()
        os.remove(ACTION_FILE)

        if action.lower() == "quit":
            print("Quitting.", flush=True)
            break

        try:
            obs, reward, done, info = env.step(action)
            step_num += 1
            total_reward += reward
            state = {
                "step": step_num,
                "action_taken": action,
                "reward": reward,
                "total_reward": total_reward,
                "done": done,
                "goal": goal,
                "observation": obs_to_dict(obs),
                "status": "done" if done else "waiting_for_action",
            }
        except Exception as e:
            state = {
                "step": step_num,
                "action_taken": action,
                "error": str(e),
                "reward": 0,
                "total_reward": total_reward,
                "done": False,
                "goal": goal,
                "observation": obs_to_dict(obs),
                "status": "error",
            }

        write_state(state)
        print(f"Step {step_num}: action='{action}' reward={reward} done={done}", flush=True)

    env.close()
    print(f"Game over. Total steps: {step_num}, Total reward: {total_reward}", flush=True)

if __name__ == "__main__":
    main()
