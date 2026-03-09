#!/usr/bin/env python3
"""
Interactive terminal player for MiniHack/NetHack environments.

Displays the same observation the agent sees at each step, then prompts
for a text action (e.g. "north", "pickup", "eat").

Usage:
    python scripts/play.py
    python scripts/play.py MiniHack-Room-5x5-v0
    python scripts/play.py NetHackScore-v0 --env-type nle
"""

import argparse
import json
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "BALROG"))

import gym
import minihack  # noqa: F401

from balrog.environments.nle import NLELanguageWrapper, AutoMore
from balrog.environments.minihack import get_task_goal as minihack_get_task_goal, get_available_actions as minihack_get_available_actions
from balrog.environments.nle import ACTIONS as NLE_ACTIONS
from balrog.environments.wrappers import NLETimeLimit


OBSERVATION_KEYS = [
    "glyphs",
    "blstats",
    "tty_chars",
    "inv_letters",
    "inv_strs",
    "tty_cursor",
    "tty_colors",
    "screen_descriptions",
]


def make_play_env(task, env_type="minihack", skip_more=False):
    """Create an environment with the same wrappers the agent uses."""
    env = gym.make(task, observation_keys=OBSERVATION_KEYS)
    if skip_more:
        env = AutoMore(env)
    # include_lang_obs=False to skip the verbose language observation
    env = NLELanguageWrapper(env, vlm=False, include_lang_obs=False, include_perc_obs=False)
    env = NLETimeLimit(env)
    return env


def get_lang_env(env):
    """Walk wrappers to find the NLELanguageWrapper."""
    e = env
    while e is not None:
        if isinstance(e, NLELanguageWrapper):
            return e
        e = getattr(e, "env", None)
    return None


def get_available_actions(env, env_type):
    """Get dict of {action_name: description} for available actions."""
    lang_env = get_lang_env(env)
    if lang_env is None:
        return {}
    if env_type == "minihack":
        return minihack_get_available_actions(lang_env)
    else:
        return dict(NLE_ACTIONS)


def parse_sections(text):
    """Parse 'name:\\ncontent\\n' formatted text into a dict."""
    sections = {}
    # Split on lines that look like a section header (word(s) followed by colon at end of line)
    parts = re.split(r'^([a-zA-Z][a-zA-Z ]*):[ ]*\n', text, flags=re.MULTILINE)
    # parts[0] is text before first header (usually empty), then alternating header/content
    for i in range(1, len(parts) - 1, 2):
        key = parts[i].strip().lower()
        val = parts[i + 1].rstrip("\n")
        sections[key] = val
    return sections


ANSI_RE = re.compile(r'\033\[[0-9;]*m')


def visible_len(s):
    """Length of string excluding ANSI escape codes."""
    return len(ANSI_RE.sub('', s))


def visible_ljust(s, width):
    """Left-justify string to width based on visible (non-ANSI) length."""
    pad = width - visible_len(s)
    if pad > 0:
        return s + " " * pad
    return s


def visible_truncate(s, width):
    """Truncate string to width visible characters, preserving ANSI codes."""
    vis = 0
    i = 0
    while i < len(s) and vis < width:
        m = ANSI_RE.match(s, i)
        if m:
            i = m.end()
        else:
            vis += 1
            i += 1
    # Include any trailing ANSI codes (e.g. reset)
    while i < len(s):
        m = ANSI_RE.match(s, i)
        if m:
            i = m.end()
        else:
            break
    return s[:i]


def side_by_side(left_lines, right_lines, left_width, gap=1):
    """Merge two columns of lines side by side with a vertical separator."""
    height = max(len(left_lines), len(right_lines))
    pad = " " * gap
    out = []
    for i in range(height):
        l = left_lines[i] if i < len(left_lines) else ""
        r = right_lines[i] if i < len(right_lines) else ""
        l = visible_ljust(visible_truncate(l, left_width), left_width)
        out.append(l + pad + "\u2502" + pad + r)
    return "\n".join(out)


def format_actions_block(actions_dict, width):
    """Format actions with descriptions, one per line."""
    lines = ["\033[1mActions:\033[0m"]
    name_width = max(len(a) for a in actions_dict) + 1 if actions_dict else 10
    for name, desc in actions_dict.items():
        line = f"  {name.ljust(name_width)} {desc}"
        lines.append(line)
    return lines


def format_display(obs, actions, step_num, total_reward, goal=None):
    """Format observation into two-pane layout."""
    text = obs["text"]
    long_term = text.get("long_term_context", "")
    short_term = text.get("short_term_context", "")

    lt_sections = parse_sections(long_term)
    st_sections = parse_sections(short_term)

    term_width = shutil.get_terminal_size((120, 40)).columns
    left_width = min(55, term_width // 2)
    gap = 1  # on each side of the separator

    # === Left pane: goal + message + stats + inventory + actions ===
    left = []

    # Goal (wrapped to fit left pane)
    if goal:
        import textwrap
        prefix = "Goal: "
        wrap_width = max(20, left_width - len(prefix))
        wrapped = textwrap.wrap(goal, width=wrap_width)
        if wrapped:
            left.append(f"\033[1mGoal:\033[0m {wrapped[0]}")
            for line in wrapped[1:]:
                left.append(" " * len(prefix) + line)
        left.append("")

    # Message at top
    message = lt_sections.get("message", "").strip()
    if message:
        left.append(f"\033[1mMessage:\033[0m {message}")
        left.append("")

    # Cursor
    cursor = lt_sections.get("cursor", "").strip()
    if cursor:
        for line in cursor.splitlines():
            left.append(line.strip())
        left.append("")

    # Statistics
    stats = st_sections.get("statistics", "").strip()
    if stats:
        left.append("\033[1mStats:\033[0m")
        for line in stats.splitlines():
            line = line.strip()
            if line:
                left.append(line)
        left.append("")

    # Inventory
    inv = st_sections.get("inventory", "").strip()
    if inv:
        left.append("\033[1mInventory:\033[0m")
        for line in inv.splitlines():
            line = line.strip()
            if line:
                left.append(line)
        left.append("")

    # Available actions
    left.extend(format_actions_block(actions, left_width))

    # === Right pane: map ===
    map_text = lt_sections.get("map", "").rstrip()
    right = []
    right.append(f"\033[1mMap:\033[0m  (step {step_num}, reward {total_reward:+.1f})")
    right.append("")
    for line in map_text.splitlines():
        right.append(line)

    # Combine
    header = "=" * term_width
    output = [header]
    output.append(side_by_side(left, right, left_width, gap))
    output.append(header)
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description="Play MiniHack/NetHack with the agent's observation format")
    parser.add_argument("task", nargs="?", default="MiniHack-Quest-Easy-v0",
                        help="Environment ID (default: MiniHack-Quest-Easy-v0)")
    parser.add_argument("--env-type", choices=["minihack", "nle"], default=None,
                        help="Environment type (auto-detected from task name)")
    parser.add_argument("--skip-more", action="store_true",
                        help="Automatically handle --More-- messages")
    parser.add_argument("--log-dir", default="trajectories",
                        help="Directory to save trajectory logs (default: trajectories/)")
    parser.add_argument("--no-log", action="store_true",
                        help="Disable trajectory logging")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (random if not set)")
    args = parser.parse_args()

    # Auto-detect env type
    if args.env_type is None:
        args.env_type = "minihack" if args.task.startswith("MiniHack") else "nle"

    env = make_play_env(args.task, env_type=args.env_type, skip_more=args.skip_more)
    lang_env = get_lang_env(env)
    actions = get_available_actions(env, args.env_type)

    # Trajectory logging setup
    logging_enabled = not args.no_log
    log_dir = None
    if logging_enabled:
        log_dir = Path(args.log_dir) / args.task
        log_dir.mkdir(parents=True, exist_ok=True)

    def obs_to_log(obs):
        """Extract complete observation state for logging/visualisation."""
        text = obs["text"]
        long_term = text.get("long_term_context", "")
        short_term = text.get("short_term_context", "")
        lt_sections = parse_sections(long_term)
        st_sections = parse_sections(short_term)
        return {
            "long_term_context": long_term,
            "short_term_context": short_term,
            # Parsed sections for easier visualisation
            "message": lt_sections.get("message", "").strip(),
            "cursor": lt_sections.get("cursor", "").strip(),
            "map": lt_sections.get("map", "").rstrip(),
            "statistics": st_sections.get("statistics", "").strip(),
            "inventory": st_sections.get("inventory", "").strip(),
        }

    # Seed handling (mirrors evaluator.py)
    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    random.seed(seed)

    # Get the goal the agent sees
    if args.env_type == "minihack":
        goal = minihack_get_task_goal(args.task)
    else:
        goal = "Your goal is to get as far as possible."

    # Track failed actions across the episode
    failed_actions = []
    action_frequency = defaultdict(int)

    def new_trajectory():
        return {
            "task": args.task,
            "env_type": args.env_type,
            "seed": seed,
            "goal": goal,
            "available_actions": {k: v for k, v in actions.items()},
            "start_time": datetime.now().isoformat(),
            "steps": [],
        }

    trajectory_saved = False  # Guard against double-saving

    def save_trajectory(traj, done):
        nonlocal trajectory_saved
        if not logging_enabled:
            return
        traj["end_time"] = datetime.now().isoformat()
        traj["num_steps"] = len(traj["steps"])
        traj["total_reward"] = sum(s.get("reward", 0) for s in traj["steps"])
        traj["done"] = done
        traj["action_frequency"] = dict(action_frequency)
        traj["failed_actions"] = failed_actions.copy()
        # Environment stats (progression, score, depth, etc.)
        if lang_env is not None:
            try:
                traj["env_stats"] = lang_env.get_stats()
            except Exception:
                traj["env_stats"] = {}
        timestamp = traj["start_time"].replace(":", "-")
        path = log_dir / f"{timestamp}.json"
        with open(path, "w") as f:
            json.dump(traj, f, indent=2, default=str)
        trajectory_saved = True
        return path

    def clear_and_show(obs, actions, step_num, total_reward, status=None):
        os.system("clear")
        print(format_display(obs, actions, step_num, total_reward, goal=goal))
        if status:
            print(status)

    print(f"\nPlaying: {args.task}  (seed: {seed})")
    if logging_enabled:
        print(f"Logging trajectories to: {log_dir}/")
    print("Type an action and press Enter. 'quit' to exit.\n")

    # Reset with seed
    env.seed(seed)
    obs = env.reset()
    step_num = 0
    total_reward = 0.0
    trajectory = new_trajectory()
    trajectory["initial_observation"] = obs_to_log(obs)

    clear_and_show(obs, actions, step_num, total_reward)

    try:
        while True:
            try:
                action = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nQuitting.")
                break

            if not action:
                continue
            if action.lower() == "quit":
                break

            # Pass any input through — needed for multi-step actions (zap -> letter, etc.)
            try:
                obs, reward, done, info = env.step(action)
            except Exception as e:
                failed_actions.append({"step": step_num, "action": action, "error": str(e)})
                clear_and_show(obs, actions, step_num, total_reward,
                               f"  Invalid action '{action}': {e}")
                continue
            step_num += 1
            total_reward += reward
            action_frequency[action] += 1

            # Log step with complete state
            trajectory["steps"].append({
                "step": step_num,
                "action": action,
                "reward": reward,
                "cumulative_reward": total_reward,
                "done": done,
                "observation": obs_to_log(obs),
            })

            status = None
            if reward != 0:
                status = f"  [Reward: {reward:+.1f}  Total: {total_reward:.1f}]"

            clear_and_show(obs, actions, step_num, total_reward, status)

            if done:
                path = save_trajectory(trajectory, done=True)
                print(f"\n  EPISODE ENDED  |  Steps: {step_num}  |  Total reward: {total_reward:.1f}")
                if path:
                    print(f"  Trajectory saved: {path}")
                try:
                    again = input("\nPlay again? [y/N] ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    break
                if again == "y":
                    env.seed(seed)
                    obs = env.reset()
                    step_num = 0
                    total_reward = 0.0
                    failed_actions.clear()
                    action_frequency.clear()
                    trajectory_saved = False
                    trajectory = new_trajectory()
                    trajectory["initial_observation"] = obs_to_log(obs)
                    clear_and_show(obs, actions, step_num, total_reward)
                else:
                    break
    finally:
        # Save trajectory on quit (skip if already saved for a completed episode)
        if not trajectory_saved:
            path = save_trajectory(trajectory, done=False)
            if path:
                print(f"  Trajectory saved: {path}")

        env.close()


if __name__ == "__main__":
    main()
