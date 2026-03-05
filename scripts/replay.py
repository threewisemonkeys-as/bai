#!/usr/bin/env python3
"""
Replay trajectory JSON logs in the terminal.

Reuses the same two-pane display from play.py, stepping through saved
observations with interactive controls.

Usage:
    python scripts/replay.py trajectories/MiniHack-Quest-Easy-v0/<file>.json
"""

import argparse
import json
import os
import sys
import time

# Import display helpers from play.py (no env dependencies needed)
sys.path.insert(0, os.path.dirname(__file__))
from play import format_display, parse_sections  # noqa: E402


def obs_from_log(logged_obs):
    """Reconstruct the obs dict format that format_display expects."""
    return {
        "text": {
            "long_term_context": logged_obs.get("long_term_context", ""),
            "short_term_context": logged_obs.get("short_term_context", ""),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Replay a trajectory JSON log")
    parser.add_argument("trajectory", help="Path to trajectory JSON file")
    parser.add_argument("--speed", type=float, default=None,
                        help="Auto-play speed in seconds per step")
    args = parser.parse_args()

    with open(args.trajectory) as f:
        traj = json.load(f)

    task = traj.get("task", "unknown")
    goal = traj.get("goal")
    actions = traj.get("available_actions", {})
    steps = traj.get("steps", [])
    initial_obs = traj.get("initial_observation")

    if initial_obs is None:
        print("Error: trajectory has no initial_observation")
        sys.exit(1)

    # Build list of frames: (obs_dict, step_num, cumulative_reward, action_taken, step_reward)
    frames = []
    # Frame 0: initial observation
    frames.append((obs_from_log(initial_obs), 0, 0.0, None, None))
    # Subsequent frames from steps
    cum_reward = 0.0
    for step in steps:
        reward = step.get("reward", 0)
        cum_reward += reward
        step_num = step.get("step", len(frames))
        obs = obs_from_log(step["observation"])
        frames.append((obs, step_num, cum_reward, step.get("action"), reward))

    total_frames = len(frames)
    idx = 0
    auto_play = args.speed is not None

    def show(i):
        obs, step_num, cum_reward, action, reward = frames[i]
        os.system("clear")
        print(format_display(obs, actions, step_num, cum_reward, goal=goal))
        # Show action/reward info
        info_parts = [f"  [{task}]  Frame {i}/{total_frames - 1}"]
        if action is not None:
            info_parts.append(f"Action: {action}")
        if reward is not None and reward != 0:
            info_parts.append(f"Reward: {reward:+.1f}")
        if i == total_frames - 1:
            done = traj.get("done", False)
            info_parts.append("DONE" if done else "INCOMPLETE")
        print("  ".join(info_parts))
        print("  [Enter]=next  [b]=back  [q]=quit  [g N]=goto  [a]=auto-play")

    show(idx)

    if auto_play:
        # Auto-play mode
        try:
            while idx < total_frames - 1:
                time.sleep(args.speed)
                idx += 1
                show(idx)
        except KeyboardInterrupt:
            pass
        return

    # Interactive mode
    try:
        while True:
            try:
                cmd = input("").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if cmd == "q":
                break
            elif cmd == "b":
                if idx > 0:
                    idx -= 1
                show(idx)
            elif cmd.startswith("g ") or cmd.startswith("g"):
                # Parse "g N" or "gN"
                num_str = cmd[1:].strip()
                try:
                    target = int(num_str)
                    idx = max(0, min(target, total_frames - 1))
                except ValueError:
                    pass
                show(idx)
            elif cmd == "a":
                # Auto-play from current position
                try:
                    while idx < total_frames - 1:
                        time.sleep(0.5)
                        idx += 1
                        show(idx)
                except KeyboardInterrupt:
                    show(idx)
            else:
                # Enter or anything else = next
                if idx < total_frames - 1:
                    idx += 1
                show(idx)
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
