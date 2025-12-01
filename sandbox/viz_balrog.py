#!/usr/bin/env python3
"""
Visualize Balrog trajectories from CSV files.

This script reads CSV files containing action-observation pairs and displays
them sequentially with a 1-second delay between each step.

Usage:
    python viz_balrog.py /path/to/directory --delay=1.0
    python viz_balrog.py /path/to/file.csv --delay=0.5
    python viz_balrog.py /path/to/file.csv --manual  # Manual navigation mode
"""

import csv
import io
import os
import sys
import termios
import time
import tty
from pathlib import Path

import fire


def clear_screen():
    """Clear the terminal screen."""
    os.system("clear" if os.name != "nt" else "cls")


def get_char():
    """
    Get a single character from stdin without waiting for Enter.
    Works on Unix-like systems.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def extract_map_from_observation(observation):
    """
    Extract the map section from the observation string.

    Args:
        observation: The full observation string

    Returns:
        The extracted map string
    """
    if not observation or observation.strip() == "":
        return "No map available"

    # The map section comes after "map:" and before the status line
    lines = observation.split("\n")

    # Find where "map:" appears
    map_start_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "map:":
            map_start_idx = i + 1
            break

    if map_start_idx is None:
        return "No map found in observation"

    # Extract all lines from map start until we hit the status line
    # (status line typically starts with "Agent" or character name)
    map_lines = []
    for i in range(map_start_idx, len(lines)):
        line = lines[i]
        # Stop when we hit the status line (usually starts with character name)
        if line.startswith("Agent ") or (
            line.strip()
            and not any(c in line for c in ["|", "-", ".", "@", "<", ">", "#", " "])
        ):
            break
        map_lines.append(line)

    return "\n".join(map_lines)


def display_step(csv_path, row, current_step, total_steps, manual_mode=False):
    """
    Display a single step of the trajectory.

    Args:
        csv_path: Path to the CSV file
        row: CSV row containing step data
        current_step: Current step index (0-based)
        total_steps: Total number of steps
        manual_mode: Whether in manual navigation mode
    """
    clear_screen()

    step = row.get("Step", "N/A")
    action = row.get("Action", "N/A")
    reward = row.get("Reward", "N/A")
    done = row.get("Done", "N/A")
    observation = row.get("Observation", "")

    # Extract and display the map
    map_display = extract_map_from_observation(observation)

    # Display information
    print(f"{'=' * 80}")
    print(f"File: {csv_path.name}")
    print(f"Step: {step} ({current_step + 1}/{total_steps})")
    print(f"{'=' * 80}\n")

    print(f"Action: {action}")
    print(f"Reward: {reward} | Done: {done}\n")

    print("Map:")
    print(map_display)

    print(f"\n{'=' * 80}")

    if manual_mode:
        print("Controls: [h] Previous | [l] Next | [q] Quit")


def visualize_trajectory(csv_file, delay=1.0, manual=False):
    """
    Visualize a single trajectory from a CSV file.

    Args:
        csv_file: Path to the CSV file
        delay: Delay in seconds between steps (ignored in manual mode)
        manual: Enable manual navigation mode with keyboard controls
    """
    csv_path = Path(csv_file)

    print(f"\n{'=' * 80}")
    print(f"Trajectory: {csv_path}")
    print(f"{'=' * 80}\n")

    try:
        # Read file and filter out NUL bytes
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().replace("\x00", "")

        # Parse CSV from cleaned content using StringIO to preserve multiline fields
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)

        if not rows:
            print(f"No data found in {csv_path.name}")
            return

        if manual:
            # Manual navigation mode
            current_idx = 0
            total_steps = len(rows)

            while True:
                display_step(
                    csv_path,
                    rows[current_idx],
                    current_idx,
                    total_steps,
                    manual_mode=True,
                )

                # Get keyboard input
                try:
                    key = get_char()
                except KeyboardInterrupt:
                    print("\n\nVisualization interrupted by user.")
                    raise

                if key == "h":  # Go back
                    if current_idx > 0:
                        current_idx -= 1
                elif key == "l":  # Go forward
                    if current_idx < total_steps - 1:
                        current_idx += 1
                    else:
                        # Reached the end
                        break
                elif key == "q" or key == "\x03":  # q or Ctrl+C
                    raise KeyboardInterrupt
        else:
            # Automatic mode
            total_steps = len(rows)
            for idx, row in enumerate(rows):
                display_step(csv_path, row, idx, total_steps, manual_mode=False)
                # Wait before showing next step
                time.sleep(delay)

    except Exception as e:
        print(f"\n\nError processing {csv_path.name}: {e}")
        print("Skipping to next trajectory...")
        time.sleep(2)
        return

    # Show completion message
    print(f"\n\nTrajectory complete: {csv_path.name}")
    print("Press Enter to continue to next trajectory (or Ctrl+C to exit)...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user.")
        raise


def visualize(path, delay=1.0, manual=False):
    """
    Visualize Balrog trajectories from a CSV file or directory.

    Args:
        path: Path to a CSV file or directory containing CSV files
        delay: Delay in seconds between steps (default: 1.0, ignored in manual mode)
        manual: Enable manual navigation mode with keyboard controls (h=back, l=forward)
    """
    path = Path(path).expanduser()

    # Check if path is a file or directory
    if path.is_file():
        if path.suffix != ".csv":
            print(f"Error: '{path}' is not a CSV file")
            return 1
        csv_files = [path]
    elif path.is_dir():
        # Find all CSV files recursively
        csv_files = sorted(path.rglob("*.csv"))

        if not csv_files:
            print(f"No CSV files found in '{path}'")
            return 1
    else:
        print(f"Error: '{path}' does not exist")
        return 1

    print(f"Found {len(csv_files)} trajectory file(s)")
    if manual:
        print("Manual navigation mode enabled")
        print("Controls: [h] Previous step | [l] Next step | [q] Quit\n")
    else:
        print("Automatic mode enabled")
    print("Starting visualization...\n")
    time.sleep(2)

    try:
        for csv_file in csv_files:
            visualize_trajectory(csv_file, delay, manual)
    except KeyboardInterrupt:
        print("\n\nVisualization stopped by user.")
        return 0

    print("\n\nAll trajectories visualized!")
    return 0


if __name__ == "__main__":
    fire.Fire(visualize)
