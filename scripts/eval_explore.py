#!/usr/bin/env python3
"""Evaluate all steps of an explore run.

Given an explore run directory, evaluates each step's beliefs + perception
in parallel and presents a summary table.

Usage:
    python sandbox/eval_explore.py <explore_run_dir> [options] [-- hydra_overrides...]

Examples:
    python sandbox/eval_explore.py logs/dev/feb23/some_explore_run --num-episodes 2
    python sandbox/eval_explore.py logs/dev/feb23/some_explore_run --steps 15-20
    python sandbox/eval_explore.py logs/dev/feb23/some_explore_run -- client.model_id=gpt-4o
    python sandbox/eval_explore.py logs/dev/feb23/some_explore_run --max-workers 4 -- envs.names=minihack
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml


def discover_steps(run_dir: Path) -> list[dict]:
    """Discover step directories and locate beliefs + perception files.

    Supports both layouts:
      - Single-step: step_N/beliefs.txt, step_N/perception.py
      - Two-step:    step_N/explore/beliefs.txt, step_N/explore/perception.py
      - Fallback:    step_N/baseline/beliefs.txt, step_N/baseline/perception.py
    """
    steps = []
    for item in sorted(run_dir.iterdir()):
        if not item.is_dir() or not item.name.startswith("step_"):
            continue
        try:
            step_num = int(item.name.split("_")[1])
        except (ValueError, IndexError):
            continue

        # Try each layout in priority order
        beliefs_path = None
        perception_path = None
        for subdir in [item / "explore", item / "baseline", item]:
            b = subdir / "beliefs.txt"
            p = subdir / "perception.py"
            if b.exists() and p.exists():
                beliefs_path = b
                perception_path = p
                break

        if beliefs_path is None or perception_path is None:
            print(f"  Warning: step_{step_num} has no beliefs.txt + perception.py, skipping")
            continue

        steps.append({
            "step_num": step_num,
            "beliefs_path": str(beliefs_path),
            "perception_path": str(perception_path),
        })

    steps.sort(key=lambda s: s["step_num"])
    return steps


def load_run_config(run_dir: Path) -> dict:
    """Load config.yaml from the explore run directory."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def build_hydra_overrides(run_config: dict) -> list[str]:
    """Extract key Hydra overrides from the saved run config."""
    overrides = []

    # Client settings
    client = run_config.get("client", {})
    if client.get("client_name"):
        overrides.append(f"client.client_name={client['client_name']}")
    if client.get("model_id"):
        overrides.append(f"client.model_id={client['model_id']}")

    # Agent settings
    agent = run_config.get("agent", {})
    if agent.get("type"):
        overrides.append(f"agent.type={agent['type']}")
    if agent.get("max_text_history") is not None:
        overrides.append(f"agent.max_text_history={agent['max_text_history']}")
    if agent.get("max_image_history") is not None:
        overrides.append(f"agent.max_image_history={agent['max_image_history']}")

    # Environment settings
    envs = run_config.get("envs", {})
    if envs.get("names"):
        overrides.append(f"envs.names={envs['names']}")

    # Eval settings
    eval_cfg = run_config.get("eval", {})
    if eval_cfg.get("save_trajectories") is not None:
        overrides.append(f"eval.save_trajectories={str(eval_cfg['save_trajectories']).lower()}")
    if eval_cfg.get("beliefs_in_system_prompt") is not None:
        overrides.append(f"eval.beliefs_in_system_prompt={str(eval_cfg['beliefs_in_system_prompt']).lower()}")

    return overrides


def run_step_eval(
    step: dict,
    output_dir: Path,
    num_episodes: int,
    base_overrides: list[str],
    extra_overrides: list[str],
    explore_py: str,
) -> dict:
    """Run eval for a single step. Returns result dict."""
    step_num = step["step_num"]
    step_out = output_dir / f"step_{step_num}"
    step_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, explore_py,
        "eval.mode=eval",
        f"eval.beliefs_path={step['beliefs_path']}",
        f"eval.perception_path={step['perception_path']}",
        f"eval.num_episodes={num_episodes}",
        f"eval.output_dir={step_out}",
        # Disable trajectory saving for eval runs (faster)
        "eval.save_trajectories=false",
    ]
    cmd.extend(base_overrides)
    cmd.extend(extra_overrides)

    result = {"step": step_num, "beliefs_path": step["beliefs_path"], "perception_path": step["perception_path"]}

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        result["returncode"] = proc.returncode

        if proc.returncode != 0:
            result["error"] = proc.stderr[-2000:] if proc.stderr else "unknown error"
            print(f"  Step {step_num}: FAILED (exit code {proc.returncode})")
            return result

        # Find the summary.json — Hydra changes the working dir, so the eval output
        # lands inside a timestamped subdirectory under step_out
        summary = _find_summary(step_out)
        if summary is not None:
            result["summary"] = summary
            # Quick status line
            env_name = next(iter(summary), None)
            if env_name:
                s = summary[env_name]
                print(f"  Step {step_num}: avg_prog={s.get('avg_prog', 0):.3f}, "
                      f"num_perfect={s.get('num_perfect', 0)}")
        else:
            result["error"] = "summary.json not found in output"
            print(f"  Step {step_num}: completed but summary.json not found")

    except subprocess.TimeoutExpired:
        result["error"] = "timeout (3600s)"
        print(f"  Step {step_num}: TIMEOUT")
    except Exception as e:
        result["error"] = str(e)
        print(f"  Step {step_num}: ERROR - {e}")

    return result


def _find_summary(output_dir: Path) -> dict | None:
    """Recursively find and load the first summary.json under output_dir."""
    for p in output_dir.rglob("summary.json"):
        try:
            with open(p) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
    return None


def print_summary_table(results: list[dict]):
    """Print a formatted summary table to stdout."""
    # Collect rows with data
    rows = []
    for r in sorted(results, key=lambda x: x["step"]):
        if "summary" not in r:
            rows.append({
                "step": r["step"],
                "error": r.get("error", "unknown"),
            })
            continue

        # Aggregate across environments (usually just one)
        total_prog = 0.0
        total_steps = 0.0
        total_perfect = 0
        total_solved = 0
        total_cost = 0.0
        num_envs = 0
        for env_name, stats in r["summary"].items():
            total_prog += stats.get("avg_prog", 0)
            total_steps += stats.get("avg_steps", 0)
            total_perfect += stats.get("num_perfect", 0)
            total_solved += stats.get("num_solved", 0)
            total_cost += stats.get("total_cost", 0)
            num_envs += 1

        avg_prog = total_prog / num_envs if num_envs else 0
        avg_steps = total_steps / num_envs if num_envs else 0

        rows.append({
            "step": r["step"],
            "avg_prog": avg_prog,
            "avg_steps": avg_steps,
            "num_perfect": total_perfect,
            "num_solved": total_solved,
            "total_cost": total_cost,
        })

    # Print table
    header = f"{'Step':>5}  {'Avg Prog':>9}  {'Avg Steps':>10}  {'Num Perfect':>12}  {'Num Solved':>11}  {'Total Cost':>11}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for row in rows:
        if "error" in row:
            print(f"{row['step']:>5}  {'ERROR':>9}  {row['error']}")
        else:
            print(
                f"{row['step']:>5}  "
                f"{row['avg_prog']:>9.3f}  "
                f"{row['avg_steps']:>10.1f}  "
                f"{row['num_perfect']:>12}  "
                f"{row['num_solved']:>11}  "
                f"${row['total_cost']:>10.4f}"
            )

    print("=" * len(header))


def main():
    # Split argv on "--" to separate our args from Hydra passthrough overrides
    argv = sys.argv[1:]
    if "--" in argv:
        split_idx = argv.index("--")
        our_args = argv[:split_idx]
        extra_overrides = argv[split_idx + 1:]
    else:
        our_args = argv
        extra_overrides = []

    parser = argparse.ArgumentParser(
        description="Evaluate all steps of an explore run",
        usage="python sandbox/eval_explore.py <explore_run_dir> [options] [-- hydra_overrides...]",
    )
    parser.add_argument("explore_run_dir", type=str, help="Path to the explore run directory")
    parser.add_argument("--max-workers", type=int, default=1, help="Max parallel step evals (default: 1)")
    parser.add_argument("--num-episodes", type=int, default=20, help="Episodes per step eval (default: 20)")
    parser.add_argument("--steps", type=str, default=None,
                        help="Steps to eval, e.g. '2,5,9,12,14' or '15-20' or '5-' or '-10' or '3' (default: all)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: <run_dir>/evals)")
    parser.add_argument("--explore-py", type=str, default=None, help="Path to explore.py (default: auto-detect)")
    args = parser.parse_args(our_args)

    run_dir = Path(args.explore_run_dir).resolve()
    if not run_dir.is_dir():
        print(f"Error: {run_dir} is not a directory")
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "evals"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Locate explore.py
    if args.explore_py:
        explore_py = str(Path(args.explore_py).resolve())
    else:
        # Try common locations relative to this script
        candidates = [
            Path(__file__).resolve().parent.parent / "explore.py",
            Path.cwd() / "explore.py",
        ]
        explore_py = None
        for c in candidates:
            if c.exists():
                explore_py = str(c)
                break
        if explore_py is None:
            print("Error: could not find explore.py. Use --explore-py to specify its path.")
            sys.exit(1)

    print(f"Explore run dir: {run_dir}")
    print(f"Output dir:      {output_dir}")
    print(f"explore.py:      {explore_py}")

    # Discover steps
    steps = discover_steps(run_dir)
    if not steps:
        print("No steps found with beliefs.txt + perception.py")
        sys.exit(1)

    # Filter by step range if specified
    if args.steps:
        # Support comma-separated list (e.g. "2,5,9,12,14"), ranges (e.g. "5-10"), or single step
        allowed = set()
        for token in args.steps.split(","):
            token = token.strip()
            if "-" in token:
                parts = token.split("-", 1)
                lo = int(parts[0]) if parts[0] else None
                hi = int(parts[1]) if parts[1] else None
                for s in steps:
                    if (lo is None or s["step_num"] >= lo) and (hi is None or s["step_num"] <= hi):
                        allowed.add(s["step_num"])
            else:
                allowed.add(int(token))
        steps = [s for s in steps if s["step_num"] in allowed]
        if not steps:
            print(f"No steps found matching '{args.steps}'")
            sys.exit(1)

    print(f"Found {len(steps)} steps: {[s['step_num'] for s in steps]}")

    # Load config from run dir for default overrides
    run_config = load_run_config(run_dir)
    base_overrides = build_hydra_overrides(run_config) if run_config else []
    print(f"Base overrides from config.yaml: {base_overrides}")
    if extra_overrides:
        print(f"Extra overrides: {extra_overrides}")

    # Run evals in parallel
    max_workers = args.max_workers
    print(f"\nRunning {len(steps)} evals with max_workers={max_workers}, num_episodes={args.num_episodes}")
    print("-" * 60)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_step = {}
        for step in steps:
            future = executor.submit(
                run_step_eval,
                step=step,
                output_dir=output_dir,
                num_episodes=args.num_episodes,
                base_overrides=base_overrides,
                extra_overrides=extra_overrides,
                explore_py=explore_py,
            )
            future_to_step[future] = step

        for future in as_completed(future_to_step):
            result = future.result()
            results.append(result)

    # Print summary table
    print_summary_table(results)

    # Save combined results
    eval_summary_path = output_dir / "eval_summary.json"
    with open(eval_summary_path, "w") as f:
        json.dump(sorted(results, key=lambda x: x["step"]), f, indent=2)
    print(f"\nSaved eval summary to: {eval_summary_path}")


if __name__ == "__main__":
    main()
